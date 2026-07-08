"""
Phase A trainer — the alternating self-play/train loop (see docs/training.md).

One process: play a chunk of self-play games with the current network → append to a replay
buffer → train a chunk of minibatches → checkpoint → repeat. (The concurrent two-process
version is a later throughput optimization; the data and losses are identical.)

Losses per developed sample (from the acting player's perspective):
  - value  : MSE(value_head, outcome z)                          — batched.
  - policy : sum over the chosen action's fired stages of the cross-entropy between the
             stage's masked softmax and its normalized MCTS visit counts (autoregressive:
             each stage conditioned on the earlier chosen sub-choices).                — per-sample.

The trunk runs once per minibatch (batched); the value head reads the batched core, and the
policy stages read per-sample slices of the same cache (no re-forward).
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
sys.path.insert(0, os.path.dirname(__file__))
import polyshark  # noqa: E402

from arena import Agent, Arena, MCTSStrategy  # noqa: E402
from mcts import HeuristicEvaluator, NetworkEvaluator, _UNIT_STAGE_TYPES  # noqa: E402
from model import PolysharkNet  # noqa: E402
from policy import PolicyHead, T_MOVE, T_ATTACK, T_TRAIN, T_RESEARCH  # noqa: E402
from factored import FactoredActions  # noqa: E402
from features import collate, encode_entities  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402

_MODEL_ARG_ORDER = [
    "unit_types", "unit_feats", "unit_mask", "unit_tiles",
    "city_feats", "city_mask", "city_tiles", "board", "globals",
]
_UPGRADE = polyshark.ActionType.UpgradeCity


# --------------------------------------------------------------------------- losses

def _stage_logits(policy, stage, path, fa, core, unit_tok, city_tok, feature_map):
    """Masked logits for one factored stage (grad-enabled). Mirrors NetworkEvaluator.priors
    but keeps the graph. Slices come from the batched trunk cache for one sample."""
    if stage == "type":
        mask = torch.tensor([fa.type_mask()], dtype=torch.bool)
        return policy.type_logits(core, mask)
    if stage == "entity":
        t = path[0]
        toks = unit_tok if t in _UNIT_STAGE_TYPES else city_tok
        m = fa.entity_mask(t)
        mask = torch.tensor([m], dtype=torch.bool)
        return policy.entity_logits(core, torch.tensor([t]), toks[:, :len(m)], mask)
    if stage == "tile":
        t = path[0]
        if t in (T_MOVE, T_ATTACK):
            emb = unit_tok[:, path[1]]
            mask = torch.tensor([fa.tile_mask(t, path[1])], dtype=torch.bool)
        else:
            emb = None
            mask = torch.tensor([fa.tile_mask(t)], dtype=torch.bool)
        return policy.tile_logits(core, torch.tensor([t]), emb, feature_map, mask)
    if stage == "train_unit":
        emb = city_tok[:, path[1]]
        mask = torch.tensor([fa.train_unit_mask(path[1])], dtype=torch.bool)
        return policy.train_unit_logits(core, torch.tensor([T_TRAIN]), emb, mask)
    if stage == "research":
        mask = torch.tensor([fa.research_mask()], dtype=torch.bool)
        return policy.research_logits(core, torch.tensor([T_RESEARCH]), mask)
    raise ValueError(f"unknown stage {stage}")


def _policy_loss_for_sample(policy, s, cache, b):
    """Summed cross-entropy over the fired stages of sample `s`'s chosen action, using row
    `b` of the batched trunk cache. Returns (loss_tensor, n_stages). The upgrade modal is
    skipped (its head isn't wired into the search yet — see docs/policy_head.md)."""
    if s.targets is None or int(s.action.type) == int(_UPGRADE):
        return None, 0
    # Rebuild FactoredActions with the same (default) encoding collate used, so entity slot
    # indices line up with the batched tokens. Acting player == current player (turn-local),
    # so default me/fog matches how the sample was recorded.
    fa = FactoredActions(s.state, encoded=encode_entities(s.state))
    path = fa._path_of(s.action)
    if path is None:
        return None, 0

    core = cache.core[b:b + 1]
    fmap = cache.feature_map[b:b + 1]
    utok = cache.unit_tok[b:b + 1]
    ctok = cache.city_tok[b:b + 1]

    loss, n = None, 0
    for stage_name, visits in s.targets:
        if stage_name == "upgrade":
            continue
        total = sum(visits.values())
        if total <= 0:
            continue
        logits = _stage_logits(policy, stage_name, path, fa, core, utok, ctok, fmap)
        logq = F.log_softmax(logits, dim=-1)          # [1, K]
        ce = -sum((v / total) * logq[0, c] for c, v in visits.items())
        loss = ce if loss is None else loss + ce
        n += 1
    return loss, n


def train_step(net, policy, opt, batch):
    """One optimizer step over a minibatch of Samples. Returns (value_loss, policy_loss)."""
    states = [s.state for s in batch]
    tensors = {k: torch.from_numpy(v) for k, v in collate(states).items()}
    cache = net.trunk(*(tensors[k] for k in _MODEL_ARG_ORDER))

    # Value: batched MSE against the outcome z.
    value = net.value(cache.core).squeeze(-1)                     # [B]
    outcomes = torch.tensor([s.outcome for s in batch], dtype=torch.float32)
    value_loss = F.mse_loss(value, outcomes)

    # Policy: per-sample autoregressive CE, reading slices of the same cache.
    policy_terms, n_stages = None, 0
    for b, s in enumerate(batch):
        term, n = _policy_loss_for_sample(policy, s, cache, b)
        if term is not None:
            policy_terms = term if policy_terms is None else policy_terms + term
            n_stages += n
    policy_loss = policy_terms / n_stages if n_stages else torch.zeros(())

    loss = value_loss + policy_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    pl = policy_loss.item() if n_stages else 0.0
    return value_loss.item(), pl


# --------------------------------------------------------------------------- loop

def _self_play_evaluator(net, policy, gen, bootstrap_gen0):
    """Which evaluator drives self-play this generation. Gen 0 can bootstrap on the heuristic
    (meaningful value + fog-honest visit targets) rather than the random-init net."""
    if gen == 0 and bootstrap_gen0:
        return HeuristicEvaluator()
    net.eval(); policy.eval()
    return NetworkEvaluator(net, policy)


def run_training(
    n_gens=5, games_per_gen=8, train_steps_per_gen=200, minibatch=32,
    buffer_capacity=20000, turn_limit=30, n_sims=60, c_puct=1.5,
    temperature=1.0, temp_turns=6, add_noise=True, lr=1e-3,
    base_seed=0, bootstrap_gen0=True, ckpt_dir=None, net=None, policy=None,
    log=print,
):
    """Phase A alternating loop. Returns per-generation stats dicts and the trained modules."""
    net = net or PolysharkNet()
    policy = policy or PolicyHead()
    opt = torch.optim.Adam(list(net.parameters()) + list(policy.parameters()), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    history = []

    for gen in range(n_gens):
        # 1. SELF-PLAY with the current weights -> samples.
        ev = _self_play_evaluator(net, policy, gen, bootstrap_gen0)
        mk = lambda name: Agent(name, MCTSStrategy(  # noqa: E731
            ev, n_sims=n_sims, c_puct=c_puct,
            temperature=temperature, temp_turns=temp_turns, add_noise=add_noise))
        a, b = mk("p0"), mk("p1")

        decisive, added = 0, 0
        for g in range(games_per_gen):
            res = Arena([a, b]).play_game(
                seed=base_seed + gen * games_per_gen + g, max_turns=turn_limit, collect=True)
            buffer.extend(res.samples)
            added += len(res.samples)
            decisive += (res.winner >= 0)

        # 2. TRAIN on random minibatches from the buffer.
        net.train(); policy.train()
        vlosses, plosses = [], []
        for _ in range(train_steps_per_gen):
            mb = buffer.sample(minibatch)
            if not mb:
                break
            vl, pl = train_step(net, policy, opt, mb)
            vlosses.append(vl); plosses.append(pl)

        # 3. CHECKPOINT.
        ckpt_path = None
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"gen{gen:03d}.pt")
            torch.save({"net": net.state_dict(), "policy": policy.state_dict(), "gen": gen},
                       ckpt_path)

        stat = {
            "gen": gen, "eval": type(ev).__name__, "buffer": len(buffer),
            "samples_added": added, "decisive": decisive,
            "value_loss": float(np.mean(vlosses)) if vlosses else float("nan"),
            "policy_loss": float(np.mean(plosses)) if plosses else float("nan"),
            "ckpt": ckpt_path,
        }
        history.append(stat)
        log(f"gen {gen:2d} | eval={stat['eval']:<18} buffer={stat['buffer']:<6} "
            f"decisive={decisive}/{games_per_gen} | "
            f"value_loss={stat['value_loss']:.4f} policy_loss={stat['policy_loss']:.4f}"
            + (f" | {ckpt_path}" if ckpt_path else ""))

    return history, net, policy
