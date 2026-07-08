"""
Trainer — the self-play/train loop (see docs/training.md).

Each generation: play `games_per_gen` self-play games with the current network → append to a
replay buffer → train a chunk of minibatches → checkpoint → repeat. Self-play is parallelised
across `num_workers` processes (Phase B): a persistent process pool plays the generation's
games concurrently, each worker loading the current weights from a file (once per gen). The
train step stays in the main process. Samples cross the process boundary via pickle (GameState
and Action are picklable — see bindings.cpp). `num_workers=1` runs everything in-process.

Losses per developed sample (from the acting player's perspective):
  - value  : MSE(value_head, outcome z)                          — batched.
  - policy : sum over the chosen action's fired stages of the cross-entropy between the
             stage's masked softmax and its normalized MCTS visit counts (autoregressive:
             each stage conditioned on the earlier chosen sub-choices).                — per-sample.

The trunk runs once per minibatch (batched); the value head reads the batched core, and the
policy stages read per-sample slices of the same cache (no re-forward).
"""

import csv
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
sys.path.insert(0, os.path.dirname(__file__))
import polyshark  # noqa: E402

from arena import Agent, Arena, MCTSStrategy, make_heuristic_terminal_value  # noqa: E402
from mcts import (  # noqa: E402
    HeuristicEvaluator, NetworkEvaluator, HEURISTIC_VALUE_SCALE, _UNIT_STAGE_TYPES,
)
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


# --------------------------------------------------------------------------- self-play

def _make_agents(evaluator, cfg):
    def mk(name):
        return Agent(name, MCTSStrategy(
            evaluator, n_sims=cfg["n_sims"], c_puct=cfg["c_puct"],
            temperature=cfg["temperature"], temp_turns=cfg["temp_turns"],
            add_noise=cfg["add_noise"]))
    return mk("p0"), mk("p1")


def _play_one_game(evaluator, cfg, seed):
    """Play one self-play game; return (samples, per-game stats)."""
    a, b = _make_agents(evaluator, cfg)
    tvf = make_heuristic_terminal_value(cfg["heuristic_scale"])
    t0 = time.time()
    res = Arena([a, b], terminal_value_fn=tvf).play_game(
        seed=seed, max_turns=cfg["turn_limit"], collect=True)
    stats = {"seed": seed, "winner": res.winner, "reason": res.reason, "turns": res.turns,
             "v_finals": res.v_finals, "n_samples": len(res.samples), "time": time.time() - t0}
    return res.samples, stats


def _build_evaluator(weights_path, heuristic_scale):
    """Heuristic (weights_path is None) or network evaluator loaded from a weights file."""
    if weights_path is None:
        return HeuristicEvaluator(scale=heuristic_scale)
    net, policy = PolysharkNet(), PolicyHead()
    blob = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(blob["net"]); policy.load_state_dict(blob["policy"])
    return NetworkEvaluator(net, policy)


# --- worker-process side (used only when num_workers > 1) ---
_WORKER = {}


def _worker_init(cfg):
    torch.set_num_threads(1)                     # avoid BLAS oversubscription across workers
    _WORKER.clear()
    _WORKER["cfg"] = cfg
    _WORKER["gen"] = None
    _WORKER["evaluator"] = None


def _worker_play(task):
    seed, gen, weights_path = task
    if _WORKER["gen"] != gen:                     # (re)load this gen's weights once per worker
        _WORKER["evaluator"] = _build_evaluator(weights_path, _WORKER["cfg"]["heuristic_scale"])
        _WORKER["gen"] = gen
    return _play_one_game(_WORKER["evaluator"], _WORKER["cfg"], seed)


# --------------------------------------------------------------------------- loop

def run_training(
    n_gens=5, games_per_gen=8, train_steps_per_gen=200, minibatch=32,
    buffer_capacity=20000, turn_limit=30, n_sims=60, c_puct=1.5,
    temperature=1.0, temp_turns=6, add_noise=True, lr=1e-3,
    base_seed=0, bootstrap_gen0=True, heuristic_scale=HEURISTIC_VALUE_SCALE,
    num_workers=1, ckpt_dir=None, net=None, policy=None, log=print,
):
    """Self-play/train loop with parallel self-play. Returns (history, net, policy).

    num_workers > 1 plays each generation's games across a process pool; =1 runs in-process.
    Writes per-game and per-generation logs via `log`, and a metrics.csv into ckpt_dir."""
    net = net or PolysharkNet()
    policy = policy or PolicyHead()
    opt = torch.optim.Adam(list(net.parameters()) + list(policy.parameters()), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    cfg = dict(n_sims=n_sims, c_puct=c_puct, temperature=temperature, temp_turns=temp_turns,
               add_noise=add_noise, turn_limit=turn_limit, heuristic_scale=heuristic_scale)

    # Weights file that parallel workers read each network generation.
    weights_path = (os.path.join(ckpt_dir, "_worker_weights.pt") if ckpt_dir
                    else os.path.join(tempfile.gettempdir(), f"polyshark_w_{os.getpid()}.pt"))

    # metrics.csv (value/policy loss across generations) + a live handle.
    csv_file = writer = None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        csv_file = open(os.path.join(ckpt_dir, "metrics.csv"), "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["gen", "eval", "value_loss", "policy_loss", "buffer",
                         "decisive", "selfplay_s", "train_s", "total_s"])

    pool = None
    if num_workers > 1:
        pool = mp.get_context("spawn").Pool(num_workers, initializer=_worker_init, initargs=(cfg,))

    history = []
    try:
        for gen in range(n_gens):
            gen_t0 = time.time()
            use_heuristic = (gen == 0 and bootstrap_gen0)

            # 1. SELF-PLAY (parallel across workers, or in-process).
            sp_t0 = time.time()
            if not use_heuristic:
                torch.save({"net": net.state_dict(), "policy": policy.state_dict()}, weights_path)
            gen_weights = None if use_heuristic else weights_path
            seeds = [base_seed + gen * games_per_gen + g for g in range(games_per_gen)]

            if pool is not None:
                results = pool.map(_worker_play, [(s, gen, gen_weights) for s in seeds])
            else:
                ev = _build_evaluator(gen_weights, heuristic_scale)
                results = [_play_one_game(ev, cfg, s) for s in seeds]
            selfplay_s = time.time() - sp_t0

            decisive = 0
            for samples, st in results:
                buffer.extend(samples)
                decisive += (st["winner"] >= 0)
                who = f"p{st['winner']} wins" if st["winner"] >= 0 else "turn cap "
                log(f"  game seed={st['seed']:<5} {who:<9} v=({st['v_finals'][0]:+.2f},"
                    f"{st['v_finals'][1]:+.2f}) turns={st['turns']:<3} "
                    f"samples={st['n_samples']:<4} {st['time']:.1f}s")

            # 2. TRAIN on random minibatches from the buffer.
            net.train(); policy.train()
            tr_t0 = time.time()
            vlosses, plosses = [], []
            for _ in range(train_steps_per_gen):
                mb = buffer.sample(minibatch)
                if not mb:
                    break
                vl, pl = train_step(net, policy, opt, mb)
                vlosses.append(vl); plosses.append(pl)
            train_s = time.time() - tr_t0
            vloss = float(np.mean(vlosses)) if vlosses else float("nan")
            ploss = float(np.mean(plosses)) if plosses else float("nan")

            # 3. CHECKPOINT.
            ckpt_path = None
            if ckpt_dir:
                ckpt_path = os.path.join(ckpt_dir, f"gen{gen:03d}.pt")
                torch.save({"net": net.state_dict(), "policy": policy.state_dict(), "gen": gen},
                           ckpt_path)

            total_s = time.time() - gen_t0
            evname = "heuristic" if use_heuristic else "network"
            log(f"gen {gen:2d} | eval={evname:<9} buffer={len(buffer):<6} "
                f"decisive={decisive}/{games_per_gen} | value_loss={vloss:.4f} "
                f"policy_loss={ploss:.4f} | selfplay={selfplay_s:.0f}s train={train_s:.0f}s "
                f"total={total_s:.0f}s")
            if writer:
                writer.writerow([gen, evname, f"{vloss:.6f}", f"{ploss:.6f}", len(buffer),
                                 decisive, f"{selfplay_s:.2f}", f"{train_s:.2f}", f"{total_s:.2f}"])
                csv_file.flush()

            history.append({
                "gen": gen, "eval": evname, "buffer": len(buffer), "decisive": decisive,
                "value_loss": vloss, "policy_loss": ploss, "selfplay_s": selfplay_s,
                "train_s": train_s, "total_s": total_s, "ckpt": ckpt_path,
            })
    finally:
        if pool is not None:
            pool.close(); pool.join()
        if csv_file:
            csv_file.close()

    return history, net, policy
