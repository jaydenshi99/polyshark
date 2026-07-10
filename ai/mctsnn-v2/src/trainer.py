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
import math
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
sys.path.insert(0, os.path.dirname(__file__))
import polyshark  # noqa: E402

from arena import (  # noqa: E402
    Agent, Arena, MCTSStrategy, make_heuristic_terminal_value, make_winner_terminal_value,
)
from mcts import (  # noqa: E402
    HeuristicEvaluator, NetworkEvaluator, HEURISTIC_VALUE_SCALE, _UNIT_STAGE_TYPES,
)
from model import PolysharkNet  # noqa: E402
from policy import PolicyHead, T_MOVE, T_ATTACK, T_TRAIN, T_RESEARCH  # noqa: E402
from factored import FactoredActions  # noqa: E402
from features import collate, d8_transform, encode_entities  # noqa: E402
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
        # Forced stage (one legal choice): the mask already decides — CE against a
        # one-hot-by-construction target is pure marginal drift, zero information.
        # `visits` covers every legal choice (zero-visit ones included), so its length
        # is the stage's legal-choice count.
        if len(visits) <= 1:
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


def train_step(net, policy, opt, batch, value_symmetry=False):
    """One optimizer step over a minibatch of Samples. Returns (value_loss, policy_loss);
    value_loss is NaN if no sample in the batch is value-eligible (see _mark_value_positions).

    value_symmetry=True trains the value head on a random D8 transform of the eligible
    rows (targets are rotation/flip-invariant) — ~8x effective value data and it deletes
    the map-fingerprint memorization channel. Policy stays in the original orientation
    (its targets live in board coordinates)."""
    states = [s.state for s in batch]
    raw = collate(states)
    tensors = {k: torch.from_numpy(v) for k, v in raw.items()}
    cache = net.trunk(*(tensors[k] for k in _MODEL_ARG_ORDER))

    # Value: batched MSE against the outcome z, over value-eligible samples only —
    # within-game states share one label, so the trainer subsamples a few per game.
    outcomes = torch.tensor([s.outcome for s in batch], dtype=torch.float32)
    vsel = torch.tensor([getattr(s, "train_value", True) for s in batch], dtype=torch.bool)
    if not vsel.any():
        value_loss = torch.zeros(())
    elif value_symmetry:
        rows = np.flatnonzero(vsel.numpy())
        aug = {k: torch.from_numpy(np.ascontiguousarray(v))
               for k, v in d8_transform(raw, random.randrange(8), rows=rows).items()}
        vcache = net.trunk(*(aug[k] for k in _MODEL_ARG_ORDER))
        value_loss = F.mse_loss(net.value(vcache.core).squeeze(-1), outcomes[vsel])
    else:
        value = net.value(cache.core).squeeze(-1)                 # [B]
        value_loss = F.mse_loss(value[vsel], outcomes[vsel])

    # Policy: per-sample autoregressive CE, reading slices of the same cache.
    policy_terms, n_stages = None, 0
    for b, s in enumerate(batch):
        term, n = _policy_loss_for_sample(policy, s, cache, b)
        if term is not None:
            policy_terms = term if policy_terms is None else policy_terms + term
            n_stages += n
    policy_loss = policy_terms / n_stages if n_stages else torch.zeros(())

    loss = value_loss + policy_loss
    if loss.requires_grad:                # can be all-constant (no value sample, no stage)
        opt.zero_grad()
        loss.backward()
        opt.step()
    pl = policy_loss.item() if n_stages else 0.0
    vl = value_loss.item() if vsel.any() else float("nan")
    return vl, pl


def make_optimizer(net, policy, lr, weight_decay):
    """AdamW over net + policy. Decay only matrix params; biases and norm scales are
    exempt (standard practice — decaying them hurts without regularizing anything)."""
    params = list(net.parameters()) + list(policy.parameters())
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def _mark_value_positions(samples, per_game, rng):
    """AlphaGo-style value subsampling: keep only a few positions of one game
    value-eligible (split evenly across the two players), the rest train policy only.
    All of a game's states share a single outcome label, so training the value head on
    every state is ~90 gradient hits on one memorizable data point (see
    docs/endturn_collapse.md). per_game <= 0 disables (everything stays eligible).

    The budget scales with game length — min(per_game, ~1/4 of the game's samples;
    relaxed from 1/8 once mixed targets gave every state its own label) —
    so short curriculum games (see turn_cap_for_gen) keep the same *fraction* eligible
    instead of quietly marking most of the game (at 130+ samples/game, len//8 ≈ 16,
    matching the historical per_game default)."""
    if per_game <= 0 or not samples:
        return
    budget = min(per_game, max(4, len(samples) // 4))
    by_player = {}
    for s in samples:
        s.train_value = False
        by_player.setdefault(s.player, []).append(s)
    k = max(budget // len(by_player), 1)
    for group in by_player.values():
        keep = group if len(group) <= k else rng.sample(group, k)
        for s in keep:
            s.train_value = True


def _mix_search_values(samples, weight):
    """Mixed value targets: outcome <- (1-w)·z + w·v̂ where v̂ is the search root value
    recorded at the decision (Sample.search_value, acting player's frame).

    Pure outcome labels give every state of a game the SAME target — no within-game
    contrast, so credit for e.g. approach moves only accrues across games (slow, noisy;
    see docs/endturn_collapse.md). v̂ differs per state and encodes what the search
    *found* there (including unplayed lines), which also severs the "unplayed states
    drift negative -> played even less" oscillation. w is annealed from 0 (early nets'
    search values are noise) up to its configured maximum. Val samples are never mixed —
    validation must score against the true outcome."""
    if weight <= 0:
        return
    for s in samples:
        if s.outcome is not None and getattr(s, "search_value", None) is not None:
            s.outcome = (1.0 - weight) * s.outcome + weight * s.search_value


def value_metrics(net, samples, chunk=256):
    """Value-head validation metrics over `samples` (no grad, eval mode): (mse, sign_acc).

    - mse      : high train/val gap = memorizing games, not evaluating positions.
    - sign_acc : fraction of DECISIVE (±1-labelled) samples where sign(V) == sign(z).
      Far more sensitive early than MSE: with ±1 labels, a timid-but-correct head
      (|V|~0.1, right sign 60% of the time) still scores ~0.97 MSE — indistinguishable
      from no skill — while sign_acc reads 0.60 immediately. 0.5 = coin flip."""
    if not samples:
        return float("nan"), float("nan")
    was_training = net.training
    net.eval()
    total, n, sign_hits, n_decisive = 0.0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(samples), chunk):
            part = samples[i:i + chunk]
            tensors = {k: torch.from_numpy(v)
                       for k, v in collate([s.state for s in part]).items()}
            cache = net.trunk(*(tensors[k] for k in _MODEL_ARG_ORDER))
            v = net.value(cache.core).squeeze(-1)
            z = torch.tensor([s.outcome for s in part], dtype=torch.float32)
            total += F.mse_loss(v, z, reduction="sum").item()
            n += len(part)
            decisive = z.abs() > 0.5
            sign_hits += int(((v > 0) == (z > 0))[decisive].sum().item())
            n_decisive += int(decisive.sum().item())
    if was_training:
        net.train()
    mse = total / n
    acc = sign_hits / n_decisive if n_decisive else float("nan")
    return mse, acc


# --------------------------------------------------------------------------- self-play

def _make_agents(evaluator, cfg, turn_cap=None):
    # temp_frac scales the exploratory opening with the curriculum cap (e.g. 0.2 ->
    # temperature on for the first 20% of the game, greedy after). A fixed temp_turns
    # near the cap means almost the whole game is sampled play, and the ±1 winner label
    # then mostly records sampling luck — poison for the value head. None = fixed turns.
    if cfg.get("temp_frac") is not None and turn_cap:
        temp_turns = max(1, round(cfg["temp_frac"] * turn_cap))
    else:
        temp_turns = cfg["temp_turns"]

    def mk(name):
        return Agent(name, MCTSStrategy(
            evaluator, n_sims=cfg["n_sims"], c_puct=cfg["c_puct"],
            temperature=cfg["temperature"], temp_turns=temp_turns,
            add_noise=cfg["add_noise"]))
    return mk("p0"), mk("p1")


def _make_terminal_value(cfg):
    """Turn-cap labeller from config: ±1 winner-by-margin (default; see
    docs/endturn_collapse.md actionable #1) or the legacy tanh margin."""
    if cfg.get("turn_cap_winner", True):
        return make_winner_terminal_value(cfg.get("winner_dead_zone", 1.0))
    return make_heuristic_terminal_value(cfg["heuristic_scale"])


def turn_cap_for_gen(gen, turn_limit, turn_cap_start=None, turn_cap_grow=1.0):
    """Curriculum horizon: the turn cap for generation `gen`. With turn_cap_start set,
    the cap starts there and grows `turn_cap_grow` turns per gen up to `turn_limit` —
    short horizons early make each action a large share of the outcome (strong value
    gradient), then the horizon stretches as play earns it. None = constant turn_limit."""
    if turn_cap_start is None:
        return turn_limit
    return min(turn_limit, int(turn_cap_start + gen * turn_cap_grow))


def _play_one_game(evaluator, cfg, seed, turn_cap=None):
    """Play one self-play game; return (samples, per-game stats)."""
    a, b = _make_agents(evaluator, cfg, turn_cap)
    tvf = _make_terminal_value(cfg)
    t0 = time.time()
    res = Arena([a, b], terminal_value_fn=tvf).play_game(
        seed=seed, max_turns=turn_cap or cfg["turn_limit"], collect=True)
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
    seed, gen, weights_path, turn_cap = task
    if _WORKER["gen"] != gen:                     # (re)load this gen's weights once per worker
        _WORKER["evaluator"] = _build_evaluator(
            weights_path, _WORKER["cfg"]["gen0_search_scale"])
        _WORKER["gen"] = gen
    return _play_one_game(_WORKER["evaluator"], _WORKER["cfg"], seed, turn_cap)


# --------------------------------------------------------------------------- loop

def run_training(
    n_gens=5, games_per_gen=8, train_steps_per_gen=200, minibatch=32,
    buffer_capacity=20000, turn_limit=30, n_sims=60, c_puct=1.5,
    temperature=1.0, temp_turns=6, temp_frac=None, add_noise=True, lr=1e-3, weight_decay=1e-4,
    value_samples_per_game=16, val_games=8, val_seed_base=1_000_000,
    turn_cap_winner=True, winner_dead_zone=1.0, gen0_search_scale=1.0,
    turn_cap_start=None, turn_cap_grow=1.0,
    search_value_weight=0.0, search_value_anneal_gens=10, lr_final=None,
    value_symmetry=False,
    base_seed=0, bootstrap_gen0=True, heuristic_scale=HEURISTIC_VALUE_SCALE,
    num_workers=1, ckpt_dir=None, net=None, policy=None, log=print,
):
    """Self-play/train loop with parallel self-play. Returns (history, net, policy).

    num_workers > 1 plays each generation's games across a process pool; =1 runs in-process.
    Writes per-game and per-generation logs via `log`, and a metrics.csv into ckpt_dir.

    Value-overfit guards (see docs/endturn_collapse.md):
      - value_samples_per_game : only this many positions per game keep their value label
        for training (<=0 = all). Policy targets are unaffected.
      - weight_decay           : AdamW decay on matrix params (biases/norms exempt).
      - val_games              : per gen, this many extra self-play games on held-out seeds
        (val_seed_base + ...) are played, kept OUT of the buffer, and scored after training
        (val_value_loss column). Train/val gap = memorization meter. 0 disables.

    Turn-cap labelling (actionable #1):
      - turn_cap_winner        : True (default) = ±1 by heuristic-margin sign at the cap
        (winner_dead_zone points -> 0); the value head estimates win probability and
        mutual passing is no longer label-neutral. False = legacy tanh(heuristic_scale·m).
      - gen0_search_scale      : tanh scale of the gen-0 bootstrap SEARCH evaluator —
        deliberately sharper than any label squash so bootstrap play is decisive.

    Horizon curriculum:
      - turn_cap_start/grow    : if start is set, gen g's games are capped at
        min(turn_limit, start + g*grow) turns. Short horizons early = each action is a
        big share of the outcome (strong value gradient); the cap then stretches toward
        turn_limit as generations pass. None = constant turn_limit.

    Mixed value targets (see _mix_search_values):
      - search_value_weight    : w in target = (1-w)·z + w·v̂ (v̂ = search root value per
        state). Per-state credit + oscillation damping. 0 disables (pure outcomes).
      - search_value_anneal_gens : w ramps linearly from 0 to its maximum over this many
        gens (early nets' search values are noise; don't bootstrap into them).

    LR schedule:
      - lr_final : if set, cosine-decay the learning rate from `lr` (gen 0) to `lr_final`
        (last gen). Damps late-run value sloshing (the dV oscillation is partly optimizer
        noise: 200 steps at 1e-3 on batch-32 can swing value regions each gen). None =
        constant `lr`."""
    net = net or PolysharkNet()
    policy = policy or PolicyHead()
    opt = make_optimizer(net, policy, lr, weight_decay)
    buffer = ReplayBuffer(buffer_capacity)
    cfg = dict(n_sims=n_sims, c_puct=c_puct, temperature=temperature, temp_turns=temp_turns,
               temp_frac=temp_frac, add_noise=add_noise, turn_limit=turn_limit,
               heuristic_scale=heuristic_scale, turn_cap_winner=turn_cap_winner,
               winner_dead_zone=winner_dead_zone, gen0_search_scale=gen0_search_scale)

    # Weights file that parallel workers read each network generation.
    weights_path = (os.path.join(ckpt_dir, "_worker_weights.pt") if ckpt_dir
                    else os.path.join(tempfile.gettempdir(), f"polyshark_w_{os.getpid()}.pt"))

    # metrics.csv (value/policy loss across generations) + a live handle.
    csv_file = writer = None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        csv_file = open(os.path.join(ckpt_dir, "metrics.csv"), "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["gen", "eval", "turn_cap", "mix_w", "value_loss",
                         "val_value_loss", "val_sign_acc", "policy_loss", "buffer",
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
            # Held-out validation games: same evaluator/config, disjoint seed range, never
            # enter the buffer — scored after training as the memorization meter.
            val_seeds = [val_seed_base + gen * val_games + g for g in range(val_games)]

            # Horizon curriculum: this generation's turn cap.
            cap = turn_cap_for_gen(gen, turn_limit, turn_cap_start, turn_cap_grow)
            # Mixed-target weight, annealed 0 -> max over the first anneal_gens.
            mix_w = search_value_weight * min(1.0, gen / max(search_value_anneal_gens, 1))
            # Cosine LR decay lr -> lr_final across the run (None = constant).
            if lr_final is not None and n_gens > 1:
                cur_lr = lr_final + 0.5 * (lr - lr_final) * (
                    1 + math.cos(math.pi * gen / (n_gens - 1)))
            else:
                cur_lr = lr
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            # Stream results as games finish (imap_unordered) so progress is visible
            # during the long self-play phase; training vs val games are told apart by
            # their disjoint seed ranges.
            tasks = [(s, gen, gen_weights, cap) for s in seeds + val_seeds]
            if pool is not None:
                result_iter = pool.imap_unordered(_worker_play, tasks)
            else:
                ev = _build_evaluator(gen_weights, gen0_search_scale)
                result_iter = (_play_one_game(ev, cfg, s, cap) for s, _, _, cap in tasks)

            val_seed_set = set(val_seeds)
            val_samples, decisive = [], 0
            for samples, st in result_iter:
                is_val = st["seed"] in val_seed_set
                if is_val:
                    val_samples.extend(samples)          # held out: never mixed, never buffered
                else:
                    _mark_value_positions(samples, value_samples_per_game,
                                          random.Random(st["seed"]))
                    _mix_search_values(samples, mix_w)
                    buffer.extend(samples)
                    decisive += (st["winner"] >= 0)
                who = f"p{st['winner']} wins" if st["winner"] >= 0 else "turn cap "
                log(f"  {'val ' if is_val else 'game'} seed={st['seed']:<7} {who:<9} "
                    f"v=({st['v_finals'][0]:+.2f},{st['v_finals'][1]:+.2f}) "
                    f"turns={st['turns']:<3} samples={st['n_samples']:<4} {st['time']:.1f}s")
            selfplay_s = time.time() - sp_t0

            # 2. TRAIN on random minibatches from the buffer. Steps are throttled to at
            # most ~2 epochs over the current buffer so a small buffer (short curriculum
            # games, early gens) isn't hammered 5-10x per gen — that reuse is how the
            # value head memorizes. At full buffer the configured step count applies.
            steps = min(train_steps_per_gen, max(1, (2 * len(buffer)) // minibatch))
            net.train(); policy.train()
            tr_t0 = time.time()
            vlosses, plosses = [], []
            for _ in range(steps):
                mb = buffer.sample(minibatch)
                if not mb:
                    break
                vl, pl = train_step(net, policy, opt, mb, value_symmetry=value_symmetry)
                if not math.isnan(vl):        # batch may hold no value-eligible sample
                    vlosses.append(vl)
                plosses.append(pl)
            train_s = time.time() - tr_t0
            vloss = float(np.mean(vlosses)) if vlosses else float("nan")
            ploss = float(np.mean(plosses)) if plosses else float("nan")
            val_vloss, val_sign_acc = value_metrics(net, val_samples)

            # 3. CHECKPOINT.
            ckpt_path = None
            if ckpt_dir:
                ckpt_path = os.path.join(ckpt_dir, f"gen{gen:03d}.pt")
                torch.save({"net": net.state_dict(), "policy": policy.state_dict(), "gen": gen},
                           ckpt_path)

            total_s = time.time() - gen_t0
            evname = "heuristic" if use_heuristic else "network"
            log(f"gen {gen:2d} | eval={evname:<9} cap={cap:<3} mix={mix_w:.2f} lr={cur_lr:.1e} "
                f"buffer={len(buffer):<6} "
                f"decisive={decisive}/{games_per_gen} | value_loss={vloss:.4f} "
                f"val={val_vloss:.4f} val_sign={val_sign_acc:.2f} policy_loss={ploss:.4f} | "
                f"selfplay={selfplay_s:.0f}s train={train_s:.0f}s total={total_s:.0f}s")
            if writer:
                writer.writerow([gen, evname, cap, f"{mix_w:.3f}", f"{vloss:.6f}",
                                 f"{val_vloss:.6f}", f"{val_sign_acc:.4f}",
                                 f"{ploss:.6f}", len(buffer), decisive,
                                 f"{selfplay_s:.2f}", f"{train_s:.2f}", f"{total_s:.2f}"])
                csv_file.flush()

            history.append({
                "gen": gen, "eval": evname, "turn_cap": cap, "buffer": len(buffer),
                "decisive": decisive, "value_loss": vloss, "val_value_loss": val_vloss,
                "val_sign_acc": val_sign_acc,
                "policy_loss": ploss, "selfplay_s": selfplay_s, "train_s": train_s,
                "total_s": total_s, "ckpt": ckpt_path,
            })
    finally:
        if pool is not None:
            pool.close(); pool.join()
        if csv_file:
            csv_file.close()

    return history, net, policy
