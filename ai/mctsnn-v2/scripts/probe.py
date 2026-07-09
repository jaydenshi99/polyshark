"""
Checkpoint probe — the value/prior/search diagnostics from docs/endturn_collapse.md.

Give it one or more checkpoints and it reports, per probe state:

  1. V(s)        — value-head calibration (a fresh symmetric state should read ~0;
                   large |V| on an unseen seed = memorization).
  2. dV per type — V(s') - V(s) for every legal action, applied one deep. This is the
                   Q-gap the search feeds on; |dV| ~ 0.01 = no usable value gradient
                   (end_turn only needs a tie to win on prior).
  3. type priors — the policy head's masked softmax at the root; what breaks ties when
                   Q is flat. Watch P(end) creep across generations.
  4. search dump — a real MCTS search: root visits / Q / prior per type + the action it
                   commits. Shows how 1-3 combine into behaviour (flat Q -> visits track
                   priors -> argmax follows prior).

Usage (repo root, venv active):

    python ai/mctsnn-v2/scripts/probe.py <ckpt.pt> [<ckpt2.pt> ...]
        [--seeds 999 1234]   probe map seeds (default: 999; never train on these)
        [--sims 200]         search budget for measurement 4
        [--midgame 8]        also probe a developed state (heuristic self-play advances
                             this many turns first; 0 = fresh state only)
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))               # ai/mctsnn-v2/scripts
_PKG  = os.path.dirname(_HERE)                                    # ai/mctsnn-v2
_ROOT = os.path.dirname(os.path.dirname(_PKG))                    # repo root
sys.path.insert(0, os.path.join(_PKG, "src"))
sys.path.insert(0, os.path.join(_ROOT, "build", "bindings"))

import polyshark  # noqa: E402
from arena import Agent, Arena, MCTSStrategy  # noqa: E402
from factored import FactoredActions  # noqa: E402
from features import visible_snapshot  # noqa: E402
from mcts import MCTS, HeuristicEvaluator, NetworkEvaluator  # noqa: E402
from policy import N_TYPES, T_END  # noqa: E402

TYPE_NAMES = ["move", "attack", "harvest", "capture", "train", "research", "recover", "end"]


def load_evaluator(path):
    import torch
    from model import PolysharkNet
    from policy import PolicyHead
    net, pol = PolysharkNet(), PolicyHead()
    blob = torch.load(path, map_location="cpu")
    try:
        net.load_state_dict(blob["net"])
        pol.load_state_dict(blob["policy"])
    except RuntimeError as e:
        raise SystemExit(
            f"checkpoint {path} doesn't match the current model "
            f"(pre-globals-[21] runs can't load with current code):\n  {e}")
    net.eval(); pol.eval()
    return NetworkEvaluator(net, pol)


def midgame_state(seed, turns):
    """Advance a fresh game `turns` turns with sharp heuristic self-play (no torch)."""
    ev = HeuristicEvaluator(scale=1.0)
    a = Agent("h0", MCTSStrategy(ev, n_sims=60, add_noise=False, temperature=0.0))
    b = Agent("h1", MCTSStrategy(ev, n_sims=60, add_noise=False, temperature=0.0))
    res = Arena([a, b]).play_game(seed=seed, max_turns=turns, collect=True)
    return res.samples[-1].state if res.samples else None


def probe_state(ev, state, n_sims, label):
    me = state.current_player()
    vis = visible_snapshot(state, me)
    ev.begin_search(me, vis)
    v_root, ctx = ev.evaluate(state)
    fa = FactoredActions(state, encoded=ctx.enc, root_visible=vis)

    # 3. type priors
    choices = [i for i, m in enumerate(fa.type_mask()) if m]
    pri = dict(zip(choices, ev.priors(ctx, "type", (), fa, choices)))

    # 2. dV per action, grouped by type
    deltas = {}
    for path, act in fa.by_path.items():
        if path[0] == T_END:
            continue
        v2, _ = ev.evaluate(state.apply_action(act))
        deltas.setdefault(path[0], []).append(v2 - v_root)

    print(f"  [{label}] turn={state.get_turn()} player={me}")
    pstr = "  ".join(f"{TYPE_NAMES[c]}={p:.3f}" for c, p in sorted(pri.items()))
    print(f"    V(s) = {v_root:+.4f}      priors: {pstr}")
    for t, ds in sorted(deltas.items()):
        ds = np.array(ds)
        print(f"    dV {TYPE_NAMES[t]:<9} mean={ds.mean():+.4f}  max={ds.max():+.4f}  "
              f"min={ds.min():+.4f}  n={len(ds)}")

    # 4. real search
    mcts = MCTS(ev, c_puct=1.5, add_noise=False)
    action, root, _ = mcts.search(state, n_sims)
    ranked = sorted(root.N.items(), key=lambda kv: -kv[1])
    vstr = "  ".join(f"{TYPE_NAMES[c]}:N={n},q={root.W[c]/max(n,1):+.3f},p={root.P[c]:.3f}"
                     for c, n in ranked)
    print(f"    search({n_sims}) plays {polyshark.ActionType(action.type)!s:<24} {vstr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="checkpoint .pt path(s), oldest first")
    ap.add_argument("--seeds", type=int, nargs="+", default=[999],
                    help="probe map seeds (use seeds the run never trained on)")
    ap.add_argument("--sims", type=int, default=200, help="search budget for the dump")
    ap.add_argument("--midgame", type=int, default=0,
                    help="also probe a state this many turns into a heuristic game")
    args = ap.parse_args()

    states = []
    for seed in args.seeds:
        states.append((f"seed {seed} fresh", polyshark.make_random_game(seed)))
        if args.midgame > 0:
            s = midgame_state(seed, args.midgame)
            if s is not None:
                states.append((f"seed {seed} midgame(t{args.midgame})", s))

    for path in args.checkpoints:
        print(f"\n=== {path} ===")
        ev = load_evaluator(path)
        for label, state in states:
            probe_state(ev, state, args.sims, label)


if __name__ == "__main__":
    main()
