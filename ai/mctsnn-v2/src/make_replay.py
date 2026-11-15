"""
Play one game and write a .replay file the visualiser can load.

Usage (from repo root, venv active):
    python ai/mctsnn-v2/src/make_replay.py                    # random vs random, seed 42
    python ai/mctsnn-v2/src/make_replay.py --seed 7 --mcts    # MCTS(heuristic) vs random
    python ai/mctsnn-v2/src/make_replay.py --seed 7 --out foo.replay

Then view it:
    ./build/visualizer/visualizer --replay ai/mctsnn-v2/data/random_seed42.replay
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from arena import Agent, Arena, RandomStrategy, MCTSStrategy, write_replay  # noqa: E402
from mcts import HeuristicEvaluator  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "../data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="map seed")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--mcts", action="store_true",
                    help="player 0 uses MCTS(heuristic) instead of random")
    ap.add_argument("--n-sims", type=int, default=40)
    ap.add_argument("--out", default=None, help="output path (default: data/<tag>_seed<seed>.replay)")
    args = ap.parse_args()

    if args.mcts:
        p0 = Agent("mcts", MCTSStrategy(HeuristicEvaluator(), n_sims=args.n_sims, add_noise=False))
        tag = "mcts_vs_random"
    else:
        p0 = Agent("rand-0", RandomStrategy(seed=1))
        tag = "random"
    p1 = Agent("rand-1", RandomStrategy(seed=2))

    res = Arena([p0, p1]).play_game(seed=args.seed, max_turns=args.max_turns)

    out = args.out or os.path.join(_DATA, f"{tag}_seed{args.seed}.replay")
    write_replay(out, res)

    print(f"reason={res.reason} winner={res.winner} turns={res.turns} "
          f"actions={len(res.history)} v_finals=({res.v_finals[0]:+.3f},{res.v_finals[1]:+.3f})")
    print(f"wrote {out}")
    print(f"view: ./build/visualizer/visualizer --replay {out}")


if __name__ == "__main__":
    main()
