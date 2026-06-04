"""
Self-play driver for the phase-1 heuristic MCTS bot.

Mirrors `ai/random_bot.py` but swaps the random policy for `MCTS.search`. Run
from anywhere:

    python ai/mcts-phase1/run.py --sims 200 --seed 1

Replay format matches `random_bot.py` so the visualizer's `--replay` flag
works as-is.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# Engine bindings (built into ../../build/bindings).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark  # noqa: E402

# Local modules (script-style imports — the folder name has a hyphen so it
# isn't importable as a package).
sys.path.insert(0, os.path.dirname(__file__))
from mcts import MCTS  # noqa: E402

REPLAYS_DIR = os.path.join(os.path.dirname(__file__), "../../replays")


def save_replay(actions, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for a in actions:
            f.write(f"{int(a.type)} {a.src} {a.dst} {a.param}\n")


def run_game(sims: int, seed: int | None, max_turns: int) -> None:
    state   = polyshark.make_random_game(seed) if seed is not None else polyshark.make_random_game()
    bot     = MCTS(n_simulations=sims, seed=seed)
    history = []
    started = time.perf_counter()

    while not state.is_terminal() and state.get_turn() < max_turns:
        action = bot.search(state)
        if action is None:
            print(f"[mcts] no legal action at turn {state.get_turn()} — stopping")
            break
        history.append(action)
        state = state.apply_action(action)

    elapsed = time.perf_counter() - started
    winner  = state.winner()
    outcome = ("draw" if winner == -1
               else f"player {winner} wins")
    print(f"Game over — {outcome} after {len(history)} actions "
          f"in {elapsed:.1f}s ({len(history)/elapsed:.1f} actions/s)")

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_path = os.path.join(REPLAYS_DIR, f"mcts1_{ts}.replay")
    save_replay(history, replay_path)
    print(f"Replay saved to {replay_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-1 heuristic MCTS self-play.")
    p.add_argument("--sims",      type=int, default=200, help="simulations per move")
    p.add_argument("--seed",      type=int, default=None, help="RNG seed (None = fresh game)")
    p.add_argument("--max-turns", type=int, default=200, help="hard cutoff to avoid runaways")
    args = p.parse_args()

    run_game(sims=args.sims, seed=args.seed, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
