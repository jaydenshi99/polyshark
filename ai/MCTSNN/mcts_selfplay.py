import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark

sys.path.insert(0, os.path.dirname(__file__))
from spec import AISpec
from model import PolysharkNet
from mcts import MCTS
from action_codec import index_to_action
from log_utils import format_action, log_state

REPLAYS_DIR   = os.path.join(os.path.dirname(__file__), "../../replays")
N_SIMULATIONS = 800
TEMPERATURE   = 1.0

# Match train.py settings — model checkpoints are spec-sensitive.
SETTINGS = polyshark.EngineSettings()
SETTINGS.map_size = 9
SPEC = AISpec(SETTINGS)


def pick_action(mcts, state):
    root = mcts.search(state, n_simulations=N_SIMULATIONS, add_noise=True)
    policy = mcts.get_policy(root, temperature=TEMPERATURE)
    action_idx = int(np.random.choice(len(policy), p=policy))
    return index_to_action(action_idx, state, SPEC)


def save_replay(history, seed, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"seed {seed}\n")
        for a in history:
            f.write(f"{int(a.type)} {a.src} {a.dst} {a.param}\n")


def run_game():
    import torch
    import random
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | {SPEC}")

    model = PolysharkNet(SPEC).to(device)
    mcts  = MCTS(model, device=device, spec=SPEC)

    seed = random.randint(0, 2**32 - 1)
    SETTINGS.seed = seed
    state = polyshark.make_game(SETTINGS)
    history = []
    last_logged_turn = -1

    while not state.is_terminal():
        turn   = state.get_turn()
        player = state.current_player()

        if turn != last_logged_turn:
            print(f"\n=== Turn {turn} ===")
            log_state(state)
            last_logged_turn = turn

        action = pick_action(mcts, state)
        print(f"  P{player}: {format_action(action, SPEC.map_size)}")
        history.append(action)
        state = state.apply_action(action)

    winner = state.winner()
    print(f"\n=== Game over — player {winner} wins ({len(history)} actions) ===")
    log_state(state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_path = os.path.join(REPLAYS_DIR, f"mcts_{timestamp}.replay")
    save_replay(history, seed, replay_path)
    print(f"Replay saved to {replay_path}")


if __name__ == "__main__":
    run_game()
