import random
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build/bindings"))
import polyshark

REPLAYS_DIR = os.path.join(os.path.dirname(__file__), "../replays")


def pick_action(actions):
    affordable = [a for a in actions if a.affordable and a.type != polyshark.ActionType.DebugAddPop]
    pool = affordable if affordable else actions
    return random.choice(pool)


def save_replay(actions, path, seed, sz):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"seed {seed} {sz}\n")
        for a in actions:
            f.write(f"{int(a.type)} {a.src} {a.dst} {a.param}\n")


def run_game():
    seed = random.randint(1, 2**32 - 1)
    state = polyshark.make_random_game(seed=seed)
    sz = state.map_size()
    history = []

    while not state.is_terminal():
        actions = state.legal_actions()
        action = pick_action(actions)
        history.append(action)
        state = state.apply_action(action)

    winner = state.winner()
    print(f"Game over — player {winner} wins ({len(history)} actions)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_path = os.path.join(REPLAYS_DIR, f"game_{timestamp}.replay")
    save_replay(history, replay_path, seed, sz)
    print(f"Replay saved to {replay_path}")


if __name__ == "__main__":
    run_game()
