import random
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build/bindings"))
import polyshark

REPLAYS_DIR = os.path.join(os.path.dirname(__file__), "../replays")


def pick_action(actions):
    affordable = [a for a in actions if a.affordable]
    pool = affordable if affordable else actions
    return random.choice(pool)


def save_replay(actions, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for a in actions:
            f.write(f"{int(a.type)} {a.src} {a.dst} {a.param}\n")


def run_game():
    state = polyshark.make_random_game()
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
    save_replay(history, replay_path)
    print(f"Replay saved to {replay_path}")


if __name__ == "__main__":
    run_game()
