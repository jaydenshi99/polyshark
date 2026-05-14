import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build/bindings"))
import polyshark


def pick_action(actions):
    affordable = [a for a in actions if a.affordable]
    pool = affordable if affordable else actions
    return random.choice(pool)


def run_game():
    state = polyshark.make_game()

    while not state.is_terminal():
        actions = state.legal_actions()
        action = pick_action(actions)
        state = state.apply_action(action)

    print(f"Game over — player {state.winner()} wins")


if __name__ == "__main__":
    run_game()
