import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark

sys.path.insert(0, os.path.dirname(__file__))
from model import PolysharkNet
from mcts import MCTS
from action_codec import index_to_action

REPLAYS_DIR = os.path.join(os.path.dirname(__file__), "../../replays")
N_SIMULATIONS = 800
TEMPERATURE = 1.0

MAP_SIZE = 11

_ACTION_NAMES = {
    polyshark.ActionType.Move:              "Move",
    polyshark.ActionType.Attack:            "Attack",
    polyshark.ActionType.TrainUnit:         "TrainUnit",
    polyshark.ActionType.ConstructBuilding: "ConstructBuilding",
    polyshark.ActionType.ResearchTech:      "ResearchTech",
    polyshark.ActionType.CaptureCity:       "CaptureCity",
    polyshark.ActionType.HarvestResource:   "HarvestResource",
    polyshark.ActionType.UpgradeCity:       "UpgradeCity",
    polyshark.ActionType.EndTurn:           "EndTurn",
}

_UNIT_NAMES  = {1: "Warrior", 2: "Archer", 3: "Rider"}
_TECH_NAMES  = {0: "Origin", 1: "Hunting", 2: "Organisation", 3: "Farming",
                4: "Riding",  5: "Climbing", 6: "Archery",      7: "Mining"}


def tile_coords(idx):
    r, c = divmod(idx, MAP_SIZE)
    return f"({r},{c})"


def format_action(action):
    t = action.type
    name = _ACTION_NAMES.get(t, str(t))
    src, dst, param = action.src, action.dst, action.param

    if t in (polyshark.ActionType.Move, polyshark.ActionType.Attack):
        return f"{name} {tile_coords(src)} -> {tile_coords(dst)}"
    if t == polyshark.ActionType.TrainUnit:
        return f"{name} {_UNIT_NAMES.get(param, param)} at {tile_coords(src)}"
    if t == polyshark.ActionType.ResearchTech:
        return f"{name} {_TECH_NAMES.get(param, param)}"
    if t == polyshark.ActionType.CaptureCity:
        return f"{name} {tile_coords(dst)}"
    if t == polyshark.ActionType.HarvestResource:
        return f"{name} at {tile_coords(src)}"
    if t == polyshark.ActionType.UpgradeCity:
        return f"{name} at {tile_coords(src)} upgrade={param}"
    return name


def log_state(state):
    sz = state.map_size()
    for p in range(2):
        cities = [state.get_city(state.tile_at(i).city_id)
                  for i in range(sz * sz)
                  if state.tile_at(i).has_city and state.get_city(state.tile_at(i).city_id).owner == p]
        units  = [state.get_unit(state.tile_at(i).unit_id)
                  for i in range(sz * sz)
                  if state.tile_at(i).has_unit and state.get_unit(state.tile_at(i).unit_id).owner == p
                  and state.get_unit(state.tile_at(i).unit_id).is_alive]
        stars  = state.get_stars(p)
        print(f"  P{p}: {stars} stars | {len(cities)} cities | {len(units)} units")


def pick_action(mcts, state):
    root = mcts.search(state, n_simulations=N_SIMULATIONS, add_noise=True)
    policy = mcts.get_policy(root, temperature=TEMPERATURE)
    action_idx = int(np.random.choice(len(policy), p=policy))
    return index_to_action(action_idx, state)


def save_replay(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for a in history:
            f.write(f"{int(a.type)} {a.src} {a.dst} {a.param}\n")


def run_game():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = PolysharkNet().to(device)
    mcts = MCTS(model, device=device)

    state = polyshark.make_game()
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
        print(f"  P{player}: {format_action(action)}")
        history.append(action)
        state = state.apply_action(action)

    winner = state.winner()
    print(f"\n=== Game over — player {winner} wins ({len(history)} actions) ===")
    log_state(state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_path = os.path.join(REPLAYS_DIR, f"mcts_{timestamp}.replay")
    save_replay(history, replay_path)
    print(f"Replay saved to {replay_path}")


if __name__ == "__main__":
    run_game()
