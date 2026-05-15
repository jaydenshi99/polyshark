import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark

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

_UNIT_NAMES = {1: "Warrior", 2: "Archer", 3: "Rider"}
_TECH_NAMES = {0: "Origin", 1: "Hunting", 2: "Organisation", 3: "Farming",
               4: "Riding",  5: "Climbing", 6: "Archery",      7: "Mining"}


def tile_coords(idx):
    r, c = divmod(idx, MAP_SIZE)
    return f"({r},{c})"


def format_action(action):
    t    = action.type
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
                  if state.tile_at(i).has_city
                  and state.get_city(state.tile_at(i).city_id).owner == p]
        units  = [state.get_unit(state.tile_at(i).unit_id)
                  for i in range(sz * sz)
                  if state.tile_at(i).has_unit
                  and state.get_unit(state.tile_at(i).unit_id).owner == p
                  and state.get_unit(state.tile_at(i).unit_id).is_alive]
        stars  = state.get_stars(p)
        print(f"  P{p}: {stars} stars | {len(cities)} cities | {len(units)} units")
