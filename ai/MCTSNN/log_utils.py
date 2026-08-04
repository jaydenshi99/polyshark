import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark

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

# Pull names from the engine enums so adding a unit/tech updates labels for free.
_UNIT_NAMES = {int(v): v.name for v in polyshark.UnitType.__members__.values()}
_TECH_NAMES = {int(v): v.name for v in polyshark.TechType.__members__.values()}


def tile_coords(idx, sz):
    r, c = divmod(idx, sz)
    return f"({r},{c})"


def format_action(action, sz: int = None):
    """sz defaults to the largest reasonable; pass state.map_size() for accuracy."""
    if sz is None:
        sz = 32  # cosmetic fallback only used when caller forgets to pass it
    t    = action.type
    name = _ACTION_NAMES.get(t, str(t))
    src, dst, param = action.src, action.dst, action.param

    if t in (polyshark.ActionType.Move, polyshark.ActionType.Attack):
        return f"{name} {tile_coords(src, sz)} -> {tile_coords(dst, sz)}"
    if t == polyshark.ActionType.TrainUnit:
        return f"{name} {_UNIT_NAMES.get(param, param)} at {tile_coords(src, sz)}"
    if t == polyshark.ActionType.ResearchTech:
        return f"{name} {_TECH_NAMES.get(param, param)}"
    if t == polyshark.ActionType.CaptureCity:
        return f"{name} {tile_coords(dst, sz)}"
    if t == polyshark.ActionType.HarvestResource:
        return f"{name} at {tile_coords(src, sz)}"
    if t == polyshark.ActionType.UpgradeCity:
        return f"{name} at {tile_coords(src, sz)} upgrade={param}"
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
