"""
Translates between engine Action objects and integer indices used by the NN.

Action space layout (total = 7995):

  Offset   Count   Action          Key fields
  ------   -----   ------          ----------
       0    3025   Move            src_tile * 25 + offset(dr, dc)
    3025    3025   Attack          src_tile * 25 + offset(dr, dc)
    6050     363   TrainUnit       city_tile * 3  + (unit_type - 1)
    6413     484   HarvestResource city_tile * 4  + (resource_type - 1)
    6897     968   UpgradeCity     city_tile * 8  + upgrade_type
    7865       8   ResearchTech    tech_type
    7873     121   CaptureCity     to_tile
    7994       1   EndTurn         —

offset(dr, dc) = (dr + 2) * 5 + (dc + 2),  dr/dc ∈ {-2,-1,0,1,2}
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark

MAP_SIZE     = 11
ACTION_SIZE  = 7995

MOVE_OFF     = 0
ATTACK_OFF   = 3025
TRAIN_OFF    = 6050
HARVEST_OFF  = 6413
UPGRADE_OFF  = 6897
RESEARCH_OFF = 7865
CAPTURE_OFF  = 7873
END_TURN_IDX = 7994


def _offset_idx(src: int, dst: int) -> int:
    dr = dst // MAP_SIZE - src // MAP_SIZE
    dc = dst %  MAP_SIZE - src %  MAP_SIZE
    return (dr + 2) * 5 + (dc + 2)


def action_to_index(action, state) -> int:
    """
    Convert an engine Action to its integer index.

    Requires state for HarvestResource and UpgradeCity to resolve city_id → tile_index.
    Raises ValueError for action types not in the action space (ConstructBuilding, DebugAddPop).
    """
    t = action.type
    src, dst, param = action.src, action.dst, action.param

    if t == polyshark.ActionType.Move:
        return MOVE_OFF + src * 25 + _offset_idx(src, dst)

    if t == polyshark.ActionType.Attack:
        return ATTACK_OFF + src * 25 + _offset_idx(src, dst)

    if t == polyshark.ActionType.TrainUnit:
        return TRAIN_OFF + src * 3 + (param - 1)

    if t == polyshark.ActionType.HarvestResource:
        city_tile = state.get_city(src).tile_index
        return HARVEST_OFF + city_tile * 4 + (param - 1)

    if t == polyshark.ActionType.UpgradeCity:
        city_tile = state.get_city(src).tile_index
        return UPGRADE_OFF + city_tile * 8 + param

    if t == polyshark.ActionType.ResearchTech:
        return RESEARCH_OFF + param

    if t == polyshark.ActionType.CaptureCity:
        return CAPTURE_OFF + dst

    if t == polyshark.ActionType.EndTurn:
        return END_TURN_IDX

    raise ValueError(f"Action type {t} is not in the NN action space")


def index_to_action(idx: int, state):
    """
    Convert an integer index to an engine Action by searching legal_actions.

    Searching guarantees correct param values (e.g. remaining move points for Move)
    and correct city_id for actions where from=city_id. Raises ValueError if the
    index doesn't match any legal action.
    """
    if not (0 <= idx < ACTION_SIZE):
        raise ValueError(f"Index {idx} out of [0, {ACTION_SIZE})")

    for action in state.legal_actions():
        try:
            if action_to_index(action, state) == idx:
                return action
        except ValueError:
            pass  # skip unmapped types

    raise ValueError(f"No legal action maps to index {idx}")


def legal_action_indices(state) -> list[int]:
    """Return the list of integer indices for every affordable legal action in state."""
    indices = []
    for action in state.legal_actions():
        if not action.affordable:
            continue
        try:
            indices.append(action_to_index(action, state))
        except ValueError:
            pass  # skip ConstructBuilding / DebugAddPop
    return indices
