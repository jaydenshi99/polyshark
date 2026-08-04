"""
Translates between engine Action objects and integer NN indices.

All sizing constants are derived from an `AISpec` so the codec auto-adapts
to map size, unit count, tech count. Layout (per spec):

  offset   count                     action          key fields
  ------   --------                  --------------- ------------------------
  move_off n_tiles * 25              Move            src * 25 + offset(dr, dc)
  attack   n_tiles * 25              Attack          src * 25 + offset(dr, dc)
  train    n_tiles * n_units         TrainUnit       city_tile * n_units + (unit - 1)
  harvest  n_tiles * n_resources     HarvestResource city_tile * n_res   + (res  - 1)
  upgrade  n_tiles * n_upgrades      UpgradeCity     city_tile * n_upg   + upgrade
  research n_tech_types              ResearchTech    tech_type
  capture  n_tiles                   CaptureCity     dst_tile
  end                  1             EndTurn         —

offset(dr, dc) = (dr + 2) * 5 + (dc + 2),  dr/dc ∈ {-2,-1,0,1,2}
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark

from spec import AISpec, DEFAULT_SPEC


def _offset_idx(src: int, dst: int, sz: int) -> int:
    dr = dst // sz - src // sz
    dc = dst %  sz - src %  sz
    return (dr + 2) * 5 + (dc + 2)


def action_to_index(action, state, spec: AISpec = DEFAULT_SPEC) -> int:
    """Convert an engine Action to its integer index for `spec`'s NN."""
    t = action.type
    src, dst, param = action.src, action.dst, action.param
    sz = spec.map_size

    if t == polyshark.ActionType.Move:
        return spec.move_off + src * spec.move_offsets + _offset_idx(src, dst, sz)

    if t == polyshark.ActionType.Attack:
        return spec.attack_off + src * spec.move_offsets + _offset_idx(src, dst, sz)

    if t == polyshark.ActionType.TrainUnit:
        return spec.train_off + src * spec.n_unit_types + (param - 1)

    if t == polyshark.ActionType.HarvestResource:
        city_tile = state.get_city(src).tile_index
        return spec.harvest_off + city_tile * spec.n_resource_types + (param - 1)

    if t == polyshark.ActionType.UpgradeCity:
        city_tile = state.get_city(src).tile_index
        return spec.upgrade_off + city_tile * spec.n_upgrade_types + param

    if t == polyshark.ActionType.ResearchTech:
        return spec.research_off + param

    if t == polyshark.ActionType.CaptureCity:
        return spec.capture_off + dst

    if t == polyshark.ActionType.EndTurn:
        return spec.end_turn_idx

    raise ValueError(f"Action type {t} is not in the NN action space")


def index_to_action(idx: int, state, spec: AISpec = DEFAULT_SPEC):
    """Resolve an index by searching `state.legal_actions()`."""
    if not (0 <= idx < spec.action_size):
        raise ValueError(f"Index {idx} out of [0, {spec.action_size})")

    for action in state.legal_actions():
        try:
            if action_to_index(action, state, spec) == idx:
                return action
        except ValueError:
            pass  # skip unmapped types (ConstructBuilding, DebugAddPop)

    raise ValueError(f"No legal action maps to index {idx}")


def legal_action_indices(state, spec: AISpec = DEFAULT_SPEC) -> list[int]:
    """Return indices for every affordable legal action in `state`."""
    indices = []
    for action in state.legal_actions():
        if not action.affordable:
            continue
        try:
            indices.append(action_to_index(action, state, spec))
        except ValueError:
            pass
    return indices
