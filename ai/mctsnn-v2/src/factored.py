"""
Factored legal-action helper (see docs/policy_head.md).

Turns the engine's flat `legal_actions()` into the head's factored tree: it buckets each
legal Action under its (type, entity, target) path, produces the per-stage legal masks,
and maps a chosen path back to the concrete engine Action to apply.

Entity slots follow the encoder's ordering: slot i is the i-th unit/city in tile-scan
order (i.e. the i-th row of encode_entities' unit_tok/city_tok). We invert unit_tiles /
city_tiles to map an Action's tile back to its slot.

Field mapping per type (confirmed against engine/src/game_state/legal_actions.cpp):
  move/attack : from = unit tile,  to = target tile   -> (type, unit_slot, to)
  recover     : from = unit tile                       -> (type, unit_slot)
  harvest     : to = resource tile (city implied)      -> (type, to)
  capture     : to = city/village tile                 -> (type, to)
  train       : from = city tile, param = unit type    -> (type, city_slot, param-1)
  research    : param = tech                            -> (type, param-1)
  end_turn    : -                                       -> (type,)
  upgrade     : MODAL (phase-gated) — kept in a separate options list, not the tree
  DebugAddPop / ConstructBuilding : excluded (not policy actions)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
import polyshark  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from features import encode_entities  # noqa: E402
from policy import (  # noqa: E402
    N_TYPES, K_TRAIN_UNITS, K_TECH,
    T_MOVE, T_ATTACK, T_HARVEST, T_CAPTURE, T_TRAIN, T_RESEARCH, T_RECOVER, T_END,
)

_AT = polyshark.ActionType
_TYPE_OF = {
    _AT.Move: T_MOVE, _AT.Attack: T_ATTACK, _AT.HarvestResource: T_HARVEST,
    _AT.CaptureCity: T_CAPTURE, _AT.TrainUnit: T_TRAIN, _AT.ResearchTech: T_RESEARCH,
    _AT.Recover: T_RECOVER, _AT.EndTurn: T_END,
}
_UNIT_TYPES = (T_MOVE, T_ATTACK, T_RECOVER)  # entity stage points at units
_SPATIAL_TARGET = (T_MOVE, T_ATTACK, T_HARVEST, T_CAPTURE)


class FactoredActions:
    """Per-state factored view of legal_actions(): masks + path<->Action mapping."""

    def __init__(self, state, encoded=None, root_visible=None):
        enc = encoded if encoded is not None else encode_entities(state)
        self.map_tiles = state.map_tiles()
        self.n_units = int(enc["unit_tiles"].shape[0])
        self.n_cities = int(enc["city_tiles"].shape[0])
        self._unit_slot = {int(t): i for i, t in enumerate(enc["unit_tiles"])}
        self._city_slot = {int(t): i for i, t in enumerate(enc["city_tiles"])}
        # Frozen root fog: drop actions whose spatial target isn't root-visible (see
        # docs/mcts.md). None = no restriction. Own units/cities are always self-visible.
        self._root_visible = root_visible

        self.by_path = {}          # path tuple -> Action
        self.upgrade_options = []  # modal (UpgradingCity phase), emission order

        for a in state.legal_actions():
            if not a.affordable:
                continue
            if a.type == _AT.UpgradeCity:
                self.upgrade_options.append(a)
                continue
            if self._target_hidden(a):
                continue
            path = self._path_of(a)
            if path is not None:
                self.by_path[path] = a

    def _target_hidden(self, a):
        """True if the action's spatial target tile (a.dst) isn't visible at the root."""
        if self._root_visible is None:
            return False
        if _TYPE_OF.get(a.type) in _SPATIAL_TARGET:
            return not self._root_visible[a.dst]
        return False

    def _path_of(self, a):
        t = _TYPE_OF.get(a.type)
        if t is None:                       # DebugAddPop / ConstructBuilding
            return None
        if t in (T_MOVE, T_ATTACK):
            return (t, self._unit_slot[a.src], a.dst)
        if t == T_RECOVER:
            return (t, self._unit_slot[a.src])
        if t in (T_HARVEST, T_CAPTURE):
            return (t, a.dst)
        if t == T_TRAIN:
            return (t, self._city_slot[a.src], a.param - 1)
        if t == T_RESEARCH:
            return (t, a.param - 1)
        return (T_END,)

    # --- per-stage legal masks (True = legal) ---

    def type_mask(self):
        m = [False] * N_TYPES
        for path in self.by_path:
            m[path[0]] = True
        return m

    def entity_mask(self, type_idx):
        n = self.n_units if type_idx in _UNIT_TYPES else self.n_cities
        m = [False] * n
        for path in self.by_path:
            if path[0] == type_idx:
                m[path[1]] = True
        return m

    def tile_mask(self, type_idx, entity_slot=None):
        m = [False] * self.map_tiles
        for path in self.by_path:
            if path[0] != type_idx:
                continue
            if type_idx in (T_MOVE, T_ATTACK):
                if path[1] == entity_slot:
                    m[path[2]] = True
            else:  # harvest / capture: target is path[1]
                m[path[1]] = True
        return m

    def train_unit_mask(self, city_slot):
        m = [False] * K_TRAIN_UNITS
        for path in self.by_path:
            if path[0] == T_TRAIN and path[1] == city_slot:
                m[path[2]] = True
        return m

    def research_mask(self):
        m = [False] * K_TECH
        for path in self.by_path:
            if path[0] == T_RESEARCH:
                m[path[1]] = True
        return m

    # --- path -> concrete engine Action (to apply) ---

    def action_for(self, path):
        return self.by_path[path]
