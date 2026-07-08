"""
Round-trip correctness for the factored legal-action helper. Run:

    source .venv/bin/activate
    python ai/mctsnn-v2/tests/test_factored.py

The gate that makes masking trustworthy:
  - bijection: the tree's stored Actions == the engine's in-scope legal Actions (both ways).
  - masks: walking the per-stage masks reproduces exactly the tree's paths.
  - apply: every reconstructed Action actually applies to a valid state.
  - modal: UpgradingCity phase yields the upgrade options and nothing else.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../build/bindings"))
import polyshark  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from factored import FactoredActions  # noqa: E402
from features import visible_snapshot  # noqa: E402
from policy import (  # noqa: E402
    N_TYPES, T_MOVE, T_ATTACK, T_HARVEST, T_CAPTURE, T_TRAIN, T_RESEARCH, T_RECOVER, T_END,
)

_AT = polyshark.ActionType
_EXCLUDED = {_AT.ConstructBuilding, _AT.DebugAddPop}
_SPATIAL = {_AT.Move, _AT.Attack, _AT.CaptureCity, _AT.HarvestResource}


def _sig(a):
    return (int(a.type), a.src, a.dst, a.param)


def _in_scope(a):
    return a.affordable and a.type not in _EXCLUDED


def _paths_from_masks(fa):
    """Reconstruct all legal paths by walking the per-stage masks (what MCTS does)."""
    paths = set()
    tm = fa.type_mask()
    for ti in range(N_TYPES):
        if not tm[ti]:
            continue
        if ti == T_END:
            paths.add((T_END,))
        elif ti in (T_MOVE, T_ATTACK):
            for slot, ok in enumerate(fa.entity_mask(ti)):
                if ok:
                    for tile, ok2 in enumerate(fa.tile_mask(ti, slot)):
                        if ok2:
                            paths.add((ti, slot, tile))
        elif ti == T_RECOVER:
            for slot, ok in enumerate(fa.entity_mask(ti)):
                if ok:
                    paths.add((ti, slot))
        elif ti in (T_HARVEST, T_CAPTURE):
            for tile, ok in enumerate(fa.tile_mask(ti)):
                if ok:
                    paths.add((ti, tile))
        elif ti == T_TRAIN:
            for slot, ok in enumerate(fa.entity_mask(ti)):
                if ok:
                    for u, ok2 in enumerate(fa.train_unit_mask(slot)):
                        if ok2:
                            paths.add((ti, slot, u))
        elif ti == T_RESEARCH:
            for t, ok in enumerate(fa.research_mask()):
                if ok:
                    paths.add((T_RESEARCH, t))
    return paths


def _check_state(state):
    fa = FactoredActions(state)

    legal = {_sig(a) for a in state.legal_actions() if _in_scope(a)}
    stored = {_sig(a) for a in fa.by_path.values()} | {_sig(a) for a in fa.upgrade_options}

    # Bijection: stored Actions exactly match the engine's in-scope legal set.
    assert stored == legal, f"stored != legal\n  missing {legal - stored}\n  extra {stored - legal}"
    assert len(fa.by_path) + len(fa.upgrade_options) == len(legal), "path collision lost an action"

    # Masks reproduce exactly the tree's paths.
    assert _paths_from_masks(fa) == set(fa.by_path), "masks disagree with the tree"

    # Every reconstructed Action applies to a valid state.
    for path, a in fa.by_path.items():
        assert state.apply_action(a).check_invariants(), f"apply failed for {path}"
    for a in fa.upgrade_options:
        assert state.apply_action(a).check_invariants(), "apply failed for upgrade option"

    return fa


def test_roundtrip_random():
    import random
    rng = random.Random(0)
    n_states = phases_idle = phases_upgrade = 0

    for seed in range(1, 40):
        state = polyshark.make_random_game(seed)
        for _ in range(60):
            if state.is_terminal():
                break
            fa = _check_state(state)
            n_states += 1
            if fa.upgrade_options:
                phases_upgrade += 1
            else:
                phases_idle += 1
            legal = [a for a in state.legal_actions() if a.affordable]
            if not legal:
                break
            state = state.apply_action(rng.choice(legal))

    print(f"[random]  {n_states} states checked  (idle={phases_idle}, upgrading={phases_upgrade})")


def test_upgrade_modal():
    """Force a pending upgrade via DebugAddPop, then verify the modal handling."""
    state = polyshark.make_random_game(1)
    for _ in range(40):
        if state.phase() == polyshark.GameStateType.UpgradingCity:
            break
        pops = [a for a in state.legal_actions()
                if a.type == _AT.DebugAddPop and a.param == 10]
        if not pops:
            break
        state = state.apply_action(pops[0])

    assert state.phase() == polyshark.GameStateType.UpgradingCity, "failed to force pending upgrade"
    fa = _check_state(state)
    assert len(fa.by_path) == 0, "no normal actions should be legal while upgrading"
    assert len(fa.upgrade_options) >= 2, "expected >=2 upgrade options"
    assert all(a.type == _AT.UpgradeCity for a in fa.upgrade_options)
    print(f"[modal]   UpgradingCity: {len(fa.upgrade_options)} upgrade options, tree empty OK")


def test_visibility_filter():
    """root_visible drops spatial actions whose target tile isn't visible at the root."""
    # A state with real spatial actions (advance a game a few plies).
    state = polyshark.make_random_game(3)
    for _ in range(6):
        la = [a for a in state.legal_actions() if a.affordable]
        if not la:
            break
        state = state.apply_action(la[0])

    fa_all = FactoredActions(state)
    spatial = [(p, a) for p, a in fa_all.by_path.items() if a.type in _SPATIAL]
    assert spatial, "test needs a spatial-target action"
    hidden_path, hidden_act = spatial[0]
    tgt = hidden_act.dst

    # Hide exactly that target tile; everything else visible.
    rv = [True] * state.map_tiles()
    rv[tgt] = False
    fa = FactoredActions(state, root_visible=rv)

    assert hidden_path not in fa.by_path, "action targeting hidden tile was not filtered"
    assert all(not (a.type in _SPATIAL and a.dst == tgt) for a in fa.by_path.values()), \
        "a surviving spatial action still targets the hidden tile"

    expected = {_sig(a) for a in state.legal_actions()
                if _in_scope(a) and not (a.type in _SPATIAL and a.dst == tgt)}
    stored = {_sig(a) for a in fa.by_path.values()} | {_sig(a) for a in fa.upgrade_options}
    assert stored == expected, "round-trip broke under the visibility filter"
    removed = len(fa_all.by_path) - len(fa.by_path)
    print(f"[visfilter] hidden target filtered ({removed} action(s) dropped); round-trip holds OK")

    # Invariant under real frozen fog: no surviving spatial action targets a hidden tile.
    import random
    rng = random.Random(1)
    for seed in range(1, 15):
        s = polyshark.make_random_game(seed)
        for _ in range(30):
            if s.is_terminal():
                break
            snap = visible_snapshot(s)
            fa2 = FactoredActions(s, root_visible=snap)
            for a in fa2.by_path.values():
                if a.type in _SPATIAL:
                    assert snap[a.dst], "surviving spatial action targets a hidden tile"
            la = [a for a in s.legal_actions() if a.affordable]
            if not la:
                break
            s = s.apply_action(rng.choice(la))
    print("[visfilter] frozen-fog invariant holds across states OK")


if __name__ == "__main__":
    test_roundtrip_random()
    test_upgrade_modal()
    test_visibility_filter()
    print("\nAll factored-action checks passed.")
