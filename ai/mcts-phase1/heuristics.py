"""
Heuristic functions used by the phase-1 MCTS bot.

Three pluggable hooks:
- ``state_value(state, player)``  → float in [-1, 1], +1 = winning for `player`
- ``rollout_policy(state, rng)``  → Action to play next during a rollout
- ``action_prior(state, action)`` → optional float for PUCT-style selection

The defaults are deliberately tiny — they keep the bot runnable end-to-end
without inventing engine knowledge. Replace them with something smarter as
the heuristic search matures (own-stars vs enemy-stars, capital threat,
city counts, unit material, board control, etc).
"""

from __future__ import annotations

import random
import sys
import os

# Same binding-path shim as random_bot.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark  # noqa: E402


# ---------------------------------------------------------------------------
# State evaluation
# ---------------------------------------------------------------------------

def state_value(state, player: int) -> float:
    """
    Estimate `player`'s value of the position in [-1, 1].

    Terminal states are handled by the caller (MCTS sees winner() directly),
    so this only needs to score non-terminal positions. The default is a
    crude star + city + unit material diff squashed to [-1, 1] via tanh.
    """
    if state.is_terminal():
        w = state.winner()
        if w == player:    return 1.0
        if w == -1:        return 0.0
        return -1.0

    me, them = player, 1 - player

    stars = state.get_stars(me) - state.get_stars(them)

    # Material on the board: each tile holding a city/unit contributes for
    # its owner. Cheap O(N) sweep — fine until rollouts dominate cost.
    cities  = [0, 0]
    units   = [0, 0]
    capital = [None, None]
    for i in range(state.map_tiles()):
        t = state.tile_at(i)
        if t.has_city:
            c = state.get_city(t.city_id)
            cities[c.owner] += 1 + c.level
            if c.is_capital:
                capital[c.owner] = c
        if t.has_unit:
            u = state.get_unit(t.unit_id)
            units[u.owner] += u.hp  # weight by remaining HP

    city_diff = cities[me] - cities[them]
    unit_diff = units[me]  - units[them]

    # Capital captured? That's effectively decisive even pre-terminal.
    if capital[them] is not None and capital[them].owner == me: return 0.95
    if capital[me]   is not None and capital[me].owner   == them: return -0.95

    raw = 0.04 * stars + 0.15 * city_diff + 0.01 * unit_diff
    return _tanh(raw)


# ---------------------------------------------------------------------------
# Rollout policy
# ---------------------------------------------------------------------------

def rollout_policy(state, rng: random.Random):
    """
    Pick an action to play during a rollout. Default: random among affordable
    actions (mirrors random_bot.py). Replace with a smarter policy — e.g.
    "if any Attack is affordable, prefer it" — to tighten rollout signal.
    """
    actions   = state.legal_actions()
    if not actions:
        return None
    affordable = [a for a in actions if a.affordable]
    pool = affordable if affordable else actions
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Action prior (PUCT — optional, used if MCTS is configured for it)
# ---------------------------------------------------------------------------

def action_prior(state, action) -> float:
    """
    Hand-rolled prior in (0, 1]. Default treats all legal actions equally
    and just nudges Attack / CaptureCity above EndTurn. Caller normalises.
    """
    t = action.type
    if t == polyshark.ActionType.CaptureCity:    return 4.0
    if t == polyshark.ActionType.Attack:         return 2.0
    if t == polyshark.ActionType.HarvestResource:return 1.5
    if t == polyshark.ActionType.EndTurn:        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# Small utility — tanh without pulling in numpy for one call.
# ---------------------------------------------------------------------------

def _tanh(x: float) -> float:
    import math
    return math.tanh(x)
