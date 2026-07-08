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

import math
import random
import sys
import os

# Same binding-path shim as random_bot.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark  # noqa: E402


# Per-unit-type combat worth used by state_value. Roughly tracks the unit's
# cost, with Giant bumped well above its 0-star training cost since its battle
# value is much higher than what you pay for it.
ARMY_VALUE = {
    polyshark.UnitType.Warrior:  2,
    polyshark.UnitType.Archer:   3,
    polyshark.UnitType.Rider:    3,
    polyshark.UnitType.Defender: 3,
    polyshark.UnitType.Giant:   15,
}


# Intrinsic worth of a researched tech, tracking tech_def.h's tier field (higher
# tiers cost more and gate stronger units/resources). Tier-0 Origin is free and
# always owned, so it's absent and scores nothing.
TECH_INTRINSIC_VALUE = {
    polyshark.TechType.Hunting:      1,
    polyshark.TechType.Organisation: 1,
    polyshark.TechType.Riding:       4,
    polyshark.TechType.Climbing:     1,
    polyshark.TechType.Farming:      1,
    polyshark.TechType.Archery:      4,
    polyshark.TechType.Mining:       1,
    polyshark.TechType.Strategy:     4,
}

# Population a resource yields when harvested, and the tech needed to harvest it
# (mirrors resource_def.h). A resource only counts once its tech is owned.
RESOURCE_POP = {
    polyshark.ResourceType.Fruit:  1,
    polyshark.ResourceType.Crop:   2,
    polyshark.ResourceType.Animal: 1,
    polyshark.ResourceType.Metal:  2,
}
RESOURCE_TECH = {
    polyshark.ResourceType.Fruit:  polyshark.TechType.Organisation,
    polyshark.ResourceType.Crop:   polyshark.TechType.Farming,
    polyshark.ResourceType.Animal: polyshark.TechType.Hunting,
    polyshark.ResourceType.Metal:  polyshark.TechType.Mining,
}


# action_prior bonus for a Move whose destination is closer to an uncaptured
# city (Village tile) than its origin — pulls the tree search toward expansion.
MOVE_TOWARD_CITY_PRIOR = 6.0


# ---------------------------------------------------------------------------
# Tech heuristic
# ---------------------------------------------------------------------------

def tech_heuristic(state, player: int) -> float:
    """
    Per-player tech worth = each owned tech's intrinsic tier value plus the
    harvestable population it unlocks. The population term sweeps every tile in
    the player's territory and sums the pop_reward of each resource the player
    has the tech to harvest. Returns me - them.
    """
    score = [0.0, 0.0]

    # Intrinsic value: each owned tech contributes its intrinsic worth (Origin = 0).
    for p in (player, 1 - player):
        for tech in state.get_techs(p):
            score[p] += TECH_INTRINSIC_VALUE.get(tech, 0)

    # Harvestable population unlocked within each player's territory.
    for i in range(state.map_tiles()):
        t = state.tile_at(i)
        bcid = t.border_city_id
        if bcid == -1:
            continue
        r = t.resource
        tech = RESOURCE_TECH.get(r)
        if tech is None:
            continue                            # no resource / nothing to harvest
        owner = state.get_city(bcid).owner
        if state.has_tech(owner, tech):
            score[owner] += RESOURCE_POP[r]

    return score[player] - score[1 - player]


# ---------------------------------------------------------------------------
# State evaluation
# ---------------------------------------------------------------------------

def state_value(state, player: int) -> float:
    if state.is_terminal():
        w = state.winner()
        if w == player:    return 100      # win  → peg bar to +100
        if w == -1:        return 0        # draw → centre
        return -100                        # loss → peg bar to -100

    me, them = player, 1 - player

    # Material on the board: each tile holding a city/unit contributes for
    # its owner. Cheap O(N) sweep — fine until rollouts dominate cost.
    cities    = [0, 0]
    units     = [0, 0]
    stars_pt  = [0, 0]   # income rate (sum of City.stars_per_turn) per player
    capital   = [None, None]
    size       = state.map_size()
    villages   = []           # (x, y) of uncaptured village tiles
    unit_xy    = [[], []]     # (x, y) of each player's units
    city_xy    = [[], []]     # (x, y) of each player's cities
    for i in range(state.map_tiles()):
        t = state.tile_at(i)
        if t.terrain == polyshark.TerrainType.Village:
            villages.append((i % size, i // size))
        if t.has_city:
            c = state.get_city(t.city_id)
            cities[c.owner]   += 1
            stars_pt[c.owner] += c.stars_per_turn
            city_xy[c.owner].append((i % size, i // size))
            if c.is_capital:
                capital[c.owner] = c
        if t.has_unit:
            u = state.get_unit(t.unit_id)
            v = ARMY_VALUE.get(u.type, 1)
            units[u.owner] += v * (u.hp / max(u.max_hp, 1))
            unit_xy[u.owner].append((i % size, i // size))

    # Tech worth: resource-usability heuristic (already a me - them diff).
    tech_diff = tech_heuristic(state, player)

    # Village proximity: for each player, the shortest Chebyshev distance from
    # any of their units to any uncaptured village (global min over villages).
    # The diff (theirs - mine) rewards me being closer to a village than them.
    def nearest_village_dist(p):
        # Min over BOTH pools: any unit (+0) and any city (+1, since a fresh unit
        # must still be trained there and step out). A nearby city can beat a
        # distant unit even with the +1.
        cands = [max(abs(vx - ux), abs(vy - uy)) + extra
                 for (vx, vy) in villages
                 for (pts, extra) in ((unit_xy[p], 0), (city_xy[p], 1))
                 for (ux, uy) in pts]
        dist_penalty = min(cands) if cands else size   # no units & no cities → max far
        print("Side %d nearest village dist: %d" % (p, dist_penalty))
        return dist_penalty

    village_diff = 0.0
    if villages:
        village_diff = nearest_village_dist(them) - nearest_village_dist(me)

    # Stars term reflects income rate, not the current stockpile — a sieged
    # / captured city contributes 0 (matches City::stars_per_turn in C++).
    stars_pt  = stars_pt[me]  - stars_pt[them]
    city_diff = cities[me]    - cities[them]
    unit_diff = units[me]     - units[them]
    print(f"Breakdown: stars_pt={stars_pt:.1f}, city_diff={city_diff}, unit_diff={unit_diff:.1f}, tech_diff={tech_diff:.1f}, village_diff={village_diff:.1f}")
    raw =  3 * stars_pt + 5 * city_diff + unit_diff + tech_diff + village_diff
    return raw


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
    Hand-rolled prior (caller normalises). Nudges Attack / CaptureCity above
    EndTurn, and gives a big bonus to Moves that step toward an uncaptured city.
    """
    t = action.type
    if t == polyshark.ActionType.CaptureCity:    return 4.0
    if t == polyshark.ActionType.Attack:         return 2.0
    if t == polyshark.ActionType.HarvestResource:return 1.5
    if t == polyshark.ActionType.EndTurn:        return 0.5
    if t == polyshark.ActionType.Move and _move_approaches_city(state, action):
        return MOVE_TOWARD_CITY_PRIOR
    return 1.0


def _move_approaches_city(state, action) -> bool:
    """True if `action`'s destination is strictly closer (Chebyshev) to the
    nearest uncaptured city (Village tile) than its origin is."""
    size = state.map_size()
    villages = [(i % size, i // size)
                for i in range(state.map_tiles())
                if state.tile_at(i).terrain == polyshark.TerrainType.Village]
    if not villages:
        return False

    def nearest(tile):
        x, y = tile % size, tile // size
        return min(max(abs(vx - x), abs(vy - y)) for vx, vy in villages)

    return nearest(action.dst) < nearest(action.src)


# ---------------------------------------------------------------------------
# Small utility — tanh without pulling in numpy for one call.
# ---------------------------------------------------------------------------

def _tanh(x: float) -> float:
    import math
    return math.tanh(x)
