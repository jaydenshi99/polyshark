# Polyshark — Game Spec

A stripped-down Polytopia-inspired turn-based strategy game.

---

## Map

- Square grid (suggested 11×11 for 2 players)
- Terrain types: **field**, **forest**, **mountain**, **water**
- Fog of war — unexplored tiles hidden
- Villages scattered across map — capturable, become cities
- One capital per player (level 1 city, pre-revealed)

### Terrain properties

| Terrain | Movement | Defence Bonus |
|---|---|---|
| Field | Normal | ×1.0 |
| Forest | Blocks (enter only) | ×1.5 |
| Mountain | Impassable without Climbing tech (enter only) | ×1.5 |
| Water | Impassable | — |
| Village | Normal | ×1.0 (terrain changes to Field on capture; city placed by engine) |

### Vision
- Units reveal all 8 surrounding tiles (3×3 centred on unit)
- Mountains extend vision further (exact radius TBD)
- Cities reveal surrounding 3×3 tiles
- Explored tiles stay permanently revealed; once a tile is uncovered it stays uncovered forever, and any unit on it is always visible to the explorer

---

## Stars (Currency)

- Collected at the start of each turn from all owned cities
- Spent on techs, units, and harvesting

---

## Cities

| Property | Rule |
|---|---|
| Income | 1★/turn per level |
| Capital bonus | +1★/turn |
| Unit capacity | level + 1 |
| Upgrade requirement | n population to reach level n+1 |
| Level cap | None |

### Levelling up
Gaining enough population levels the city up automatically. Every level-up gives +1★/turn and presents the player with **two upgrade choices** (must be resolved before any other actions that turn).

| Level transition | Option A | Option B |
|---|---|---|
| 1 → 2 | Workshop | Explorer |
| 2 → 3 | Resources (+5★) | Walls |
| 3 → 4 | Border Growth (3×3 → 5×5) | Population Growth (+3 pop) |
| 4+ | Park | Superunit |

### Border
- Levels 1–3: 3×3 area centred on city tile (radius 1, up to 8 harvestable tiles)
- Level 4+: 5×5 area centred on city tile (radius 2, up to 24 harvestable tiles)
- Borders between two cities are clipped at the midpoint — no overlapping

### Population sources
Harvesting resources on tiles within the city's border.

### Siege
Enemy unit on a city tile → city produces 0★ that turn.

---

## Resources & Harvesting

Resource sub-types (used in code): `Fruit` (field), `Animal` (forest), `Metal` (mountain).

### Food — Fruit (Fields)
- Requires **Organisation** tech
- Cost: **2★**
- Reward: **+1 population** to nearest city
- One-time — tile cleared after harvest

### Food — Animal (Forests)
- Requires **Hunting** tech
- Cost: **2★**
- Reward: **+1 population** to nearest city
- One-time — tile cleared after harvest

### Metal (Mountains)
- Requires **Mining** tech
- Cost: **5★**
- Reward: **+2 population** to nearest city
- Places a **Mine** building on the tile after harvest

### Rules
- Resource must be within the city's border
- Cannot harvest a tile occupied by an enemy unit

---

## Tech Tree

| Tech | Cost | Prereq | Unlocks |
|---|---|---|---|
| Hunting | 5★ | — | Harvest Animal (forest food); unlocks Archery |
| Organisation | 5★ | — | Harvest Fruit (field food) |
| Riding | 5★ | — | Train Rider |
| Climbing | 5★ | — | Units may enter mountain tiles; unlocks Mining |
| Archery | 6★ | Hunting | Train Archer |
| Mining | 6★ | Climbing | Harvest Metal (mountain ore) |

Warrior requires no tech — available from turn 0.

---

## Units

| Unit | HP | ATK | DEF | MOV | Cost | Skills |
|---|---|---|---|---|---|---|
| Warrior | 10 | 2 | 2 | 1 | 2★ | Fortify, Dash |
| Archer | 10 | 2 | 1 | 1 | 3★ | Ranged (2 tiles), Fortify |
| Rider | 10 | 2 | 1 | 2 | 3★ | Escape, Fortify, Dash |

### Skills
- **Fortify** — ×1.5 defence bonus in forests, mountains, and cities
- **Dash** — can move after a non-attack action
- **Escape** (Rider) — can move again after attacking
- **Ranged** (Archer) — attacks up to 2 tiles away; does not move after kill; cannot retaliate against melee

### Healing
A unit that neither moves nor attacks recovers HP at the start of its next turn:
- **4 HP** if the tile is within a friendly city's border
- **2 HP** anywhere else
- Capped at max HP. Applies to all standard units.

### Veteran promotion
Kill 3 enemies → unit becomes promotion-ready. Player can accept at any time: +5 max HP, full heal.

---

## Combat

### Damage formula
```
attackForce  = ATK × (currentHP / maxHP)
defenceForce = DEF × (currentHP / maxHP) × defenceBonus

total = attackForce + defenceForce

attackResult  = round((attackForce  / total) × ATK × 4.5)
defenceResult = round((defenceForce / total) × DEF × 4.5)
```
- Attacker takes `defenceResult` damage
- Defender takes `attackResult` damage
- Minimum 1 damage
- Melee kill → attacker moves into target tile
- Archer kill → attacker stays put

### Zone of Control
Moving adjacent to an enemy unit costs all remaining movement.

---

## Movement

- Entering forest or mountain uses all remaining movement
- Exiting forest or mountain has no penalty

---

## Turn Structure

1. **Collect stars** from all owned cities
2. **Move and attack** with any units in any order
3. **Spend stars** — harvest, research tech, train units
4. **End turn**

---

## Win Condition

**Domination** — capture all enemy capitals.

---

## Implementation Notes

### Fog of War
Per-player `explored` bitfield — once set, never cleared. A tile is only hidden if it's never been explored. Units on explored tiles are always shown, even when no friendly unit/city currently has line of sight on them. The `visible` bitfield exists as a transient marker but is not used to gate display or attack targeting.

### Tech Unlocks
Each player's researched techs are stored as a `uint32_t` bitmask, one bit per tech. Checking a tech: `has_tech(player, TechType::Mining)`.

### Unit Stats Lookup
Static stats (HP, ATK, DEF, etc.) live in a global `UNIT_DEFS[]` table indexed by `UnitType`. Use `unit_def(type)` to look up a unit's base stats at runtime. Per-unit mutable state (current HP, kills, etc.) lives on the `Unit` instance in `GameState`.
