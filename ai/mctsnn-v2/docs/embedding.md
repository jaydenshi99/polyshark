# Entity Embedding — Fields

Ground truth from engine headers. Two entity types to embed: **Unit** and **City**.

## Unit

**Instance state** (per-unit, mutable) — [unit.h](../../../engine/include/unit.h), all exposed in bindings:

| Field | Type | Range | Note |
|---|---|---|---|
| `type` | categorical | Warrior/Archer/Rider/Defender/Giant | key to all static stats |
| `owner` | categorical | 0/1 | encode as me/opp, not player id |
| `tile_index` | positional | 0…120 | spatial, not a scalar |
| `hp` | scalar | 0…max_hp | normalize by max_hp |
| `max_hp` | scalar | 10 / 15 (vet) / 40 (Giant) | encodes veteran status |
| `move_points` | scalar | 0…base_move | normalize by base move |
| `kills` | scalar | 0…3+ | promotion at 3 |
| `has_attacked` | bool | 0/1 | turn state |
| `promotion_ready` | bool | 0/1 | true at 3 kills |
| `city_id` | reference | −1 or city id | relational link, not a scalar |
| `last_dir` | categorical | −1, 0…7 | 8-dir, one-hot |

**Type-derived stats** (constant per `type`, from [unit_def.h](../../../engine/include/unit_def.h)): `hp`, `attack`, `defense`, `movement`, `range`, `cost`, `abilities`, `required_tech`. Attack/defense are NOT on the unit — only reachable via type. Choose one: embed `type` (net learns stats) OR expand into raw stats + drop type.

**Abilities** (5 flags, function of type): FORTIFY, DASH, ESCAPE, RANGED, STATIC.

**Notes:**
- No `is_veteran` field — derive as `max_hp − base_hp`.
- Fortify bonus is tile-conditioned (needs terrain under unit: forest/mountain/city).

## City

**Instance state** — [city.h](../../../engine/include/city.h):

| Field | Type | Range | Exposed | Note |
|---|---|---|---|---|
| `owner` | categorical | 0/1 | ✅ | me/opp |
| `level` | scalar | 1…∞ | ✅ | no cap — soft-normalize |
| `population` | scalar | 0…threshold | ✅ | progress to next level |
| `tile_index` | positional | 0…120 | ✅ | spatial |
| `border_radius` | categorical | 1 or 2 | ✅ | 2 at level 4+ |
| `units_owned` | scalar | 0…capacity | ✅ | alive trained units |
| `parks` | scalar | 0…n | ❌ | not in bindings |
| `is_capital` | bool | 0/1 | ✅ | win target |
| `is_sieged` | bool | 0/1 | ✅ | zeroes income |
| `capture_ready` | bool | 0/1 | ✅ | capturable this turn |
| `has_walls` | bool | 0/1 | ✅ | defense bonus |
| `pending_upgrade` | bool | 0/1 | ✅ | awaiting upgrade pick |
| `has_workshop` | bool | 0/1 | ✅ | income upgrade |

**Computed:** `stars_per_turn` = level + (capital?1:0), 0 if sieged · `unit_capacity` = level+1 · `can_spawn` = units_owned ≤ level.

**Gaps:**
- `parks` not exposed to Python — needs a binding to include it.
- Pending upgrade *choice* isn't a city field; only `pending_upgrade` bool exists. The actual options surface via `legal_actions()` (UpgradeCity) when `phase == UpgradingCity`.

## Cross-cutting

- **Relational, not scalar:** `unit.city_id`, both `owner` fields — suit graph/attention over flat vectors.
- **Tile-conditioned, not on entity:** unit fortify (needs terrain), city `capture_ready` (needs a unit on it).
- Everything reachable from Python **except `city.parks`**.

---

# Design Choices (v1)

Each entity → one **token**. Unit and city tokens are built from a learned type/kind
embedding ⊕ dynamic instance features. Static per-type constants are NOT passed raw —
the embedding subsumes them.

## Unit token = `type_embedding` ⊕ instance features

- **type** → learned embedding, 8 dim. Do NOT expand to raw stats (atk/def/mov/rng)
  or ability flags — all constant per type, embedding captures them + latent quirks.
- **owner** → binary me/opp. No unknown-owner class (permanent-reveal fog, no ghosts).
- **tile_index** → raw (row, col) scatter address; never categorical.
- **hp** → `hp / max_hp`.
- **is_veteran** → derived `max_hp > unit_def(type).hp` (Python; no engine edit). Replaces raw max_hp. Always 0 for Giant (STATIC).
- **move_points** → `move_points / base_move` (base_move: Rider=2, else 1).
- **kills** → `min(kills / 3, 1)` (clamp — kills counts past 3).
- **has_attacked** → 0/1.
- **promotion_ready** → 0/1 (masking-relevant).
- **Dropped:** raw max_hp, city_id, last_dir, ability flags.

## City token = instance features (no type — single kind)

- **owner** → binary me/opp (same as unit).
- **level** → `log(1 + level)` (unbounded).
- **population** → `population / (level + 1)` (threshold confirmed [city.cpp:18](../../../engine/src/city.cpp#L18)).
- **tile_index** → raw (row, col).
- **border_radius** → boolean (1→0, 2→1; expands at level 4).
- **units_owned** → `units_owned / unit_capacity` (cap = level+1).
- **Booleans (0/1):** is_capital, is_sieged, capture_ready, has_walls, pending_upgrade, has_workshop.
- **Computed kept:** `can_spawn` (masking-relevant).
- **Dropped:** parks, stars_per_turn (redundant — derivable from level+capital+sieged).

## Token dimensions

**Unit token = 17**

| Component | Dim |
|---|---|
| type embedding | 8 |
| owner (me/opp) | 1 |
| row, col | 2 |
| hp / max_hp | 1 |
| is_veteran | 1 |
| move_points / base_move | 1 |
| kills (clamped) | 1 |
| has_attacked | 1 |
| promotion_ready | 1 |

**City token = 14**

| Component | Dim |
|---|---|
| owner (me/opp) | 1 |
| row, col | 2 |
| log(1 + level) | 1 |
| population / (level+1) | 1 |
| border_radius (bool) | 1 |
| units_owned / capacity | 1 |
| is_capital | 1 |
| is_sieged | 1 |
| capture_ready | 1 |
| has_walls | 1 |
| pending_upgrade | 1 |
| has_workshop | 1 |
| can_spawn | 1 |

## Projection to shared dim

Unit (17) and city (14) tokens have different widths. Each entity type has its own
projection head mapping its raw vector to a shared model dim `d = 128`:

```
Linear(in → 128) → GELU → Linear(128 → 128) → LayerNorm
```

- Unit head: `in = 17`
- City head: `in = 14`

Both output 128-dim tokens, ready for attention.
