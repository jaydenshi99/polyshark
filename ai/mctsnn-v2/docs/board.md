# Board & Globals — Inputs

The non-entity half of the state. Two raw inputs plus a fused one:

1. **Base board grid** `[C, 11, 11]` — everything on the grid that isn't a unit/city
   (terrain, resource, building, border, fog). Fed to the main conv tower.
2. **Globals vector** `[G]` — non-spatial scalars (turn, stars, income, techs, phase).
   Fed to a small MLP / merged into the conv trunk.
3. **Scattered entity tokens** — the post-attention entity tokens (see
   [attention.md](attention.md)) written back onto the grid at each entity's tile, then
   concatenated onto the base board as extra channels before the conv.

Perspective and fog match [features.py](../features.py): current player's view, me/opp, and a
tile shows content only if `is_visible` (== explored in this engine — permanent reveal).

---

## 1. Base board grid — `[18, 11, 11]`

Every channel is 0 on an unexplored tile; the only signal there is fog = 0.
No entity info here — units/cities arrive via scatter (§3).

```
 Ch   Feature                     Source
────────────────────────────────────────────────────────────
  0   terrain: Field              tile.terrain (one-hot)
  1   terrain: Forest
  2   terrain: Mountain
  3   terrain: Water
  4   terrain: Village            uncaptured only (→ Field on capture)

  5   resource: Fruit             tile.resource (binary; 0 once harvested)
  6   resource: Crop
  7   resource: Animal
  8   resource: Metal

  9   my_Mine                     tile.building, owner via border_city_id → city.owner
 10   opp_Mine
 11   my_Farm
 12   opp_Farm
 13   my_Road
 14   opp_Road

 15   my_border                   tile.border_city_id → city.owner == me
 16   opp_border

 17   is_visible                  fog gate (== explored); 0 = unexplored, all else 0 too
```

Notes:
- Building ownership follows `border_city_id` and transfers on city capture.
- `tile.capture_ready` (village) is omitted — recoverable from Village terrain + a
  scattered friendly unit; add a channel later if it proves useful.

---

## 2. Globals vector — `[12]`

Per-player, non-spatial, current player only (opponent stars/income/tech are not
observable — imperfect information).

```
 Idx  Feature              Norm
──────────────────────────────────────────
  0   turn                 / 100
  1   my_stars             / 30
  2   my_income            / 20   (Σ city.stars_per_turn, mine)
  3   tech: Hunting        bit
  4   tech: Organisation   bit
  5   tech: Farming        bit
  6   tech: Riding         bit
  7   tech: Climbing       bit
  8   tech: Archery        bit
  9   tech: Mining         bit
 10   tech: Strategy       bit
 11   phase: UpgradingCity bool   (gates legal actions)
```

`Origin` tech dropped (always owned). 8 meaningful techs (Hunting…Strategy).

---

## 3. Entity scatter

After the entity transformer produces context-mixed tokens `[N, 128]`, **project each
token down to 16** and write it onto the grid at its tile, giving spatial planes the conv
can fuse with terrain.

- **Project before scatter:** the 128-dim width earned its keep during attention
  (relational reasoning); for the conv the token only needs a compact "an entity like
  this is here" summary. A single learned `Linear(128 → 16)` per entity type — separate
  weights for units and cities, since they compress differently.
- Two separate planes — a **unit plane** and a **city plane**, each `[16, 11, 11]`,
  zero-initialised. A unit can stand on a city tile (same row,col), so one combined
  plane would collide; separate planes avoid it.
- Place token at `(row, col) = divmod(tile_index, 11)`. Empty tiles stay zero → conv
  reads "no entity here." (Scatter planes are sparse — ~10 of 121 cells occupied.)
- Requires per-entity tile indices from the encoder: `features.encode_entities` must
  also return `unit_tiles [Nu]` / `city_tiles [Nc]` (currently it returns only feats).

### Composed input to the main conv

```
concat along channels:
  base board      18
  unit plane      16   (post-attention unit tokens, projected 128→16)
  city plane      16   (post-attention city tokens, projected 128→16)
  ─────────────────
  main conv in    50   →  [50, 11, 11]
```

The conv body is a stem (Conv 50→64) + 3 ResBlocks at 64 channels, all 3×3 pad=1
(zero-padded edges, no spatial shrink — stays 11×11), then global-average-pooled to [64].

A `LayerNorm` after the scatter projection is optional — add it only if scattered values
are badly scaled against the terrain one-hots (the conv's own norm usually handles it).

Globals `[12]` enter separately — broadcast across the grid as extra channels, or merged
into the conv trunk after pooling (architecture TBD; this doc only fixes the inputs).
