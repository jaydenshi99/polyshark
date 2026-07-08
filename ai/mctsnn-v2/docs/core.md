# Shared Core MLP

The fusion point: the residual board body and the globals meet here and become a single
256-d **shared representation** that every head (value now, policy later) branches off.

## Inputs

- **Pooled board** `[128]` — the `64×11×11` output of the residual body (stem + 3
  ResBlocks, see [board.md](board.md)) reduced by **both** average and max pooling,
  concatenated:
  - avg-pool `[64]` — the board's overall state (how much of what, on average).
  - max-pool `[64]` — the strongest signal anywhere (is there a critical tile at all).
  - Together they capture "general position" and "sharp local feature" — complementary,
    and cheap.
- **Globals** `[12]` — raw non-spatial scalars (turn, stars, income, techs, phase),
  see [board.md](board.md) §2. Concatenated directly, no lift.

```
board body [64,11,11] ── avg-pool ──> [64] ┐
                      └─ max-pool ──> [64] ┼─ concat ─> [128] ┐
globals ─────────────────────────────────── [12] ───────────┼─ concat ─> [140]
```

## Core MLP

Two `Linear → LayerNorm → GELU` blocks, `140 → 256 → 256`:

```
[140] ─ Linear(140 → 256) → LayerNorm → GELU
      ─ Linear(256 → 256) → LayerNorm → GELU
      └─> shared representation [256]
```

- **LayerNorm after each Linear** stabilizes the fused stream (board features and raw
  globals live on different scales; norm equalizes them).
- **GELU** matches the rest of the network.
- Output `[256]` is the shared representation — deliberately head-agnostic, so value and
  policy read from the same fused understanding of the position.

## Heads (branch off the [256] shared rep)

**Value head** — small readout to a scalar in `(-1, 1)`. See [value_head.md](value_head.md).

**Policy head (TBD)** — spatial actions need **per-cell** logits, so the policy head does
*not* read the pooled `[256]`; it branches from the `64×11×11` feature map *before*
pooling. Designed later with the action codec. (The `[256]` core rep may still feed the
policy head as global conditioning — decided then.)
