# Heads — Fusion MLP & Value

Where the residual board body meets the globals and produces the value scalar.
(Policy head is separate — see the bottom.)

## Inputs

- **Pooled board embedding** `[64]` — global-average-pool over the `64×11×11` output of
  the residual body (stem + 3 ResBlocks, see [board.md](board.md)). Pooling collapses the
  spatial map into one whole-board summary, which is what a scalar value wants.
- **Globals** `[12]` — non-spatial scalars (turn, stars, income, techs, phase),
  see [board.md](board.md) §2. Carries context the board grid physically can't.

## Fusion

Globals are first lifted by a small encoder so 12 raw dims aren't swamped by the 64 board
dims, then concatenated:

```
globals [12] ──Linear(12 → 32) → GELU──> [32]
pooled  [64] ─────────────────────────── [64]
                     concat ───────────> [96]
```

Open call: skip the global encoder and concat raw (`64 + 12 = 76`) for simplicity. Kept
the `12→32` lift — cheap, and balances the two streams. Easy to drop if it doesn't earn it.

## Value MLP

```
[96] ─ Linear(96 → 128) → ReLU
     ─ Linear(128 → 128) → ReLU
     ─ Linear(128 → 1)   → Tanh
     └─> value ∈ [-1, 1]     (current player's expected outcome)
```

- Two hidden layers at 128 — enough to mix board summary + globals without over-sizing a
  scalar regressor.
- `Tanh` bounds the output to `[-1, 1]`: `+1` = current player winning, `-1` = losing.
  Matches the MCTS backup and the win/loss training target.
- No dropout/BN in the head — value nets are low-variance here; add LayerNorm only if
  training is unstable.

## Policy head (TBD)

Does **not** hang off this MLP. Spatial actions (move/attack/etc.) need **per-cell**
logits, so the policy head branches from the `64×11×11` feature map *before* pooling —
likely a conv → per-tile action logits, reconciled with the engine action space by the
action codec. Designed alongside that codec.
