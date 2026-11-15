# Attention

The entity tokens (128-dim units + cities, see [embedding.md](embedding.md)) form an
unordered set. Attention lets each token gather information from every other token —
so a unit's representation can be conditioned on nearby enemies, its city, threats, etc.

## Q, K, V

Each token `x` (128-dim) is projected into three vectors via learned matrices:

- **Query** `q = W_Q x` — "what am I looking for?"
- **Key** `k = W_K x` — "what do I offer / how am I described?"
- **Value** `v = W_V x` — "what info I pass on if attended to"

## How it works

For a token `i`, compare its query against every token's key to get a relevance score,
softmax into weights, then take the weighted sum of values:

```
scores_ij = (q_i · k_j) / sqrt(d_k)     # dot-product similarity, scaled
weights_i = softmax_j(scores_ij)         # sum to 1 over all j
out_i     = Σ_j weights_i,j · v_j        # weighted blend of values
```

- `sqrt(d_k)` scaling keeps dot products from growing with dimension (stabilizes softmax).
- Every token attends to every other (including itself) → each output token is a
  context-aware mix of the whole board's entities.

## Multi-head

A single attention gives each token **one** relevance pattern — one softmax that must
blend every concern (threats, friendly support, the capital) into a single averaged
weighting. That's lossy. Multi-head runs `h` attentions in parallel so several patterns
are computed at once:

- Split the `d`-dim token into `h` slices of `d/h` each.
- Each **head** has its own `W_Q/W_K/W_V` and does full scaled-dot-product attention on its slice.
- **Concat** the `h` head outputs back to `d`, then a final `Linear(d → d)` mixes them.

Because each head learns different projections, heads can specialize (one on enemies in
range, one on friendly cities, etc.) — richer than one blurred average, at ~the same cost.

```
head_i    = softmax(Q_i K_iᵀ / √(d/h)) V_i        # i = 1..h, each on a d/h slice
MultiHead = Linear( concat(head_1, …, head_h) )   # concat → d, then mix
```

Single-head matrix form: `Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V`.

**Config:** `d = 128`, `h = 4` heads → each head is `d/h = 32` dim.

## Transformer block

One block = two sublayers, each **pre-norm + residual**: attention, then feed-forward.

**Sublayer 1 — multi-head attention**
1. 4 heads each produce a 32-dim output; concat → 128.
2. **Output projection `W_O` (128→128).** Not optional — after concat the 128-dim vector
   is just four independent 32-dim slices side by side. `W_O` mixes the heads back
   together (blends e.g. "threat info" and "support info" into one representation).
   So the sublayer output is `concat → W_O`.
3. **Residual add:** `x = x + W_O(concat(heads))`. Add, don't replace — preserves the
   token's own identity and lets gradients flow, which is what makes stacking trainable.

**Sublayer 2 — feed-forward MLP** (per token, applied identically to each)
- Two-layer expand-then-contract: `128 → 512 → 128` with GELU between. The 4×d expansion
  gives per-token processing room to work; it is **not** a flat 128→128.
- **Residual add** again.

**Pre-norm:** LayerNorm is applied to the input *before* each sublayer. Full block:

```
x = x + MultiHeadAttention(LayerNorm(x))    # relational mixing between tokens
x = x + FeedForward(LayerNorm(x))           # per-token processing, 128 → 512 → 128
```

## Stack

**2 blocks**, applied in sequence. Block 1's output tokens are block 2's input.

Two blocks give two-hop tactical reasoning (a token can attend to a neighbor that has
already gathered info from *its* neighbors), and are cheap enough to keep search fast.
