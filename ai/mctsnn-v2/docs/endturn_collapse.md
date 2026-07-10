# End-Turn Collapse — Analysis & Fixes

**Status:** diagnosed 2026-07-08. Applied so far: value-loss subsampling per game,
AdamW weight decay + value-head input dropout, held-out validation seeds with a
`val_value_loss` metrics column, more games per gen (actionable #2 below); forced
end_turn states skipped agent-free in the arena (no sample for either head) and
single-choice stages skipped in the policy loss (actionable #3); staged per-stage
search budgets (see mcts.md "Staged commitment"); turn-exhaustion + position-summary
globals (actionable #4).

**2026-07-09 — actionable #1 applied** after the `guards` run (50 gens, all guards on)
still converged to passing: no memorization (V calibrated ~0, priors near-uniform), but
dV within a turn → 0.0000 and V → 0 everywhere — the mutual-passing fixed point of the
margin labels, reached cleanly. Fixes: **±1 winner-at-cap labels**
(`make_winner_terminal_value`, `turn_cap_winner`/`winner_dead_zone` config — value head
now estimates win probability, which is steep where expected-margin is flat);
**gen-0 search scale decoupled from labels** (`gen0_search_scale=1.0`, sharp bootstrap);
**mixed value targets** (`search_value_weight=0.3`, annealed over 10 gens): per-state
targets `(1−w)·z + w·v̂` give within-game credit and damp the recurring "unplayed states
drift negative → played even less" opening-move oscillation;
**horizon curriculum** (`turn_cap_start`/`turn_cap_grow`): games start short (cap ~5,
where one action is a big share of the outcome — strong per-action value gradient, and the
objective is crisp: out-explore / grab a village) and the cap grows per gen toward
`turn_limit`, stretching the horizon as play earns it;
**exploration + aggression terms in `heuristic_score`**: +0.005/explored tile (nerfed from 0.015 after corner-seeking/village-abandonment emerged — exploration must be a tiebreaker, not a win condition; kept small
— the L1 Explorer upgrade reveals in bulk, so reveal count is an information reward, not
a movement reward) and an enemy-capital proximity term (4/(1+dist) for the nearest own
unit once the capital is visible ≈ a village when adjacent) that pulls troops toward the
Domination win condition — movement is encouraged by proximity-to-objective terms
(villages, then the capital), not by fog cleared.
**2026-07-09 (later) — anti-collapse machinery batch**, after the mix03 run's
label-saturation collapse (exploration term dominated labels, map saturates by turn ~12,
labels became seat noise at cap 13+, P(end) prior broke its 0.25–0.30 band at gen ~70 and
locked by gen 90 — see "the prior always drifts toward end_turn; the value signal holds it
up"). New machinery, all in `run_training`:
**gating** (AlphaGo-Zero evaluator: incumbent generates data, candidate promoted only on
gate_games greedy wins ≥ threshold — degeneration can't compound);
**tie contempt** (`winner_tie_value` ≈ −0.2: mirrored passivity strictly loses);
**early KL anchor** (gens 1..N type head pulled to the frozen post-gen0 policy);
**policy target pruning** (KataGo-style: exploration-forced visit floor subtracted from
recorded targets — removes the marginal-drift gradient at its source, `mcts.prune_targets`);
**D8 value symmetry augmentation** + relaxed value subsampling (~20x effective value data,
map-fingerprint memorization structurally dead); **LR cosine decay** (`lr_final`);
**exploration term nerfed to tiebreaker** (0.005/tile). Curriculum guidance: hold the cap
at/below ~12 until decisive play exists; watch P(end)'s 0.25–0.30 band in the gen sweep.

**Run analysed:** `data/checkpoints/decaying_to_end_turn` (20 gens, config in its `run_config.json`).

Later generations converge on playing `end_turn` immediately every turn (gen19 greedy
self-play: 16 actions, all `EndTurn`). This doc records why, with the empirical evidence
from the collapsed run's checkpoints, and the prioritized fixes. Related docs:
[training.md](training.md), [mcts.md](mcts.md), [value_head.md](value_head.md).

## TL;DR

The collapse is a feedback loop between three design choices — **not a search bug** (the
search demonstrably concentrates 90/100 visits on a productive action whenever the value
net gives it a real Q gap):

1. **`end_turn` is the only action whose Q is exactly the root's own value**
   (`mcts._endturn_leaf`: turn-local search, `Q(end) = V(s)`, zero variance). Whenever the
   value head is flat within a turn, end_turn ties-or-beats everything and the tie-break
   falls to the policy prior.
2. **The policy prior drifts toward end_turn** because ~40% of all type-stage target mass
   is end_turn: every player-turn necessarily terminates with an end-dominant sample, end
   collects residual visits at every other decision, and in ~13% of samples end is the
   *only* legal type (a pure marginal-shift gradient carrying zero information).
3. **No game ever ends decisively** (`decisive=0` in all 20 gens of `metrics.csv` at
   `turn_limit=30`), so the training signal never once punishes passivity. Every value
   label is a per-game-constant turn-cap heuristic margin; with 8 games/gen the value head
   memorizes game identity instead of learning, and is flat (±0.05) within a turn. Mutual
   passing is a stable fixed point of this reward.

The bot is correctly optimizing the game it is actually being trained on — which is not
Polytopia.

## The mechanism, end to end

1. Gen0 (heuristic bootstrap) plays reasonably, but its samples already carry ~40%
   end_turn mass at the type stage (measured: mean 0.41, 24% end-dominant, 13% forced).
2. The value head, trained on per-game-constant labels from 8 games/gen, memorizes rather
   than generalizes: within-turn `V(s') − V(s) ≈ 0 ± noise`. Falling `value_loss`
   (0.25 → 0.02) is memorization, not calibration — at the same symmetric turn-0 state,
   V(s) swings −0.91 → +0.91 → −0.67 across generations (true value ≈ 0).
3. In the next gen's search, `Q(end_turn) = V(s)` exactly, while every other type's Q is a
   *subtree average* over its entity/tile children — 100 sims over hundreds of leaves,
   dragged down by pointless targets. Flat value ⇒ end_turn ≥ everything.
4. With Q flat, PUCT visits ≈ priors; priors favor end_turn; the played action is
   end_turn; the new samples are even more end-heavy; the prior grows. Runaway.
5. Once both players pass constantly, two extra effects lock it in:
   - **Turn-cap labels go to ~0** (mirror passing ⇒ no margin) → value head trends flat →
     even less Q signal.
   - **Mid-turn states vanish from the data** (a passing player's only recorded state per
     turn has all units at full move points), so post-move states become
     out-of-distribution — by gens 13–14 the net *actively penalizes* moving (ΔV ≈ −0.37).

### Why end_turn is structurally unpunishable here

The search is turn-local: it never models the opponent's reply, so passing carries zero
modeled tempo cost — `end_turn` is *defined* to be worth the status quo. The only channel
through which "passing is bad" could enter is the value labels, and:

- Decisive games never occur (capitals unreachable at `turn_limit=30`, `n_sims=100`), so
  the actual win condition contributes zero gradient, ever.
- The turn-cap label is one constant shared by all ~180 states of a game — no temporal
  structure to attribute blame to passive decisions.
- In self-play, if *both* players pass, margin ≈ 0, predicted perfectly by a flat value
  head. Mutual passing is literally an equilibrium of the training signal.

In full AlphaZero, passing is punished through the tree (the opponent's extra tempo backs
up as a worse value) — but only if search crosses `end_turn` into the opponent's turn,
which this prototype deliberately doesn't (see [mcts.md](mcts.md)).

## Evidence (from the collapsed run's checkpoints)

Probe scripts measured, at a fresh turn-0 state (seed 999) and a developed mid-game state:

**P(end_turn) prior across gens** — fresh state (3 legal types, uniform = 0.33):
0.205 (gen0) → 0.36 (gen3) → 0.40 (gen10) → 0.43–0.48 (gens 11–19), near-monotone.
At the mid-game state (6 legal types, uniform = 0.167) it stays ~0.19: the head learned
the *marginal frequency* of end_turn, applied everywhere. The bias is most decisive at
low-branching states — i.e. the start of a turn, hence "ends turn as soon as it's its
turn". `policy_loss` frozen at ~1.05 for 20 gens = it fit the marginal in gen1 and
learned nothing since.

**Value flatness** — from gen5 on, |V(s′) − V(s)| < ~0.05 for nearly all legal actions at
both probe states (vs gen0-heuristic signal of +0.58 mean / +1.71 max). Gens 13–14:
moves penalized by −0.37 (post-move states are OOD by then).

**Search dumps** (100 sims, no noise, fresh state; root type visits with Q and prior):

| gen | Q(move) | Q(research) | Q(end)=V(s) | visits m/r/e | plays |
|---|---|---|---|---|---|
| 1  | **+0.297** | −0.328 | −0.326 | **86**/8/6  | Move |
| 5  | −0.917 | −0.913 | −0.913 | 24/37/**39** | **EndTurn** |
| 10 | **+0.149** | −0.837 | −0.838 | **91**/4/5  | Move |
| 15 | −0.748 | −0.691 | −0.668 | 16/31/**53** | **EndTurn** |
| 19 | −0.253 | −0.280 | −0.240 | 26/27/**47** | **EndTurn** |

Gens 1/10: a real Q gap ⇒ search concentrates and plays the productive action (search is
fine). Gens 5/15/19: Q identical to 2–3 decimals ⇒ visits split ≈ priors ⇒ end wins on
prior — and at 15/19 Q(end) is strictly highest (exact V(s) vs subtree averages dragged
down by OOD-penalized moves). Gen10's +0.149-vs-−0.838 jump from moving one warrior is a
memorization artifact — such flukes are why the collapse wasn't monotone.

## Secondary issues found along the way

- **Q-init asymmetry** (`mcts._puct`): unvisited edges get q=0, so when V(s) > 0 end_turn
  (q=V(s)) beats every unexplored action for the first ~5 sims; reversed when V(s) < 0.
  The "winning" player is systematically biased toward instant passing.
- **Forced samples train the prior**: when a stage has one legal choice (13% of type-stage
  samples are end-only), its CE target is 100% that choice — pure marginal drift, zero
  information (the mask already forces it).
- **Train/search mask mismatch**: `trainer._policy_loss_for_sample` rebuilds
  `FactoredActions` without `root_visible`, so train-time masks can include fog-target
  actions the search-time visit targets never saw (softmax support ≠ target support). Minor.
- **tanh saturation**: at `heuristic_scale=0.75`, one captured village (Δ=4) →
  tanh ≈ 0.95; labels are effectively ternary {−1, 0, +1}, encouraging memorization.
- **Harvest is invisible to gen0**: `heuristic_score` counts cities/levels/techs/units/
  village-proximity but not population or stars, so harvesting scores exactly like passing
  even under the bootstrap evaluator.

## Actionables (by leverage)

1. **Get an "acting beats passing" signal into the labels — declare a winner at the turn
   cap.** Natural decisive games (capital captures) are unreachable early, so don't wait
   for them: label turn-capped games **±1 by the sign of the heuristic margin** (0 only
   inside a small dead zone), instead of `tanh(margin)`. Mutual passing stops being safe:
   any exploration-induced asymmetry means the more active player banks +1 and the
   passive one −1. This alone should break the collapse — the search exploits any real
   Q gap it is given (see gens 1/10 above).
2. **Stop value-head memorization** (ranked):
   1. Subsample positions per game for the *value* loss (≤ ~8–16 per game; within-game
      samples are near-duplicates of one label — AlphaGo used 1/game for this reason).
      Policy loss keeps using everything.
   2. More games per gen (8 → 50+, drop n_sims to ~60 to pay for it) and cap reuse:
      currently 100 steps × 32 = 3200 draws over ~950 new samples/gen (~3.4× each);
      aim for each sample seen ~1–2× per gen.
   3. Weight decay: Adam → AdamW ~1e-4 (trunk + value head at minimum).
   4. Held-out validation seeds excluded from the buffer; track value MSE on them each
      gen — the train/val gap is the memorization meter (train loss alone is a red
      herring; the board's terrain planes make each seed a memorizable fingerprint).
   5. 8-fold board-symmetry augmentation (dihedral; requires remapping tile indices).
   6. Later, once search values mean something: mixed targets λ·z + (1−λ)·(MCTS root
      value) to give labels per-state structure instead of per-game structure.
3. **Skip single-choice stages in the policy loss**: in `_policy_loss_for_sample`, skip
   any stage whose legal mask has exactly one choice (covers forced end_turn — 13% of
   type-stage samples — and forced entity/target stages). Zero information, pure
   marginal-drift gradient. Keep the sample for the value loss, and keep genuine
   multi-choice end decisions (correct, conditioned targets).
4. **Make turn-exhaustion trivially visible to the heads**: add globals like *fraction of
   my units with moves remaining* and *actions taken this turn* (`encode_globals`). The
   per-unit signal exists (`move_points`, `has_attacked`) but must survive attention +
   pooling into the 256-d core; explicit globals make "end only when exhausted" easy to
   condition on.
5. **FPU = parent value** in `_puct` (init unvisited q to V(parent), not 0) to remove the
   sign-of-V passing bias.
6. **Longer term: model the opponent's reply so tempo has a cost.** Less blocked by
   imperfect information than it looks: explored tiles are permanently revealed
   (including units), so the only hidden info is unexplored tiles — determinize those
   (assume empty / sample) and run the opponent's turn with the same net
   (ISMCTS-style), or, cheaper, score end_turn by applying it and letting the opponent
   play one greedy policy turn before evaluating (1-turn tempo rollout). An accuracy
   upgrade, not the unlock — fix the value signal first.

Also worth fixing while in there: add population + stars (small weights) to
`heuristic_score` so econ actions register in the bootstrap; align train-time
`FactoredActions` masks with search-time (`root_visible`).
