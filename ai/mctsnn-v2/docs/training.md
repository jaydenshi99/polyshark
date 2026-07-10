# Training — Self-Play + Trainer (AlphaZero-style)

How the network learns: an AlphaZero-style loop where **self-play generates data** and a
**trainer distills it back into the network**, connected by a **shared replay buffer** and
a **shared network**. This doc fixes the target architecture and the phased plan to get
there. It ties together [mcts.md](mcts.md) (the search that produces the data),
[policy_head.md](policy_head.md) / [value_head.md](value_head.md) (the training targets),
and the arena layer (`src/arena.py`, which already yields the samples).

## The two-process structure

AlphaZero is fundamentally **two loops** running concurrently, decoupled, connected only by
the buffer and the weights:

1. **Self-play workers** — continuously play games with the *current* network + MCTS,
   writing each finished game's samples into the **replay buffer**. Search uses root
   Dirichlet noise and the temperature schedule (exploration); see [mcts.md](mcts.md).
2. **Trainer** — continuously samples minibatches from the buffer, updates the network
   (policy + value losses), and **periodically publishes** the new weights back to the
   self-play workers (a new checkpoint).

```
   ┌─────────────────┐   samples    ┌───────────────┐   minibatches   ┌──────────┐
   │  self-play      │ ───────────▶ │ replay buffer │ ──────────────▶ │ trainer  │
   │  workers (MCTS) │              │  (conveyor)   │                 │ (SGD)    │
   └─────────────────┘              └───────────────┘                 └──────────┘
          ▲                                                                 │
          └──────────────── new weights (checkpoint) ◀──────────────────────┘
```

They run at the same time and **don't wait on each other**: self-play doesn't block on
training, training doesn't block on self-play. The buffer is the shared conveyor belt; the
checkpoint is the shared network. This decoupling is what gives throughput.

## Prototype plan: alternate first, concurrent later

The concurrent two-process version is a **throughput optimization**. For a solo prototype we
start with the equivalent **alternating, single-process** loop — simpler to implement, debug,
and reason about, and behaviourally close enough:

- **Phase A (start here) — alternation.** One process. Play a chunk of games → append to the
  buffer → train a chunk of minibatches → checkpoint → repeat. No concurrency, no IPC, fully
  deterministic given seeds. This is the "play a chunk, train a chunk" loop.
- **Phase B (later) — concurrent.** Split into self-play worker processes + a trainer
  process sharing the buffer and a weights file (or a small queue/dir). Same math, more
  throughput. Only build this once Phase A works end-to-end and self-play throughput is the
  bottleneck (it will be — the search is Python; see [mcts.md](mcts.md)'s C++ note).

Everything below is written for Phase A; Phase B changes *only* the scheduling, not the data
or the losses.

## The generation loop (Phase A)

```
net, policy = fresh or gen-0 bootstrap
buffer = ReplayBuffer(capacity=...)

for gen in range(N_GENS):
    ev = NetworkEvaluator(net, policy)                     # current weights, shared both seats

    # 1. SELF-PLAY  → samples  (reuses src/arena.py, collect=True)
    a = Agent("p0", MCTSStrategy(ev, add_noise=True, temperature=1.0, temp_turns=...))
    b = Agent("p1", MCTSStrategy(ev, add_noise=True, temperature=1.0, temp_turns=...))
    for i in range(GAMES_PER_GEN):
        res = Arena([a, b]).play_game(seed=..., collect=True)
        buffer.extend(res.samples)                         # each Sample gets its outcome at game end

    # 2. TRAIN  (new code: encode + losses + optimizer)
    for step in range(TRAIN_STEPS_PER_GEN):
        batch = buffer.sample(MINIBATCH)
        loss  = policy_loss(batch) + value_loss(batch)     # see below
        loss.backward(); opt.step()

    # 3. PUBLISH checkpoint
    save(f"data/checkpoints/gen{gen:03d}.pt")              # {"net":..., "policy":...}

    # 4. GATE new vs old  (reuses Arena.play_matches, add_noise=False, temperature=0)
    if win_rate(candidate, incumbent) >= GATE_THRESHOLD:
        incumbent = candidate
```

Steps 1 and 4 are **pure reuse** of the arena layer. Steps 2 and 3 are the genuinely new
machinery (encode → loss → optimizer → checkpoint) — the outer loop that does not exist yet.

## Components mapped to the codebase

| Concept | Where it lives | Status |
|---|---|---|
| Self-play worker | `Arena.play_game(collect=True)` → `res.samples` | ✅ exists |
| MCTS + exploration | `MCTSStrategy(add_noise, temperature)` over `mcts.MCTS` | ✅ exists |
| Current network as evaluator | `NetworkEvaluator(net, policy)` | ✅ exists |
| Training sample | `arena.Sample(state, player, action, targets, outcome)` | ✅ exists (in memory only) |
| Replay buffer | — | ❌ **to build** |
| Trainer (encode + losses + optimizer) | — | ❌ **to build** |
| Checkpoint publish/load | `{"net":…, "policy":…}` dict; loader in `run_games.py` | ✅ load side exists |
| Gating (new vs old) | `Arena.play_matches(add_noise=False, temperature=0)` | ✅ exists |

**Two entry points, one library.** `run_games.py` (human eval) and the future `train.py`
(the loop) are **siblings** that both import `src/arena.py`; `train.py` does **not** call
`run_games.py`. The spec→agent and checkpoint→evaluator helpers currently inside
`run_games.py` should be lifted to a shared `src/agents.py` so both use them.

## Training targets (recap — full detail in the head docs)

Per developed sample, from the **acting player's** perspective:

- **Inputs** — `features.encode_*(state, me=player, visible=player's fog)` → entities +
  board `[18,11,11]` + globals `[21]`.
- **Policy** — for each stage the chosen action fired (type → entity → target), the
  **normalized MCTS visit counts** of that level's edges. Loss = sum of the fired stages'
  cross-entropies vs the policy head's masked softmax. **Autoregressive:** each stage is
  scored conditioned on the earlier chosen sub-choices, so training replays the chosen path
  (recoverable from `Sample.action` via `FactoredActions`). See [policy_head.md](policy_head.md).
- **Value** — mixed target `(1−w)·z + w·v̂`: `z` is the game outcome (±1 decisive; ±1
  winner-by-margin at a turn cap), `v̂` is the search root value recorded at the decision
  (`Sample.search_value`). `w` (`search_value_weight`, ~0.3) is annealed from 0 over the
  first gens. Pure `z` gives every state of a game one shared label (no within-game
  credit); `v̂` differs per state and encodes what the search found, including unplayed
  lines. Validation always scores against pure `z`. Loss = MSE vs the value head.
- **Total** `L = Σ stage-CE(policy) + MSE(value) [+ weight decay]`.

## Regularization & anti-memorization

Guards added after the end-turn collapse diagnosis (full analysis + evidence in
[endturn_collapse.md](endturn_collapse.md)). The root problem they target: all of a game's
states share **one** outcome label, and each map's terrain planes are a memorizable
fingerprint — so an unguarded value head learns to *recognize games* instead of evaluating
positions, and an unguarded policy head learns the end_turn *marginal* instead of
conditioning on the state.

**Value head:**

- **Per-game value subsampling** (`value_samples_per_game`). A budget of
  min(per_game, ~1/4 of the game's samples) positions per game keeps its value label
  for training; `train_step` masks the MSE to those (`Sample.train_value`). Originally
  1/8 (within-game states were near-duplicates of one shared label — AlphaGo used
  1 position/game); relaxed after mixed targets gave every state its own target.
  Policy targets are unaffected. `<=0` disables.
- **D8 symmetry augmentation** (`value_symmetry`): each train step, the value head
  trains on a random rotation/flip of the eligible rows (`features.d8_transform` —
  board planes, tile indices, and row/col feature columns transformed consistently;
  targets are invariant). ~8x effective value data and it deletes the map-fingerprint
  memorization channel. Policy stays in the original orientation (its targets live in
  board coordinates); extending augmentation to policy targets needs mask/slot/visit
  permutation and is deferred.
- **AdamW weight decay** (`weight_decay`, default 1e-4) via `make_optimizer` — decays
  matrix params only; biases and norm scales are exempt.
- **Value-head input dropout** (`PolysharkNet.value_dropout`, p=0.2 on the 256-d core
  input). Active in `net.train()` only; kept outside the `value_head` Sequential so
  checkpoint state_dict keys are unchanged.
- **Held-out validation seeds** (`val_games`, default 8/gen; seeds from `val_seed_base`).
  Extra self-play games each gen that never enter the buffer, scored after training
  (`val_value_loss` column in metrics.csv). **Read the gap, not the train loss**: train
  falling with val flat/rising = memorization — the collapsed run would have shown this
  from gen ~2.

**Policy head:**

- **Forced end_turn states never become samples.** The arena plays them agent-free
  (`_forced_end_turn` in `arena.py`): no search, no sample — neither head trains on a
  state with nothing to decide. (Also a self-play speedup.) These were ~13% of type-stage
  samples and pure marginal-drift gradient toward end_turn.
- **Single-choice stages are skipped in the policy CE** (`_policy_loss_for_sample`): any
  stage whose legal mask has exactly one choice carries zero information (the mask already
  decides), so its cross-entropy term is dropped. Catches forced entity/tile stages inside
  otherwise-real decisions; the sample itself stays value-eligible.

Related but not regularization: **staged per-stage search budgets** (see
[mcts.md](mcts.md) "Staged commitment") make every fired stage's visit target carry a full
`n_sims` budget, so deep-stage policy targets are search-improved distributions rather
than prior + sampling noise.

## Gen-0 bootstrap

The first generation has no trained weights, so a random-init network gives noise for *both*
value and priors (weaker than the heuristic — see below). Options to seed the loop:

- **Heuristic-bootstrapped self-play:** run gen-0 self-play with the `HeuristicEvaluator`
  (real value-margin, uniform priors) instead of the random net, so the visit-count policy
  targets and the value labels are meaningful from the start. Train gen-1 on that, then switch
  to the network evaluator.
- **Caveat (info leak):** `HeuristicEvaluator`'s value calls `heuristic_score(opp)`, which
  reads the opponent's true city/unit/tech counts **ignoring fog** — so gen-0 *value* labels
  are computed with omniscient vision. The *policy* targets (visit distributions) are still
  fog-honest. Acceptable as a bootstrap; fog-gate the heuristic value if it distorts learning.

## Replay buffer

- **Holds** finished-game samples (`Sample`s), each already outcome-labelled by the arena at
  game end. A FIFO ring of the last `capacity` samples (drop oldest) so the trainer sees a
  moving window of recent generations, not just the newest — reduces overfitting to one gen.
- **Mix across generations:** sample minibatches uniformly from the whole buffer, spanning
  several recent generations (standard AZ practice).
- **Persistence:** in-memory for Phase A; for Phase B (and crash-safety) persist to disk. The
  format is an open decision (below).

## Open decisions (resolve when building `train.py`)

- **Encode-at-collect vs encode-at-train.** Store raw `GameState` (lean sample, re-encode
  each epoch) or pre-encode to tensors (fatter sample, cheaper training)? Constrains the
  buffer/serialization format.
- **Sample serialization.** `pickle` the `Sample` list (quick, fat, Python-only) vs encoded
  `.npz`/`.pt` (training-ready) vs reconstruct from `.replay` + sidecar visit targets (like
  the old `ai/tdl` pipeline). Currently samples are **not persisted at all** — `run_games.py`
  counts and discards them.
- **Value target.** Final game outcome (current) vs bootstrapped MCTS root value.
- **Hyperparameters.** buffer `capacity`, `GAMES_PER_GEN`, `TRAIN_STEPS_PER_GEN`,
  `MINIBATCH`, learning rate/schedule, `GATE_THRESHOLD` (~0.55), gating game count.
- **Success metric (early).** Until capitals actually fall, track "beats previous gen" and
  "beats raw heuristic," not raw win rate — the value signal is a heuristic proxy at first.
