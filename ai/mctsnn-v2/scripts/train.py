"""
Phase A training runner — edit CONFIG, then run (from repo root, venv active):

    source .venv/bin/activate
    python ai/mctsnn-v2/scripts/train.py

Config front-end only; the loop lives in src/trainer.py (see docs/training.md).
Knobs not listed here (add_noise, temperature, gen0 bootstrap/scale, anneal windows,
legacy tanh labels, ...) use run_training's defaults.
"""

import datetime
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))               # ai/mctsnn-v2/scripts
_PKG  = os.path.dirname(_HERE)                                    # ai/mctsnn-v2
_ROOT = os.path.dirname(os.path.dirname(_PKG))                    # repo root
sys.path.insert(0, os.path.join(_PKG, "src"))
sys.path.insert(0, os.path.join(_ROOT, "build", "bindings"))

from trainer import run_training  # noqa: E402


# ============================================================================
# CONFIG — edit this
# ============================================================================

# --- loop sizing ---
N_GENS              = 100    # self-play -> train -> gate -> checkpoint cycles
GAMES_PER_GEN       = 32     # self-play games appended to the buffer each gen
TRAIN_STEPS_PER_GEN = 200    # optimizer steps/gen (auto-throttled on small buffers)
MINIBATCH           = 32
BUFFER_CAPACITY     = 20000  # FIFO sample window (~8-10 gens at cap 10)
NUM_WORKERS         = 4      # self-play processes; set near physical core count

# --- search (per decision) ---
N_SIMS    = 100              # sims per subaction stage (staged commitment)
C_PUCT    = 1.5
TEMP_FRAC = 0.20             # temperature on the first 20% of each game's cap, then greedy

# --- horizon curriculum ---
TURN_LIMIT     = 10          # max turn cap; also the curriculum ceiling
TURN_CAP_START = 5           # gen-0 cap; None = constant TURN_LIMIT
TURN_CAP_GROW  = 0.10        # cap growth per gen (0.10 -> +1 every 10 gens)

# --- turn-cap winner labels ---
WINNER_DEAD_ZONE = 0.25      # margin (heuristic pts) below which a capped game is a tie;
                             # weights in bindings.cpp heuristic_score set the tiers
WINNER_TIE_VALUE = -0.15     # tie contempt: ties label negative for BOTH players

# --- gating (candidate must beat incumbent to generate data) ---
GATING         = True
GATE_GAMES     = 16          # paired: each seed played twice, seats swapped (keep even);
                             # map luck + seat advantage cancel per pair. Doubles as val set
GATE_THRESHOLD = 0.55        # promote at >= this score (win=1, tie=0.5)

# --- anti-collapse / anti-overfit (docs/endturn_collapse.md) ---
KL_ANCHOR_GENS         = 5     # gens 1..N: hold type head near the post-gen0 policy
SEARCH_VALUE_WEIGHT    = 0.3   # mixed value targets: (1-w)*outcome + w*search root value
VALUE_SYMMETRY         = True  # value head trains on random D8 board transforms
VALUE_SAMPLES_PER_GAME = 32    # value-eligible positions per game (<=0 = all)
VAL_GAMES              = 6     # held-out val games — used only at gen 0 / when GATING off

# --- optimization ---
LR           = 1e-3
LR_FINAL     = 3e-4          # cosine decay target by the last gen (None = constant)
WEIGHT_DECAY = 1e-4          # AdamW, matrix params only

# --- io ---
BASE_SEED = 0
CKPT_ROOT = os.path.join(_PKG, "data", "checkpoints")
RUN_LABEL = "gated_cap10"    # suffix on the run folder — name each run distinctly

# ============================================================================
# end CONFIG
# ============================================================================


def _new_run_dir():
    """A fresh timestamped folder under CKPT_ROOT so runs never overwrite each other."""
    stamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    name = f"{stamp}_{RUN_LABEL}" if RUN_LABEL else stamp
    run_dir = os.path.join(CKPT_ROOT, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    run_dir = _new_run_dir()
    cfg = dict(
        n_gens=N_GENS, games_per_gen=GAMES_PER_GEN, train_steps_per_gen=TRAIN_STEPS_PER_GEN,
        minibatch=MINIBATCH, buffer_capacity=BUFFER_CAPACITY, num_workers=NUM_WORKERS,
        n_sims=N_SIMS, c_puct=C_PUCT, temp_frac=TEMP_FRAC,
        turn_limit=TURN_LIMIT, turn_cap_start=TURN_CAP_START, turn_cap_grow=TURN_CAP_GROW,
        winner_dead_zone=WINNER_DEAD_ZONE, winner_tie_value=WINNER_TIE_VALUE,
        gating=GATING, gate_games=GATE_GAMES, gate_threshold=GATE_THRESHOLD,
        kl_anchor_gens=KL_ANCHOR_GENS, search_value_weight=SEARCH_VALUE_WEIGHT,
        value_symmetry=VALUE_SYMMETRY, value_samples_per_game=VALUE_SAMPLES_PER_GAME,
        val_games=VAL_GAMES,
        lr=LR, lr_final=LR_FINAL, weight_decay=WEIGHT_DECAY,
        base_seed=BASE_SEED,
    )
    # Save the run's config alongside its checkpoints so each run is self-describing
    # (explicitly-set knobs only; everything else is run_training's defaults).
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Phase A training | run dir: {run_dir}")
    print(f"  {N_GENS} gens x {GAMES_PER_GEN} games x {TRAIN_STEPS_PER_GEN} steps | "
          f"n_sims={N_SIMS} turn_limit={TURN_LIMIT}\n")
    run_training(ckpt_dir=run_dir, **cfg)
    print(f"\ndone. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
