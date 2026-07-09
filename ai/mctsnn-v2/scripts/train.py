"""
Phase A training runner — configure the block below and run (from repo root, venv active):

    source .venv/bin/activate
    python ai/mctsnn-v2/scripts/train.py

No CLI flags — everything is in CONFIG. This is the alternating self-play/train loop
(see docs/training.md): each generation plays GAMES_PER_GEN self-play games with the current
network, appends them to a replay buffer, runs TRAIN_STEPS_PER_GEN minibatch updates, and
writes a checkpoint. Config front-end only; the loop lives in src/trainer.py.
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
N_GENS              = 50      # generations (self-play -> train -> checkpoint cycles)
GAMES_PER_GEN       = 30      # self-play games appended to the buffer each generation
TRAIN_STEPS_PER_GEN = 200    # minibatch optimizer steps each generation (~1 epoch over a
                             # gen's ~6k new samples at MINIBATCH=32)
MINIBATCH           = 32     # samples per optimizer step
BUFFER_CAPACITY     = 20000  # FIFO replay window (oldest samples drop off)

# --- parallelism ---
NUM_WORKERS = 6             # self-play worker processes (1 = in-process, no pool). Set near
                            # your physical core count; each worker plays whole games at once.

# --- self-play search (per decision) ---
N_SIMS      = 100           # MCTS simulations per move (strength vs speed)
C_PUCT      = 1.5           # PUCT exploration constant
ADD_NOISE   = True          # Dirichlet root noise (ON for self-play diversity)
TEMPERATURE = 1.0           # opening move sampling: 1.0 ~ visits, 0.0 greedy
TEMP_TURNS  = 6             # turns temperature applies before dropping to greedy
TURN_LIMIT  = 30            # per-game turn cap (turn-capped games use heuristic outcome)

# --- gen-0 bootstrap ---
BOOTSTRAP_GEN0 = True       # gen 0 self-plays with the heuristic (meaningful data) instead
                            # of the random-init net; gens >=1 use the net being trained

# --- turn-cap labels & gen-0 search value (see docs/endturn_collapse.md #1) ---
TURN_CAP_WINNER  = True     # True: turn-capped games label ±1 by heuristic-margin sign
                            # (winner declared at the cap; value head estimates win prob;
                            # mutual passing is never label-neutral). False: legacy tanh.
WINNER_DEAD_ZONE = 0.75     # heuristic points inside which a capped game labels 0 (a tie).
                            # A village is ~4 points; 1.0 = tech/unit dust doesn't decide.
GEN0_SEARCH_SCALE = 1.0     # tanh scale of the gen-0 bootstrap SEARCH evaluator. Sharp on
                            # purpose (decisive bootstrap play); independent of the labels.
HEURISTIC_SCALE = 0.30      # legacy tanh(scale * margin) labels, used only when
                            # TURN_CAP_WINNER = False.

# --- optimization ---
LR = 1e-3                   # AdamW learning rate (net + policy jointly)
WEIGHT_DECAY = 1e-4         # AdamW decay on matrix params (biases/norm scales exempt)

# --- value-overfit guards (see docs/endturn_collapse.md) ---
VALUE_SAMPLES_PER_GAME = 16 # positions per game that keep their value label for training
                            # (split across players; the rest train policy only). <=0 = all.
                            # A game's states all share one outcome label — training value
                            # on every state is how the head memorizes games.
VAL_GAMES = 6               # per gen: extra self-play games on held-out seeds, kept out of
                            # the buffer, scored after training (val_value_loss in
                            # metrics.csv). Train/val gap = memorization meter. 0 = off.

# --- io ---
BASE_SEED = 0
CKPT_ROOT = os.path.join(_PKG, "data", "checkpoints")  # parent; each run gets its own subfolder
RUN_LABEL = "guards"       # optional suffix on the run folder, e.g. "highsim" -> run_<ts>_highsim

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
        minibatch=MINIBATCH, buffer_capacity=BUFFER_CAPACITY, turn_limit=TURN_LIMIT,
        n_sims=N_SIMS, c_puct=C_PUCT, temperature=TEMPERATURE, temp_turns=TEMP_TURNS,
        add_noise=ADD_NOISE, lr=LR, weight_decay=WEIGHT_DECAY,
        value_samples_per_game=VALUE_SAMPLES_PER_GAME, val_games=VAL_GAMES,
        turn_cap_winner=TURN_CAP_WINNER, winner_dead_zone=WINNER_DEAD_ZONE,
        gen0_search_scale=GEN0_SEARCH_SCALE,
        base_seed=BASE_SEED, bootstrap_gen0=BOOTSTRAP_GEN0,
        heuristic_scale=HEURISTIC_SCALE, num_workers=NUM_WORKERS,
    )
    # Save the run's config alongside its checkpoints so each run is self-describing.
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Phase A training | run dir: {run_dir}")
    print(f"  {N_GENS} gens x {GAMES_PER_GEN} games x {TRAIN_STEPS_PER_GEN} steps | "
          f"n_sims={N_SIMS} turn_limit={TURN_LIMIT}\n")
    run_training(ckpt_dir=run_dir, **cfg)
    print(f"\ndone. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
