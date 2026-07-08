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
GAMES_PER_GEN       = 8      # self-play games appended to the buffer each generation
TRAIN_STEPS_PER_GEN = 100    # minibatch optimizer steps each generation
MINIBATCH           = 32     # samples per optimizer step
BUFFER_CAPACITY     = 20000  # FIFO replay window (oldest samples drop off)

# --- self-play search (per decision) ---
N_SIMS      = 100            # MCTS simulations per move (strength vs speed)
C_PUCT      = 1.5           # PUCT exploration constant
ADD_NOISE   = True          # Dirichlet root noise (ON for self-play diversity)
TEMPERATURE = 1.0           # opening move sampling: 1.0 ~ visits, 0.0 greedy
TEMP_TURNS  = 6             # turns temperature applies before dropping to greedy
TURN_LIMIT  = 30            # per-game turn cap (turn-capped games use heuristic outcome)

# --- gen-0 bootstrap ---
BOOTSTRAP_GEN0 = True       # gen 0 self-plays with the heuristic (meaningful data) instead
                            # of the random-init net; gens >=1 use the net being trained

# --- optimization ---
LR = 1e-3                   # Adam learning rate (net + policy jointly)

# --- io ---
BASE_SEED = 0
CKPT_ROOT = os.path.join(_PKG, "data", "checkpoints")  # parent; each run gets its own subfolder
RUN_LABEL = ""             # optional suffix on the run folder, e.g. "highsim" -> run_<ts>_highsim

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
        add_noise=ADD_NOISE, lr=LR, base_seed=BASE_SEED, bootstrap_gen0=BOOTSTRAP_GEN0,
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
