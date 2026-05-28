"""
TDL training loop.

Each generation:
  1. Run self-play  →  replay files saved to disk
  2. Add train replays to rolling buffer (capped at TRAIN_BUF_REPLAYS)
  3. Add val replays to val buffer (capped at VAL_BUF_REPLAYS, never trained on)
  4. Build ReplayDataset → DataLoader (re-simulate + encode + augment on-the-fly)
  5. Train for EPOCHS epochs
  6. Save checkpoint  →  checkpoints/gen_NNN.pt
  7. Log to training_log.csv

Usage:
  python train.py                         # 10 gens, 40 games, 100 sims
  python train.py --gens 30 --games 40 --sims 100 --workers 8
  python train.py --resume checkpoints/gen_005.pt
"""

import sys, os, argparse, csv, time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))

from selfplay import run_selfplay
from dataset  import ReplayDataset
from model    import ValueNet

CHECKPOINTS_DIR    = os.path.join(_HERE, 'checkpoints')
LOG_PATH           = os.path.join(CHECKPOINTS_DIR, 'training_log.csv')

TRAIN_BUF_REPLAYS  = 400   # keep last ~10 gens of train games
VAL_BUF_REPLAYS    = 80    # keep last ~20 gens of val games
N_GENS             = 10
N_GAMES            = 80
N_SIMS             = 100
N_WORKERS          = 1
EPOCHS             = 2
BATCH_SIZE         = 256
LR                 = 1e-3
WEIGHT_DECAY       = 5e-4
LOADER_WORKERS     = 4     # DataLoader workers for re-simulation + encoding


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_gen(model, optimizer, train_loader, val_loader, epochs, device):
    final_train = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, n_train = 0.0, 0
        for sp, gv, tgt in train_loader:
            sp, gv, tgt = sp.to(device), gv.to(device), tgt.to(device)
            loss = F.mse_loss(model(sp, gv).squeeze(-1), tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(tgt)
            n_train    += len(tgt)
        final_train = total_loss / max(n_train, 1)
        print(f'  epoch {epoch+1:2d}/{epochs}  train_loss={final_train:.4f}')

    model.eval()
    val_loss, n_val = 0.0, 0
    with torch.no_grad():
        for sp, gv, tgt in val_loader:
            sp, gv, tgt = sp.to(device), gv.to(device), tgt.to(device)
            val_loss += F.mse_loss(model(sp, gv).squeeze(-1), tgt).item() * len(tgt)
            n_val    += len(tgt)
    val_loss /= max(n_val, 1)
    print(f'  val_loss={val_loss:.4f}')

    return final_train, val_loss


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training(n_gens=N_GENS, n_games=N_GAMES, n_sims=N_SIMS,
                 n_workers=N_WORKERS, epochs=EPOCHS, resume=None):

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}  workers: {n_workers}')

    model     = ValueNet().to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_paths = []
    val_paths   = []
    start_gen   = 0
    prev_ckpt   = None

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_gen = ckpt['gen'] + 1
        prev_ckpt = resume
        print(f'resumed from {resume}  (starting at gen {start_gen})')

    log_exists = os.path.exists(LOG_PATH)
    log_file   = open(LOG_PATH, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['gen', 'n_train_games', 'n_val_games',
                             'train_states', 'train_loss', 'val_loss', 'elapsed_s'])

    for gen in range(start_gen, start_gen + n_gens):
        print(f'\n=== Generation {gen} ===')
        t_gen = time.time()

        # 1. Self-play
        new_train, new_val = run_selfplay(
            n_games=n_games, n_sims=n_sims, gen=gen,
            n_workers=n_workers, ckpt_path=prev_ckpt)

        # 2. Update replay buffers (rotate out oldest when over cap).
        train_paths.extend(new_train)
        val_paths.extend(new_val)
        if len(train_paths) > TRAIN_BUF_REPLAYS:
            train_paths = train_paths[-TRAIN_BUF_REPLAYS:]
        if len(val_paths) > VAL_BUF_REPLAYS:
            val_paths = val_paths[-VAL_BUF_REPLAYS:]
        print(f'buffer:  {len(train_paths)} train replays  {len(val_paths)} val replays')

        # 3. Build datasets and loaders.
        train_ds = ReplayDataset(train_paths, augment=True)
        val_ds   = ReplayDataset(val_paths,   augment=False)
        print(f'         {len(train_ds)} train states  {len(val_ds)} val states')

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=LOADER_WORKERS, persistent_workers=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=LOADER_WORKERS, persistent_workers=True)

        # 4. Train
        train_loss, val_loss = train_one_gen(
            model, optimizer, train_loader, val_loader, epochs, device)

        # 5. Save checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f'gen_{gen:03d}.pt')
        torch.save({'gen': gen, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, ckpt_path)

        elapsed = time.time() - t_gen
        print(f'saved {os.path.basename(ckpt_path)}  ({elapsed:.1f}s total)')

        log_writer.writerow([gen, len(train_paths), len(val_paths),
                             len(train_ds), f'{train_loss:.4f}', f'{val_loss:.4f}',
                             f'{elapsed:.1f}'])
        log_file.flush()
        prev_ckpt = ckpt_path

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gens',    type=int, default=N_GENS)
    parser.add_argument('--games',   type=int, default=N_GAMES)  # 80
    parser.add_argument('--sims',    type=int, default=N_SIMS)
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    parser.add_argument('--epochs',  type=int, default=EPOCHS)
    parser.add_argument('--resume',  type=str, default=None)
    args = parser.parse_args()
    run_training(n_gens=args.gens, n_games=args.games, n_sims=args.sims,
                 n_workers=args.workers, epochs=args.epochs, resume=args.resume)
