"""
TDL training loop.

Each generation:
  1. Run self-play  →  encoded (spatial, gvec, target) arrays
  2. Append to rolling buffer (capped at BUFFER_MAX pairs)
  3. Split buffer into 90% train / 10% val
  4. Train with MSE loss + weight decay
  5. Compute val loss
  6. Save checkpoint  →  checkpoints/gen_NNN.pt
  7. Log to training_log.csv

Usage:
  python train.py                         # 10 gens, 20 games, 200 sims
  python train.py --gens 20 --games 24 --sims 50 --workers 8
  python train.py --resume checkpoints/gen_005.pt
"""

import sys, os, argparse, csv, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))

from selfplay  import run_selfplay
from encoder   import C_IN, G
from model     import ValueNet

CHECKPOINTS_DIR = os.path.join(_HERE, 'checkpoints')
LOG_PATH        = os.path.join(CHECKPOINTS_DIR, 'training_log.csv')

BUFFER_MAX   = 50_000
VAL_FRACTION = 0.1
N_GENS       = 10
N_GAMES      = 20
N_SIMS       = 100
N_WORKERS    = 1
EPOCHS       = 2
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_loader(sp, gv, tgt, batch_size, shuffle, device):
    return DataLoader(
        TensorDataset(torch.from_numpy(sp), torch.from_numpy(gv), torch.from_numpy(tgt)),
        batch_size=batch_size, shuffle=shuffle,
        pin_memory=(device.type == 'cuda'),
    )


def train_one_gen(model, optimizer, sp_buf, gv_buf, tgt_buf, epochs, batch_size, device):
    n      = len(tgt_buf)
    n_val  = max(1, int(n * VAL_FRACTION))
    n_train = n - n_val

    # Fixed random split — same indices each call within a gen.
    idx     = np.random.permutation(n)
    tr_idx  = idx[:n_train]
    va_idx  = idx[n_train:]

    train_loader = _make_loader(sp_buf[tr_idx], gv_buf[tr_idx], tgt_buf[tr_idx],
                                batch_size, shuffle=True,  device=device)
    val_loader   = _make_loader(sp_buf[va_idx], gv_buf[va_idx], tgt_buf[va_idx],
                                batch_size, shuffle=False, device=device)

    final_train = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for sp, gv, tgt in train_loader:
            sp, gv, tgt = sp.to(device), gv.to(device), tgt.to(device)
            loss = F.mse_loss(model(sp, gv).squeeze(-1), tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(tgt)
        final_train = epoch_loss / n_train
        print(f'  epoch {epoch+1:2d}/{epochs}  train_loss={final_train:.4f}')

    # Validation loss (no grad, no dropout)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sp, gv, tgt in val_loader:
            sp, gv, tgt = sp.to(device), gv.to(device), tgt.to(device)
            val_loss += F.mse_loss(model(sp, gv).squeeze(-1), tgt).item() * len(tgt)
    val_loss /= n_val
    print(f'  val_loss={val_loss:.4f}')

    return final_train, val_loss


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training(n_gens=N_GENS, n_games=N_GAMES, n_sims=N_SIMS,
                 n_workers=N_WORKERS, epochs=EPOCHS, batch_size=BATCH_SIZE, resume=None):

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}  workers: {n_workers}')

    model     = ValueNet().to(device)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    sp_buf  = np.empty((0, C_IN, 11, 11), dtype=np.float32)
    gv_buf  = np.empty((0, G),            dtype=np.float32)
    tgt_buf = np.empty((0,),              dtype=np.float32)

    start_gen = 0
    prev_ckpt = None

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
        log_writer.writerow(['gen', 'n_pairs', 'buffer_size',
                             'train_loss', 'val_loss', 'elapsed_s'])

    for gen in range(start_gen, start_gen + n_gens):
        print(f'\n=== Generation {gen} ===')
        t_gen = time.time()

        # 1. Self-play + encode
        t0 = time.time()
        new_sp, new_gv, new_tgt = run_selfplay(n_games=n_games, n_sims=n_sims, gen=gen,
                                               n_workers=n_workers, ckpt_path=prev_ckpt)
        print(f'self-play: {len(new_tgt)} pairs  ({time.time()-t0:.1f}s)')

        # 2. Append to rolling buffer; drop oldest when over cap.
        sp_buf  = np.concatenate([sp_buf,  new_sp],  axis=0)
        gv_buf  = np.concatenate([gv_buf,  new_gv],  axis=0)
        tgt_buf = np.concatenate([tgt_buf, new_tgt], axis=0)
        if len(tgt_buf) > BUFFER_MAX:
            sp_buf  = sp_buf[-BUFFER_MAX:]
            gv_buf  = gv_buf[-BUFFER_MAX:]
            tgt_buf = tgt_buf[-BUFFER_MAX:]
        print(f'buffer:    {len(tgt_buf)} pairs')

        # 3. Train + validate
        train_loss, val_loss = train_one_gen(model, optimizer, sp_buf, gv_buf, tgt_buf,
                                             epochs, batch_size, device)

        # 4. Save checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f'gen_{gen:03d}.pt')
        torch.save({'gen': gen, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, ckpt_path)

        elapsed = time.time() - t_gen
        print(f'saved {os.path.basename(ckpt_path)}  ({elapsed:.1f}s total)')

        log_writer.writerow([gen, len(new_tgt), len(tgt_buf),
                             f'{train_loss:.4f}', f'{val_loss:.4f}', f'{elapsed:.1f}'])
        log_file.flush()

        prev_ckpt = ckpt_path

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gens',    type=int, default=N_GENS)
    parser.add_argument('--games',   type=int, default=N_GAMES)
    parser.add_argument('--sims',    type=int, default=N_SIMS)
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    parser.add_argument('--epochs',  type=int, default=EPOCHS)
    parser.add_argument('--resume',  type=str, default=None)
    args = parser.parse_args()
    run_training(n_gens=args.gens, n_games=args.games, n_sims=args.sims,
                 n_workers=args.workers, epochs=args.epochs, resume=args.resume)
