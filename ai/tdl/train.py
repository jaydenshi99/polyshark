"""
TDL training loop.

Each generation:
  1. Run self-play  →  (GameState, float) pairs
  2. Encode pairs   →  (spatial, gvec, target) tensors
  3. Append to rolling buffer (capped at BUFFER_MAX pairs)
  4. Train with MSE loss
  5. Save checkpoint  →  checkpoints/gen_NNN.pt
  6. Promote NN as eval_fn for next generation

Usage:
  python train.py                         # 10 gens, 20 games, 200 sims
  python train.py --gens 20 --games 40 --sims 400
  python train.py --resume checkpoints/gen_005.pt
"""

import sys, os, argparse, csv, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))

from selfplay  import run_selfplay
from encoder   import C_IN, G
from model     import ValueNet

CHECKPOINTS_DIR = os.path.join(_HERE, 'checkpoints')
RUNS_DIR        = os.path.join(_HERE, 'runs')
LOG_PATH        = os.path.join(CHECKPOINTS_DIR, 'training_log.csv')

BUFFER_MAX = 200_000
N_GENS     = 10
N_GAMES    = 20
N_SIMS     = 200
N_WORKERS  = 1
EPOCHS     = 10
BATCH_SIZE = 256
LR         = 1e-3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_gen(model, optimizer, sp_buf, gv_buf, tgt_buf, epochs, batch_size, device,
                  writer=None, gen=0):
    dataset = TensorDataset(
        torch.from_numpy(sp_buf),
        torch.from_numpy(gv_buf),
        torch.from_numpy(tgt_buf),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for sp, gv, tgt in loader:
            sp, gv, tgt = sp.to(device), gv.to(device), tgt.to(device)
            loss = F.mse_loss(model(sp, gv).squeeze(-1), tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(tgt)
        avg = epoch_loss / len(dataset)
        print(f'  epoch {epoch+1:2d}/{epochs}  loss={avg:.4f}')
        if writer:
            writer.add_scalar('loss/epoch', avg, gen * epochs + epoch)
        final_loss = avg
    model.eval()
    return final_loss


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
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Rolling buffer stored as numpy arrays so re-encoding is not needed each epoch.
    sp_buf  = np.empty((0, C_IN, 11, 11), dtype=np.float32)
    gv_buf  = np.empty((0, G),            dtype=np.float32)
    tgt_buf = np.empty((0,),              dtype=np.float32)

    start_gen = 0
    prev_ckpt = None   # None = use heuristic; set to path after first checkpoint is saved

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=RUNS_DIR)

    # Optionally resume from a checkpoint.
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
        log_writer.writerow(['gen', 'n_pairs', 'buffer_size', 'train_loss', 'elapsed_s'])

    for gen in range(start_gen, start_gen + n_gens):
        print(f'\n=== Generation {gen} ===')
        t_gen = time.time()

        # 1. Self-play + encode (encoding runs inside workers when parallel)
        t0 = time.time()
        new_sp, new_gv, new_tgt = run_selfplay(n_games=n_games, n_sims=n_sims, gen=gen,
                                               n_workers=n_workers, ckpt_path=prev_ckpt)
        print(f'self-play: {len(new_tgt)} pairs  ({time.time()-t0:.1f}s)')

        # 2. Append to rolling buffer; drop oldest pairs when over cap.
        sp_buf  = np.concatenate([sp_buf,  new_sp],  axis=0)
        gv_buf  = np.concatenate([gv_buf,  new_gv],  axis=0)
        tgt_buf = np.concatenate([tgt_buf, new_tgt], axis=0)
        if len(tgt_buf) > BUFFER_MAX:
            sp_buf  = sp_buf[-BUFFER_MAX:]
            gv_buf  = gv_buf[-BUFFER_MAX:]
            tgt_buf = tgt_buf[-BUFFER_MAX:]
        print(f'buffer:    {len(tgt_buf)} pairs')
        if writer:
            writer.add_scalar('data/pairs_this_gen', len(new_tgt),  gen)
            writer.add_scalar('data/buffer_size',    len(tgt_buf),  gen)

        # 4. Train
        loss = train_one_gen(model, optimizer, sp_buf, gv_buf, tgt_buf,
                             epochs, batch_size, device, writer=writer, gen=gen)

        # 5. Save checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f'gen_{gen:03d}.pt')
        torch.save({'gen': gen, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, ckpt_path)

        elapsed = time.time() - t_gen
        print(f'saved {os.path.basename(ckpt_path)}  ({elapsed:.1f}s total)')
        if writer:
            writer.add_scalar('loss/gen',       loss,    gen)
            writer.add_scalar('perf/elapsed_s', elapsed, gen)

        log_writer.writerow([gen, len(new_tgt), len(tgt_buf), f'{loss:.4f}', f'{elapsed:.1f}'])
        log_file.flush()

        # 6. Promote checkpoint for next generation's self-play.
        prev_ckpt = ckpt_path

    writer.close()
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gens',    type=int,  default=N_GENS)
    parser.add_argument('--games',   type=int,  default=N_GAMES)
    parser.add_argument('--sims',    type=int,  default=N_SIMS)
    parser.add_argument('--workers', type=int,  default=N_WORKERS)
    parser.add_argument('--epochs',  type=int,  default=EPOCHS)
    parser.add_argument('--resume',  type=str,  default=None)
    args = parser.parse_args()
    run_training(n_gens=args.gens, n_games=args.games, n_sims=args.sims,
                 n_workers=args.workers, epochs=args.epochs, resume=args.resume)
