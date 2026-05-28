"""
Grad-CAM heatmap generator for a replay + checkpoint.

Writes a binary .heatmap sidecar alongside the replay file.
Load in the visualiser with --replay <file>; press H to toggle the overlay.

Format: int32 n_steps, int32 sz, then n_steps*sz*sz float32 (row-major).

Usage:
  python heatmap.py checkpoints/gen_020.pt replays/gen_020/game_00_seed_12345.replay
  python heatmap.py checkpoints/gen_020.pt replays/gen_020/game_00_seed_12345.replay --out /tmp/out.heatmap
"""

import sys, os, struct, argparse
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))

import polyshark
from encoder import encode
from model   import ValueNet


def grad_cam(model, sp_np, gv_np):
    """
    Grad-CAM w.r.t. the last conv ReLU (model.spatial[11], output [1,32,5,5]).
    Returns an [SZ, SZ] float32 array normalised to [0, 1].
    """
    # Capture last-conv activations via a forward hook.
    saved = [None]
    handle = model.spatial[11].register_forward_hook(lambda m, i, o: saved.__setitem__(0, o))
    with torch.enable_grad():
        sp = torch.from_numpy(sp_np).unsqueeze(0).float()
        gv = torch.from_numpy(gv_np).unsqueeze(0).float()
        model(sp, gv)   # fills saved[0]
    handle.remove()

    # Clone into a fresh leaf so we can differentiate w.r.t. A directly,
    # bypassing any in-place ops in earlier layers.
    A = saved[0].clone().detach().requires_grad_(True)   # [1, 32, 5, 5]

    with torch.enable_grad():
        x_flat = model.spatial[12](A)                    # Flatten → [1, 800]
        x      = torch.cat([x_flat, gv], dim=1)
        out    = model.head(x)                            # [1, 1]
        out.sum().backward()

    dA = A.grad   # [1, 32, 5, 5]
    if dA is None:
        sz = sp_np.shape[1]
        return np.zeros((sz, sz), dtype=np.float32)

    alpha = dA.squeeze(0).mean(dim=(1, 2))               # [32]  channel weights
    cam   = (alpha[:, None, None] * A.squeeze(0).detach()).sum(dim=0)  # [5, 5]
    cam   = torch.relu(cam)

    sz  = sp_np.shape[1]
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(sz, sz),
        mode='bilinear', align_corners=False
    ).squeeze()

    mn, mx = cam.min().item(), cam.max().item()
    if mx > mn:
        cam = (cam - mn) / (mx - mn)
    else:
        cam = torch.zeros(sz, sz)

    return cam.detach().numpy().astype(np.float32)


def parse_replay(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    parts    = lines[0].split()
    seed, sz = int(parts[1]), int(parts[2])
    actions  = []
    for line in lines[2:]:
        tokens = line.split()
        if tokens[0] == 'outcome':
            continue
        t, src, dst, param = int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])
        pb = int(tokens[4]) if len(tokens) > 4 else 0
        ps = int(tokens[5]) if len(tokens) > 5 else 0
        actions.append((t, src, dst, param, pb, ps))
    return seed, sz, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt')
    parser.add_argument('replay')
    parser.add_argument('--out', type=str, default=None,
                        help='output path (default: same name as replay with .heatmap)')
    args = parser.parse_args()

    device = torch.device('cpu')
    model  = ValueNet().to(device)
    ckpt   = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()

    seed, sz, actions = parse_replay(args.replay)
    print(f'replay: {len(actions)} actions  map_size={sz}  seed={seed}')

    state    = polyshark.make_random_game(seed=seed, sz=sz)
    heatmaps = []

    sp, gv = encode(state)
    heatmaps.append(grad_cam(model, sp, gv))

    for step, (t, src, dst, param, pb, ps) in enumerate(actions):
        state = state.apply_action_raw(t, src, dst, param, pb, ps)
        sp, gv = encode(state)
        heatmaps.append(grad_cam(model, sp, gv))
        if (step + 1) % 20 == 0:
            print(f'  {step + 1}/{len(actions)} states processed')

    out_path = args.out or (os.path.splitext(args.replay)[0] + '.heatmap')
    n = len(heatmaps)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<ii', n, sz))
        for h in heatmaps:
            f.write(h.tobytes())
    print(f'wrote {n} frames → {out_path}')


if __name__ == '__main__':
    main()
