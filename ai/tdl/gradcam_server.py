"""
Grad-CAM / activation plane coprocess for the visualiser.

Protocol (one line per request):
  stdin:  seed sz n [a_type a_from a_to a_param a_pb a_ps]*n channel \n
  stdout: v0 v1 ... v(sz*sz-1) \n   (row-major float32, space-sep, normalised 0-1)

channel == -1  →  Grad-CAM weighted sum (last conv layer)
channel 0..31  →  raw activation plane at that index (last conv layer, upsampled)
"""

import sys, os, argparse
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))

import polyshark
from encoder import encode
from model   import ValueNet

# Layer indices in model.spatial that are ReLU outputs (spatial resolution, n_channels):
#   [2]  → 32 ch, 11×11   (stem, full res)
#   [5]  → 64 ch,  9×9
#   [8]  → 64 ch,  7×7
#   [11] → 32 ch,  5×5
TARGET_LAYER = 2


def _normalise(t):
    mn, mx = t.min().item(), t.max().item()
    if mx > mn:
        return (t - mn) / (mx - mn)
    return t.new_zeros(t.shape)


def get_heatmap(model, sp_np, gv_np, channel, sz):
    sp = torch.from_numpy(sp_np).unsqueeze(0).float()
    gv = torch.from_numpy(gv_np).unsqueeze(0).float()

    # Always capture the target layer's activations.
    saved = [None]
    handle = model.spatial[TARGET_LAYER].register_forward_hook(
        lambda m, i, o: saved.__setitem__(0, o))

    if channel == -1:
        # Grad-CAM: need gradients.
        with torch.enable_grad():
            model(sp, gv)
        handle.remove()

        A = saved[0].clone().detach().requires_grad_(True)
        with torch.enable_grad():
            x = A
            for i in range(TARGET_LAYER + 1, len(model.spatial)):
                x = model.spatial[i](x)
            out = model.head(torch.cat([x, gv], dim=1))
            out.sum().backward()

        dA    = A.grad                                       # [1, 32, 5, 5]
        alpha = dA.squeeze(0).mean(dim=(1, 2))               # [32]
        cam   = (alpha[:, None, None] * A.squeeze(0).detach()).sum(dim=0)
        cam   = torch.relu(cam)
    else:
        # Raw activation plane — no backward pass needed.
        with torch.no_grad():
            model(sp, gv)
        handle.remove()

        A   = saved[0].squeeze(0)                           # [32, 5, 5]
        cam = torch.relu(A[channel])                        # [5, 5]

    cam = _normalise(cam)
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(sz, sz),
        mode='bilinear', align_corners=False
    ).squeeze()

    return cam.detach().numpy().astype('float32').flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    args = parser.parse_args()

    model = ValueNet()
    ckpt  = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    sys.stderr.write(f'gradcam_server: loaded {args.ckpt}\n')
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        tokens  = line.split()
        seed    = int(tokens[0])
        sz      = int(tokens[1])
        n       = int(tokens[2])
        channel = int(tokens[3 + n * 6])   # last field

        state = polyshark.make_random_game(seed=seed, sz=sz)
        for i in range(n):
            base = 3 + i * 6
            t, fr, to, pa, pb, ps = (int(tokens[base + j]) for j in range(6))
            state = state.apply_action_raw(t, fr, to, pa, pb, ps)

        sp, gv = encode(state)
        hmap   = get_heatmap(model, sp, gv, channel, sz)

        sys.stdout.write(' '.join(f'{v:.6f}' for v in hmap) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
