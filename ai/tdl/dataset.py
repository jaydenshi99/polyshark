"""
ReplayDataset: loads replay files into RAM, re-simulates on-the-fly,
applies random 1-of-8 dihedral augmentation per sample.

Replay format:
    seed <seed> <sz>
    outcome <v_p0> <v_p1>
    <type> <src> <dst> <param> <path_bits> <path_steps>
    ...
"""

import os, sys, random
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_STATES_PER_GAME = 50  # cap per replay to reduce within-game label correlation

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '../../build/bindings'))

import polyshark
from encoder import encode


class _Replay:
    __slots__ = ('seed', 'sz', 'v_finals', 'actions')

    def __init__(self, seed, sz, v_finals, actions):
        self.seed     = seed
        self.sz       = sz
        self.v_finals = v_finals  # [v_p0, v_p1]
        self.actions  = actions   # list of (type, src, dst, param, path_bits, path_steps)


def load_replay(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    parts    = lines[0].split()
    seed, sz = int(parts[1]), int(parts[2])
    parts    = lines[1].split()
    v_finals = [float(parts[1]), float(parts[2])]
    actions  = []
    for line in lines[2:]:
        t, src, dst, param, pb, ps = line.split()
        actions.append((int(t), int(src), int(dst), int(param), int(pb), int(ps)))
    return _Replay(seed, sz, v_finals, actions)


def _apply_transform(sp, t):
    """Apply one of 8 dihedral transforms to sp [C_IN, H, W]. Returns contiguous array."""
    if t >= 4:
        sp = np.flip(sp, axis=2)
        t -= 4
    if t > 0:
        sp = np.rot90(sp, k=t, axes=(1, 2))
    return np.ascontiguousarray(sp)


class ReplayDataset(Dataset):
    """
    Each item is one (state, augmentation) pair drawn from the replay buffer.
    Re-simulates the game to the target state on every __getitem__ call.
    DataLoader workers run this in parallel so the main process stays free.
    """

    def __init__(self, replay_paths, augment=True):
        self.augment = augment
        self.replays = [load_replay(p) for p in replay_paths]
        # Randomly subsample up to MAX_STATES_PER_GAME per replay to reduce
        # within-game label correlation across batches.
        self._index = []
        for ri, r in enumerate(self.replays):
            n = len(r.actions)
            indices = random.sample(range(n), min(MAX_STATES_PER_GAME, n))
            for si in indices:
                self._index.append((ri, si))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ri, si = self._index[idx]
        r      = self.replays[ri]

        # Re-simulate game to state si (apply si actions from the start).
        state = polyshark.make_random_game(seed=r.seed, sz=r.sz)
        for atype, afrom, ato, aparam, pb, ps in r.actions[:si]:
            state = state.apply_action_raw(atype, afrom, ato, aparam, pb, ps)

        p      = state.current_player()
        target = r.v_finals[p]

        sp, gv = encode(state)

        if self.augment:
            sp = _apply_transform(sp, random.randint(0, 7))

        return (torch.from_numpy(sp),
                torch.from_numpy(gv),
                torch.tensor(target, dtype=torch.float32))
