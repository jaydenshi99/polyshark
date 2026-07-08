"""
Replay buffer — the conveyor belt between self-play and the trainer (see docs/training.md).

A FIFO ring of the last `capacity` samples: self-play appends finished games, the trainer
draws random minibatches. Old samples fall off the back so the trainer sees a moving window
of recent generations (reduces overfitting to the newest gen). Phase A: in-memory only.
"""

import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def extend(self, samples):
        """Add a game's worth of (outcome-labelled) Samples."""
        self.buf.extend(samples)

    def sample(self, n):
        """Draw `n` samples uniformly at random, with replacement (standard for replay)."""
        if not self.buf:
            return []
        return random.choices(self.buf, k=n)

    def __len__(self):
        return len(self.buf)
