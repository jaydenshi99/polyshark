"""
Shape / masking / gradient checks for the policy head, driven with a synthetic
TrunkCache and synthetic legal-masks. Run:

    source .venv/bin/activate
    python ai/mctsnn-v2/test_policy.py

The real masks come from the (later) legal-action helper; here we just prove each stage
produces correctly-shaped, correctly-masked, differentiable logits.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import TrunkCache  # noqa: E402
from policy import (  # noqa: E402
    PolicyHead, N_TYPES, K_TRAIN_UNITS, K_TECH, BRACKET_DIM, T_MOVE, T_TRAIN,
)

B, Lu, Lc, MAP = 4, 6, 3, 121


def _mask(*shape):
    """Random boolean legal-mask with column 0 forced True (no all-illegal row)."""
    m = torch.rand(*shape) > 0.4
    m[..., 0] = True
    return m


def _fake_cache():
    return TrunkCache(
        core=torch.randn(B, 256),
        feature_map=torch.randn(B, 64, 11, 11),
        unit_tok=torch.randn(B, Lu, 128),
        city_tok=torch.randn(B, Lc, 128),
        unit_mask=torch.ones(B, Lu, dtype=torch.bool),
        city_mask=torch.ones(B, Lc, dtype=torch.bool),
    )


def _check(name, logits, mask):
    """Shape implied by mask; masked entries get 0 probability; row sums to 1."""
    assert logits.shape == mask.shape, (name, logits.shape, mask.shape)
    p = F.softmax(logits, dim=-1)
    assert torch.isfinite(p).all(), f"{name}: non-finite probs"
    assert (p[~mask] == 0).all(), f"{name}: probability leaked to a masked choice"
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[0])), f"{name}: rows don't sum to 1"
    print(f"[{name}] logits{tuple(logits.shape)} masked OK")
    return p


def test_stages():
    head = PolicyHead()
    c = _fake_cache()
    tmove = torch.full((B,), T_MOVE, dtype=torch.long)
    ttrain = torch.full((B,), T_TRAIN, dtype=torch.long)
    probs = []

    m = _mask(B, N_TYPES)
    probs.append(_check("type", head.type_logits(c.core, m), m))

    m = _mask(B, Lu)
    probs.append(_check("entity", head.entity_logits(c.core, tmove, c.unit_tok, m), m))

    m = _mask(B, MAP)
    chosen_unit = c.unit_tok[:, 0]  # pretend unit 0 was chosen
    probs.append(_check("tile", head.tile_logits(c.core, tmove, chosen_unit, c.feature_map, m), m))

    m = _mask(B, K_TRAIN_UNITS)
    chosen_city = c.city_tok[:, 0]
    probs.append(_check("train_unit", head.train_unit_logits(c.core, ttrain, chosen_city, m), m))

    m = _mask(B, K_TECH)
    probs.append(_check("research", head.research_logits(c.core, tmove, m), m))

    # Upgrade: both options always legal (no mask).
    bracket = F.one_hot(torch.randint(0, BRACKET_DIM, (B,)), BRACKET_DIM).float()
    up = head.upgrade_logits(c.core, c.city_tok[:, 0], bracket)
    assert up.shape == (B, 2)
    probs.append(F.softmax(up, dim=-1))
    print(f"[upgrade] logits{tuple(up.shape)} OK")

    # Gradient flow: a non-constant loss over every stage -> backward -> all params get grad.
    loss = sum((p * torch.randn_like(p)).sum() for p in probs)
    loss.backward()
    named = [(n, pr) for n, pr in head.named_parameters() if pr.requires_grad]
    no_grad = [n for n, pr in named if pr.grad is None]
    assert not no_grad, f"params with no grad: {no_grad}"
    total = sum(pr.grad.abs().sum().item() for _, pr in named)
    assert total > 0, "zero gradient"
    print(f"[grad] all policy-head params flow OK (grad_sum={total:.1f})")


if __name__ == "__main__":
    test_stages()
    print("\nAll policy-head checks passed.")
