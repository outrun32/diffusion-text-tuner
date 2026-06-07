"""Unit tests for src.training.losses."""

from __future__ import annotations

import math

import torch

from src.training.losses import mask_to_latent_grid, masked_flow_matching_loss


def test_zero_error_gives_zero_loss():
    pred = torch.randn(2, 64, 8)
    target = pred.clone()
    mask = torch.ones(2, 64)
    loss, parts = masked_flow_matching_loss(pred, target, mask, lam=0.7)
    assert loss.item() == 0.0
    assert parts["masked"].item() == 0.0
    assert parts["global"].item() == 0.0


def test_lam_zero_matches_global_mse():
    torch.manual_seed(0)
    pred = torch.randn(3, 32, 4)
    target = torch.randn(3, 32, 4)
    mask = torch.zeros(3, 32)  # mask irrelevant when lam=0
    loss, parts = masked_flow_matching_loss(pred, target, mask, lam=0.0)
    expected = (pred - target).pow(2).mean()
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-5)
    assert math.isclose(parts["global"].item(), expected.item(), rel_tol=1e-5)


def test_lam_one_focuses_on_mask_region():
    torch.manual_seed(1)
    B, S, C = 2, 16, 4
    pred = torch.zeros(B, S, C)
    target = torch.zeros(B, S, C)
    # Inject error only in the second half of each sequence.
    target[:, S // 2 :, :] = 1.0
    mask = torch.zeros(B, S)
    # Mask covers only the first half (zero error).
    mask[:, : S // 2] = 1.0

    loss_masked_only, _ = masked_flow_matching_loss(pred, target, mask, lam=1.0)
    loss_global_only, _ = masked_flow_matching_loss(pred, target, mask, lam=0.0)

    # Masked region has zero error, global has nonzero error.
    assert loss_masked_only.item() == 0.0
    assert loss_global_only.item() > 0.0


def test_empty_mask_does_not_nan():
    torch.manual_seed(2)
    pred = torch.randn(2, 16, 4)
    target = torch.randn(2, 16, 4)
    mask = torch.zeros(2, 16)
    loss, parts = masked_flow_matching_loss(pred, target, mask, lam=0.7)
    assert torch.isfinite(loss).item()
    # Masked term should be ~0 (numerator zero, denom = eps).
    assert parts["masked"].item() < 1e-3


def test_gradients_flow_through_pred():
    pred = torch.randn(2, 16, 4, requires_grad=True)
    target = torch.randn(2, 16, 4)
    mask = torch.rand(2, 16)
    loss, _ = masked_flow_matching_loss(pred, target, mask, lam=0.7)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()


def test_mask_to_latent_grid_shape_and_values():
    # 32x32 image mask -> 2x2 latent grid (downsample by 16).
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 0, 0] = 1.0  # one pixel in top-left 16x16 block
    out = mask_to_latent_grid(mask, (2, 2))
    assert out.shape == (1, 2, 2)
    # Max-pool: top-left cell = 1, others = 0.
    assert out[0, 0, 0].item() == 1.0
    assert out[0, 0, 1].item() == 0.0
    assert out[0, 1, 0].item() == 0.0
    assert out[0, 1, 1].item() == 0.0


def test_mask_to_latent_grid_accepts_3d_input():
    mask = torch.ones(2, 32, 32)
    out = mask_to_latent_grid(mask, (4, 4))
    assert out.shape == (2, 4, 4)
    assert torch.allclose(out, torch.ones(2, 4, 4))


if __name__ == "__main__":
    # Allow running without pytest.
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
