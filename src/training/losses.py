"""Loss functions for FLUX.2 Klein training.

Currently provides the masked flow-matching loss used by `masked_sft_trainer`.
The loss is a convex blend of a region-weighted (text-mask) MSE and the standard
global flow-matching MSE:

    L = lam * L_masked + (1 - lam) * L_global

where, for packed-latent tensors (B, S, C) with a per-token mask M ∈ [0, 1]
of shape (B, S):

    L_masked  =  mean_b [ sum_s M_bs * mse_bs / (sum_s M_bs + eps) ]
    L_global  =  mean_{b,s} mse_bs
    mse_bs    =  mean_c (pred_bsc - target_bsc) ** 2
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_flow_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_seq: torch.Tensor,
    lam: float = 0.7,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Region-weighted flow-matching loss.

    Args:
        pred:     (B, S, C) predicted velocity.
        target:   (B, S, C) target velocity (= noise - x0).
        mask_seq: (B, S) per-token mask in [0, 1], 1 = inside text region.
        lam:      weight of the masked term in the convex blend.
        eps:      numerical floor for the per-sample mask sum.

    Returns:
        loss: scalar tensor.
        parts: dict with detached "masked" and "global" components for logging.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    pred_f = pred.float()
    target_f = target.float()
    mask_f = mask_seq.float()

    # Per-token mean-squared error over channels: (B, S)
    mse_bs = (pred_f - target_f).pow(2).mean(dim=-1)

    # Masked term: per-sample mask-weighted average, then mean over batch.
    masked_num = (mask_f * mse_bs).sum(dim=-1)            # (B,)
    masked_den = mask_f.sum(dim=-1) + eps                  # (B,)
    l_masked = (masked_num / masked_den).mean()

    # Global term: standard flow-matching MSE.
    l_global = mse_bs.mean()

    loss = lam * l_masked + (1.0 - lam) * l_global

    return loss, {"masked": l_masked.detach(), "global": l_global.detach()}


def mask_to_latent_grid(mask_img: torch.Tensor, latent_hw: tuple[int, int]) -> torch.Tensor:
    """Downsample a per-pixel image mask onto the latent grid via max-pool.

    The FLUX latent grid is image_HW / 16 (VAE x8 + 2x2 patchification).
    Max-pool preserves a token's "any-text-touched" property.

    Args:
        mask_img: (B, 1, H_img, W_img) or (B, H_img, W_img) float in [0, 1].
        latent_hw: (H_lat, W_lat).

    Returns:
        (B, H_lat, W_lat) float in [0, 1].
    """
    if mask_img.dim() == 3:
        mask_img = mask_img.unsqueeze(1)
    pooled = F.adaptive_max_pool2d(mask_img.float(), latent_hw)
    return pooled.squeeze(1)
