"""Pure DPO schedule and objective helpers.

This module is intentionally import-safe for CPU characterization tests: it only
depends on PyTorch and contains no trainer, model-loading, CUDA, or filesystem
side effects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_sigma(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Return shifted flow-matching sigma for training timesteps in ``[0, 1000)``."""
    t_norm = t.float() / 1000.0
    return shift * t_norm / (1.0 + (shift - 1.0) * t_norm)


def time_dependent_beta(t: torch.Tensor, beta_conf: float, shift: float = 3.0) -> torch.Tensor:
    """Return the existing negative DPO beta schedule used by the trainer."""
    sigma_t = compute_sigma(t, shift=shift)
    return -beta_conf * (1.0 - sigma_t) ** 2 / 2.0


def compute_dpo_objective(
    w_policy_loss: torch.Tensor,
    l_policy_loss: torch.Tensor,
    w_ref_loss: torch.Tensor,
    l_ref_loss: torch.Tensor,
    t: torch.Tensor,
    beta_conf: float,
    shift: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DPO sigmoid loss from precomputed per-sample MSE losses.

    Lower model MSE corresponds to a higher implicit likelihood, so log-ratios
    are computed as ``-(policy_loss - ref_loss)``. The beta schedule intentionally
    preserves the trainer's existing negative-beta convention.
    """
    w_log_ratio = -(w_policy_loss.float() - w_ref_loss.float())
    l_log_ratio = -(l_policy_loss.float() - l_ref_loss.float())
    reward_margin = w_log_ratio - l_log_ratio
    beta_t = time_dependent_beta(t, beta_conf=beta_conf, shift=shift).to(reward_margin.device)
    logits = beta_t * reward_margin
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        metrics = {
            "w_log_ratio": w_log_ratio.detach(),
            "l_log_ratio": l_log_ratio.detach(),
            "logits": logits.detach(),
            "beta": beta_t.detach(),
            "reward_margin": reward_margin.detach().mean(),
            "accuracy": (logits > 0).float().mean().detach(),
            "w_policy_loss": w_policy_loss.detach().float().mean(),
            "l_policy_loss": l_policy_loss.detach().float().mean(),
        }

    return loss, metrics
