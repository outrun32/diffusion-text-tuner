"""CPU-safe characterization tests for fragile training objective math."""

from __future__ import annotations

import math

import pytest
import torch

from src.training.flux2_utils import pack_latents, patchify_latents, unpatchify_latents
from src.training.losses import mask_to_latent_grid, masked_flow_matching_loss
from src.training.dpo_objective import (
    compute_dpo_objective,
    compute_sigma,
    time_dependent_beta,
)


def _flow_match_scheduler_class():
    """Import the scheduler through the now import-safe reward boundary."""
    from src.training.refl_trainer import FlowMatchScheduler

    return FlowMatchScheduler


def test_patchify_unpatchify_round_trip_and_pack_ordering():
    latents = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)

    patchified = patchify_latents(latents)
    restored = unpatchify_latents(patchified)
    packed = pack_latents(patchified)

    assert patchified.shape == (1, 8, 2, 2)
    assert torch.equal(restored, latents)
    assert packed.shape == (1, 4, 8)
    assert torch.equal(packed[0, 0], patchified[0, :, 0, 0])
    assert torch.equal(packed[0, 1], patchified[0, :, 0, 1])
    assert torch.equal(packed[0, 2], patchified[0, :, 1, 0])
    assert torch.equal(packed[0, 3], patchified[0, :, 1, 1])


def test_flow_match_scheduler_euler_step_and_step_to_zero():
    FlowMatchScheduler = _flow_match_scheduler_class()
    scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
    scheduler.set_timesteps(num_inference_steps=3)

    model_output = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    sample = torch.tensor([[10.0, 20.0]], dtype=torch.float32)

    stepped = scheduler.step(model_output, sample)
    expected_sigma_delta = scheduler.sigmas[1] - scheduler.sigmas[0]
    assert torch.allclose(stepped, sample + expected_sigma_delta * model_output)

    jumped = scheduler.step_to_zero(model_output, stepped)
    expected_jump = stepped + (0.0 - scheduler.sigmas[1]) * model_output
    assert torch.allclose(jumped, expected_jump)


def test_masked_loss_and_mask_grid_stay_deterministic_for_tiny_tensors():
    pred = torch.zeros(1, 4, 2, requires_grad=True)
    target = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    loss, parts = masked_flow_matching_loss(pred, target, mask, lam=0.5)

    assert torch.allclose(parts["masked"], torch.tensor(2.5), atol=1e-5)
    assert torch.allclose(parts["global"], torch.tensor(1.25), atol=1e-5)
    assert torch.allclose(loss, torch.tensor(1.875), atol=1e-5)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    mask_img = torch.zeros(1, 1, 32, 32)
    mask_img[:, :, 16:, 16:] = 1.0
    mask_grid = mask_to_latent_grid(mask_img, (2, 2))
    assert torch.equal(mask_grid, torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]))


def test_compute_sigma_is_monotonic_and_bounded_for_training_range():
    timesteps = torch.tensor([0.0, 500.0, 999.0])

    sigma = compute_sigma(timesteps, shift=3.0)

    assert torch.all(sigma >= 0.0)
    assert torch.all(sigma <= 1.0)
    assert sigma.tolist() == sorted(sigma.tolist())
    assert sigma[0].item() == 0.0
    assert sigma[-1].item() < 1.0


def test_time_dependent_beta_is_negative_noisier_early_and_scales_with_beta_conf():
    timesteps = torch.tensor([0.0, 500.0, 999.0])

    beta = time_dependent_beta(timesteps, beta_conf=2.0, shift=3.0)
    doubled = time_dependent_beta(timesteps, beta_conf=4.0, shift=3.0)

    assert torch.all(beta <= 0.0)
    assert abs(beta[0].item()) > abs(beta[1].item()) > abs(beta[2].item())
    assert torch.allclose(doubled, beta * 2.0)


@pytest.mark.parametrize(
    ("w_policy", "l_policy", "expected_accuracy", "case_name"),
    [
        (torch.tensor([0.5]), torch.tensor([1.0]), 0.0, "winner policy loss improves"),
        (torch.tensor([1.0]), torch.tensor([0.5]), 1.0, "loser policy loss improves"),
    ],
)
def test_dpo_objective_distinguishes_winner_and_loser_better_cases(
    w_policy: torch.Tensor,
    l_policy: torch.Tensor,
    expected_accuracy: float,
    case_name: str,
):
    del case_name
    ref_loss = torch.tensor([1.0])

    loss, metrics = compute_dpo_objective(
        w_policy_loss=w_policy,
        l_policy_loss=l_policy,
        w_ref_loss=ref_loss,
        l_ref_loss=ref_loss,
        t=torch.tensor([0.0]),
        beta_conf=2.0,
        shift=3.0,
    )

    margin = metrics["w_log_ratio"] - metrics["l_log_ratio"]
    expected_logit = metrics["beta"].to(margin.dtype) * margin

    assert torch.allclose(metrics["logits"], expected_logit)
    assert math.isclose(metrics["accuracy"].item(), expected_accuracy)
    assert torch.isfinite(loss)


def test_negative_beta_convention_gives_lower_loss_when_loser_log_ratio_is_larger():
    ref_loss = torch.tensor([1.0])

    winner_better_loss, winner_metrics = compute_dpo_objective(
        w_policy_loss=torch.tensor([0.5]),
        l_policy_loss=torch.tensor([1.0]),
        w_ref_loss=ref_loss,
        l_ref_loss=ref_loss,
        t=torch.tensor([0.0]),
        beta_conf=2.0,
        shift=3.0,
    )
    loser_better_loss, loser_metrics = compute_dpo_objective(
        w_policy_loss=torch.tensor([1.0]),
        l_policy_loss=torch.tensor([0.5]),
        w_ref_loss=ref_loss,
        l_ref_loss=ref_loss,
        t=torch.tensor([0.0]),
        beta_conf=2.0,
        shift=3.0,
    )

    assert winner_metrics["reward_margin"].item() > 0.0
    assert loser_metrics["reward_margin"].item() < 0.0
    assert winner_better_loss.item() > loser_better_loss.item()


def test_dpo_trainer_re_exports_objective_helpers():
    from src.training import dpo_trainer

    t = torch.tensor([0.0, 500.0])

    assert torch.allclose(dpo_trainer.compute_sigma(t), compute_sigma(t))
    assert torch.allclose(
        dpo_trainer.time_dependent_beta(t, beta_conf=1.5),
        time_dependent_beta(t, beta_conf=1.5),
    )


def test_compute_dpo_loss_delegates_final_objective(monkeypatch: pytest.MonkeyPatch):
    from src.training import dpo_trainer

    class ConstantModel(torch.nn.Module):
        def __init__(self, offset: float):
            super().__init__()
            self.offset = offset

        def forward(self, hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
            return (hidden_states + self.offset,)

    captured: dict[str, torch.Tensor | float] = {}

    def fake_objective(**kwargs: torch.Tensor | float):
        captured.update(kwargs)
        return torch.tensor(12.0), {
            "accuracy": torch.tensor(1.0),
            "reward_margin": torch.tensor(0.5),
            "w_policy_loss": torch.tensor(0.25),
            "l_policy_loss": torch.tensor(0.75),
        }

    monkeypatch.setattr(dpo_trainer, "compute_dpo_objective", fake_objective)

    winner = torch.zeros(1, 2, 2)
    loser = torch.ones(1, 2, 2)
    noise = torch.full((1, 2, 2), 0.25)
    prompt_embeds = torch.zeros(1, 1, 2)
    ids = torch.zeros(1, 2, 4)

    loss, metrics = dpo_trainer.compute_dpo_loss(
        policy=ConstantModel(offset=0.1),
        ref_model=ConstantModel(offset=-0.2),
        winner_packed=winner,
        loser_packed=loser,
        prompt_embeds=prompt_embeds,
        text_ids=torch.zeros(1, 1, 4),
        latent_ids=ids,
        t=torch.tensor([250.0]),
        noise=noise,
        beta_conf=2.0,
        shift=3.0,
        dtype=torch.float32,
    )

    assert loss.item() == 12.0
    assert metrics["accuracy"] == 1.0
    assert captured["beta_conf"] == 2.0
    assert captured["shift"] == 3.0
    assert captured["w_policy_loss"].shape == (1,)
    assert captured["l_ref_loss"].shape == (1,)
