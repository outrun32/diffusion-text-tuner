# Testing Patterns

**Analysis Date:** 2026-05-04

## Test Framework

**Runner:**
- Pytest-compatible function tests - Version not pinned because no dependency manifest is committed.
- Config: Not detected. No `pytest.ini`, `pyproject.toml`, `setup.cfg`, or `tox.ini` exists at the repository root.
- Script fallback: `tests/test_losses.py` includes an `if __name__ == "__main__"` runner so it can run without pytest.

**Assertion Library:**
- Python `assert` statements and `math.isclose` in `tests/test_losses.py`.
- PyTorch assertions with `torch.isfinite` and `torch.allclose` in `tests/test_losses.py`.

**Run Commands:**
```bash
python -m pytest tests              # Run formal pytest-style tests when pytest is installed
python tests/test_losses.py         # Run current formal tests without pytest
python scripts/test_gradient_flow.py # Manual CUDA/model integration diagnostic, not a normal CI test
```

## Test File Organization

**Location:**
- Formal tests live under `tests/`; currently only `tests/test_losses.py` is present.
- GPU/model diagnostics live under `scripts/`, e.g. `scripts/test_gradient_flow.py` and `scripts/test_grad_magnitude.py`.
- Experimental reward-model tests live under `experiments/ocr_reward_tests/`, e.g. `experiments/ocr_reward_tests/test_qwen_yes_prob.py` and `experiments/ocr_reward_tests/test_paddleocr.py`.

**Naming:**
- Formal test functions use `test_*` names in `tests/test_losses.py`.
- Diagnostic scripts use `test_*.py` filenames but are not isolated pytest tests: `scripts/test_gradient_flow.py` executes expensive CUDA/model work at import time.
- Experimental scripts under `experiments/ocr_reward_tests/` use `test_*.py` names but require local assets/models and should be treated as research scripts.

**Structure:**
```text
tests/
└── test_losses.py              # Fast unit tests for pure tensor loss utilities

scripts/
├── test_gradient_flow.py       # Manual end-to-end gradient-flow diagnostic
└── test_grad_magnitude.py      # Manual gradient magnitude diagnostic

experiments/ocr_reward_tests/
├── test_qwen_yes_prob.py       # Local MLX/Qwen reward experiment
├── test_paddleocr.py           # PaddleOCR experiment
├── test_paddleocr_v5.py        # PaddleOCR v5 experiment
├── test_paddleocr_extended.py  # Extended OCR experiment
└── test_trocr.py               # TrOCR experiment
```

## Test Structure

**Suite Organization:**
```python
def test_zero_error_gives_zero_loss():
    pred = torch.randn(2, 64, 8)
    target = pred.clone()
    mask = torch.ones(2, 64)
    loss, parts = masked_flow_matching_loss(pred, target, mask, lam=0.7)
    assert loss.item() == 0.0
    assert parts["masked"].item() == 0.0
    assert parts["global"].item() == 0.0
```

**Patterns:**
- Unit tests construct small tensors directly in memory in `tests/test_losses.py`.
- Deterministic tests call `torch.manual_seed(...)` when random tensors are compared numerically in `tests/test_losses.py`.
- Tests verify both scalar values and returned diagnostic parts from `masked_flow_matching_loss` in `src/training/losses.py`.
- Tests cover edge cases such as empty masks, all-zero error, mask-only regions, gradient flow, and mask downsampling in `tests/test_losses.py`.
- Manual diagnostics print status and require CUDA plus local model/artifact access in `scripts/test_gradient_flow.py` and `scripts/test_grad_magnitude.py`.

## Mocking

**Framework:** Not used in the formal tests.

**Patterns:**
```python
# Current pattern is no mocking: construct tensors and call pure functions directly.
pred = torch.randn(2, 16, 4, requires_grad=True)
target = torch.randn(2, 16, 4)
mask = torch.rand(2, 16)
loss, _ = masked_flow_matching_loss(pred, target, mask, lam=0.7)
loss.backward()
assert pred.grad is not None
```

**What to Mock:**
- Mock or fake filesystem paths when adding tests for `src/training/dataset.py` using temporary `.pt`, `.csv`, and `.jsonl` files.
- Mock model-loading boundaries for tests around `src/training/rewards.py`, `src/training/flux2_utils.py`, `src/training/sft_trainer.py`, and `src/prompt_pipeline/llm_client.py` to avoid downloading/loading FLUX, Qwen, PaddleOCR, vLLM, or MLX models.
- Mock `subprocess.run` for tests around `scripts/synth/build_dataset.py` render orchestration.

**What NOT to Mock:**
- Do not mock pure tensor math in `src/training/losses.py`; construct small tensors and verify exact/close behavior.
- Do not mock collate functions when validating shape/padding behavior in `src/training/dataset.py`; use direct in-memory sample dictionaries.
- Do not run full FLUX/Qwen/PaddleOCR model loads in normal unit tests; keep those as manual integration diagnostics under `scripts/` or `experiments/`.

## Fixtures and Factories

**Test Data:**
```python
torch.manual_seed(1)
B, S, C = 2, 16, 4
pred = torch.zeros(B, S, C)
target = torch.zeros(B, S, C)
target[:, S // 2 :, :] = 1.0
mask = torch.zeros(B, S)
mask[:, : S // 2] = 1.0
```

**Location:**
- No reusable pytest fixtures or factories are present.
- Formal tests create data inline in `tests/test_losses.py`.
- Manual experiment assets live under `experiments/assets/`; `.gitignore` explicitly allows `.png` and `.jpg` there.
- Synthetic fixture generation utilities live under `scripts/synth/make_fixture.py`, but they are not wired into formal tests.

## Coverage

**Requirements:** None enforced. No coverage config, CI coverage gate, or coverage command is committed.

**View Coverage:**
```bash
python -m pytest --cov=src tests     # Suggested command if pytest-cov is installed
python tests/test_losses.py          # Current dependency-light smoke/unit run
```

## Test Types

**Unit Tests:**
- Scope and approach: `tests/test_losses.py` covers pure tensor utilities in `src/training/losses.py`.
- Add new unit tests under `tests/` for pure utilities, collators, config parsing, and path/data transforms.

**Integration Tests:**
- Scope and approach: No formal integration test suite is present.
- Existing manual integration diagnostics are `scripts/test_gradient_flow.py` and `scripts/test_grad_magnitude.py`, which load FLUX/Qwen models, require CUDA, and assume files such as `outputs/text_embeds/000000.pt`.

**E2E Tests:**
- Framework: Not used.
- Current end-to-end validation is manual through README pipeline commands using `scripts/generate_images.py`, `scripts/score_images.py`, `src/training/sft_trainer.py`, and `src/training/dpo_trainer.py`.

## Common Patterns

**Async Testing:**
```python
# Not currently used. No asyncio or async test framework detected.
```

**Error Testing:**
```python
def test_empty_mask_does_not_nan():
    torch.manual_seed(2)
    pred = torch.randn(2, 16, 4)
    target = torch.randn(2, 16, 4)
    mask = torch.zeros(2, 16)
    loss, parts = masked_flow_matching_loss(pred, target, mask, lam=0.7)
    assert torch.isfinite(loss).item()
    assert parts["masked"].item() < 1e-3
```

**Gradient Testing:**
```python
def test_gradients_flow_through_pred():
    pred = torch.randn(2, 16, 4, requires_grad=True)
    target = torch.randn(2, 16, 4)
    mask = torch.rand(2, 16)
    loss, _ = masked_flow_matching_loss(pred, target, mask, lam=0.7)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
```

---

*Testing analysis: 2026-05-04*
