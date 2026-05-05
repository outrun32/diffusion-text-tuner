from __future__ import annotations

import importlib
import math
import sys

import numpy as np
from PIL import Image


def test_reward_module_import_is_safe_without_optional_ocr_or_model_packages(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "src.training.rewards" or module_name.startswith("src.training.rewards."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    optional_roots = ["paddleocr", "paddlex", "transformers"]
    for module_name in optional_roots:
        monkeypatch.setitem(sys.modules, module_name, None)

    rewards = importlib.import_module("src.training.rewards")

    assert rewards.QwenYesProbReward.PROMPT_TEMPLATE
    assert rewards.OcrCerEntropyReward.__name__ == "OcrCerEntropyReward"
    assert not any(
        module_name in sys.modules and sys.modules[module_name] is not None
        for module_name in optional_roots
    )


def test_normalize_homoglyphs_maps_visual_latin_letters_to_cyrillic():
    from src.training.rewards import _normalize_homoglyphs

    assert _normalize_homoglyphs("ABCEHKMOPT X") == "АВСЕНКМОРТ Х"
    assert _normalize_homoglyphs("aceopx") == "асеорх"


def test_char_error_rate_and_ctc_entropy_stats_are_deterministic_for_tiny_inputs():
    from src.training.rewards import _char_error_rate, _ctc_entropy_stats

    assert _char_error_rate(["TEFT"], ["ТЕСТ"]) == 0.25

    probs = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float64,
    )
    stats = _ctc_entropy_stats(probs)
    expected_entropies = -(probs[1:] * np.log(probs[1:] + 1e-9)).sum(axis=-1)

    assert stats == {
        "entropy": float(expected_entropies.mean()),
        "min_p": 0.6,
        "frac_unc": 0.0,
    }


def test_qwen_score_batch_preserves_order_with_fake_score_single():
    import torch

    from src.training.rewards import QwenYesProbReward

    reward = object.__new__(QwenYesProbReward)
    calls: list[tuple[float, str]] = []

    def fake_score_single(image_tensor, target_text: str):
        calls.append((float(image_tensor.item()), target_text))
        return torch.tensor(float(image_tensor.item()) + len(target_text) / 10.0)

    reward.score_single = fake_score_single

    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    scores = reward.score_batch(images, ["А", "ББ", "ВВВ"])

    assert calls == [(1.0, "А"), (2.0, "ББ"), (3.0, "ВВВ")]
    assert torch.equal(scores, torch.tensor([1.1, 2.2, 3.3]))


def test_ocr_score_uses_fake_ocr_and_raw_ctc_predictions_deterministically():
    from src.training import rewards
    from src.training.rewards import OcrCerEntropyReward, _ctc_entropy_stats

    raw_probs = np.array(
        [
            [0.10, 0.80, 0.10],
            [0.15, 0.20, 0.65],
            [0.90, 0.05, 0.05],
        ],
        dtype=np.float64,
    )

    class FakeOcr:
        def ocr(self, image_path: str):
            assert image_path == "fake-image.png"
            rewards._raw_preds_store.setdefault("calls", []).append(raw_probs)
            return [{"rec_texts": ["ТЕСТ"], "rec_scores": [0.7, 0.9]}]

    reward = object.__new__(OcrCerEntropyReward)
    reward.entropy_lambda = 0.25
    reward.ocr = FakeOcr()

    result = reward.score("fake-image.png", "ТЕСТ")
    stats = _ctc_entropy_stats(raw_probs)

    assert result["rec_texts"] == ["ТЕСТ"]
    assert result["official_conf"] == 0.8
    assert result["cer"] == 0.0
    assert result["entropy"] == stats["entropy"]
    assert result["min_p"] == stats["min_p"]
    assert result["frac_unc"] == stats["frac_unc"]
    assert result["reward_ocr"] == math.exp(-0.25 * stats["entropy"])
    assert result["ocr_detected"] == "ТЕСТ"


def test_ocr_score_pil_writes_temporary_png_and_removes_it():
    from src.training.rewards import OcrCerEntropyReward

    reward = object.__new__(OcrCerEntropyReward)
    seen_paths: list[str] = []

    def fake_score(image_path: str, target_text: str):
        assert target_text == "ТЕСТ"
        seen_paths.append(image_path)
        with open(image_path, "rb") as handle:
            assert handle.read(8).startswith(b"\x89PNG")
        return {"reward_ocr": 0.75}

    reward.score = fake_score
    result = reward.score_pil(Image.new("RGB", (2, 2), color="white"), "ТЕСТ")

    assert result == {"reward_ocr": 0.75}
    assert len(seen_paths) == 1
    assert not __import__("os").path.exists(seen_paths[0])
