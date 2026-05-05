from __future__ import annotations

import importlib
import math
import sys

import numpy as np


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

    assert _char_error_rate(["TECT"], ["ТЕСТ"]) == 0.25

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
