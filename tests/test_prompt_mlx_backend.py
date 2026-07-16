"""Contract tests for the Apple Silicon prompt-generation backend."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


class _FakeTokenizer:
    def apply_chat_template(self, _messages, **_kwargs) -> str:
        return "rendered prompt"


def test_mlx_backend_uses_sampler_and_seed(monkeypatch):
    calls: dict[str, object] = {}

    mlx_module = ModuleType("mlx")
    mlx_core_module = ModuleType("mlx.core")
    mlx_core_module.random = SimpleNamespace(seed=lambda value: calls.setdefault("seed", value))
    mlx_module.core = mlx_core_module

    mlx_lm_module = ModuleType("mlx_lm")

    def fake_load(model_id: str):
        calls["model_id"] = model_id
        return object(), _FakeTokenizer()

    def fake_generate(_model, _tokenizer, **kwargs):
        calls["generate_kwargs"] = kwargs
        return "  generated phrase  "

    mlx_lm_module.load = fake_load
    mlx_lm_module.generate = fake_generate

    sample_utils_module = ModuleType("mlx_lm.sample_utils")

    def fake_make_sampler(**kwargs):
        calls["sampler_kwargs"] = kwargs
        return "sampler"

    sample_utils_module.make_sampler = fake_make_sampler

    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core_module)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils_module)

    from src.prompt_pipeline.llm_client import LLMClient

    client = LLMClient(
        model_id="mlx-community/test-model",
        backend="mlx",
        max_new_tokens=17,
        temperature=0.25,
        seed=73,
    )

    assert client.generate_phrase(3, ["слово"], "poster") == "generated phrase"
    assert calls["seed"] == 73
    assert calls["sampler_kwargs"] == {"temp": 0.25, "top_p": 0.9}
    assert calls["generate_kwargs"] == {
        "prompt": "rendered prompt",
        "max_tokens": 17,
        "sampler": "sampler",
    }


def test_prompt_cli_passes_seed_to_llm(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    from src.prompt_pipeline import generate, llm_client

    monkeypatch.setattr(llm_client, "LLMClient", FakeClient)
    monkeypatch.setattr(generate, "generate_dataset", lambda **_kwargs: None)

    result = generate.main(
        [
            "--n",
            "1",
            "--output",
            str(tmp_path / "prompts.jsonl"),
            "--backend",
            "mlx",
            "--model",
            "mlx-community/test-model",
            "--seed",
            "91",
        ]
    )

    assert result == 0
    assert captured["seed"] == 91
