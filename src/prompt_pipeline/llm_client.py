"""LLM client for phrase generation and scene pool expansion.

Supports two backends:
  • "transformers" — local model via HuggingFace transformers (CUDA / CPU)
  • "mlx"         — local model via mlx-lm (Apple Silicon)

When no LLM is available the pipeline falls back to algorithmic text
generation (see TextGenerator.generate_text_fallback).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .config import (
    CONTENT_TYPE_RU,
    LLM_PHRASE_PROMPTS,
    LLM_SCENE_PROMPT,
    LLM_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around a local text-generation model."""

    def __init__(self, model_id: str = "Qwen/Qwen3-4B",
                 backend: str = "transformers",
                 max_new_tokens: int = 128,
                 temperature: float = 0.7,
                 device: str | None = None):
        self.model_id = model_id
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._generate_fn = None
        self._device = device

        self._load(backend)

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load(self, backend: str):
        if backend == "mlx":
            self._load_mlx()
        else:
            self._load_transformers()

    def _load_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s via transformers …", self.model_id, device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=device,
        )
        self._model.eval()

        def _gen(prompt: str) -> str:
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    top_p=0.9,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        self._generate_fn = _gen

    def _load_mlx(self):
        from mlx_lm import load as mlx_load, generate as mlx_generate

        logger.info("Loading %s via mlx-lm …", self.model_id)
        self._model, self._tokenizer = mlx_load(self.model_id)

        def _gen(prompt: str) -> str:
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return mlx_generate(
                self._model, self._tokenizer, prompt=text,
                max_tokens=self.max_new_tokens, temp=self.temperature,
            ).strip()

        self._generate_fn = _gen

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_phrase(self, tier: int, must_include: list[str],
                        content_type: str = "poster") -> str:
        """Generate a natural phrase incorporating *must_include* words."""
        template = LLM_PHRASE_PROMPTS.get(tier)
        if template is None:
            raise ValueError(f"No LLM template for tier {tier}")

        prompt = template.format(
            content_type_ru=CONTENT_TYPE_RU.get(content_type, content_type),
            words=", ".join(must_include),
        )
        raw = self._generate_fn(prompt)
        return self._clean_phrase(raw)

    def generate_scenes(self, content_type: str, n: int = 20) -> list[str]:
        """Generate *n* scene descriptions for a content type."""
        prompt = LLM_SCENE_PROMPT.format(
            n=n,
            category=CONTENT_TYPE_RU.get(content_type, content_type),
        )
        raw = self._generate_fn(prompt)
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        # Remove numbering artefacts ("1. ...", "- ...", etc.)
        cleaned = []
        for ln in lines:
            ln = re.sub(r"^\d+[\.\)]\s*", "", ln)
            ln = re.sub(r"^[-•]\s*", "", ln)
            if ln:
                cleaned.append(ln)
        return cleaned

    # ------------------------------------------------------------------

    @staticmethod
    def _clean_phrase(text: str) -> str:
        """Strip quotes and stray whitespace from LLM output."""
        text = text.strip().strip("«»\"'""''")
        # Collapse internal whitespace
        text = re.sub(r"[ \t]+", " ", text)
        return text
