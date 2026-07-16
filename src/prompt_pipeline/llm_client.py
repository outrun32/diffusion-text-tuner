"""LLM client for phrase generation and scene pool expansion.

Supports three backends:
  • "transformers" — local model via HuggingFace transformers (CUDA / CPU)
  • "vllm"         — high-throughput inference via vLLM (CUDA, FP8 dynamic)
  • "mlx"          — local model via mlx-lm (Apple Silicon)

When no LLM is available the pipeline falls back to algorithmic text
generation (see TextGenerator.generate_text_fallback).
"""

from __future__ import annotations

import logging
import re

from .config import (
    CONTENT_TYPE_RU,
    LLM_PHRASE_PROMPTS,
    LLM_SCENE_PROMPT,
    LLM_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around a local text-generation model."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-4B",
        backend: str = "transformers",
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        device: str | None = None,
        seed: int = 42,
    ):
        self.model_id = model_id
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self._model = None
        self._tokenizer = None
        self._generate_fn = None
        self._device = device
        self._thinking_checked = False
        self._supports_thinking = False

        self._load(backend)

    # ------------------------------------------------------------------
    # Chat template helper  (auto-disables <think> for Qwen 3 / 3.5)
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict]) -> str:
        if not self._thinking_checked:
            self._thinking_checked = True
            try:
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                self._supports_thinking = True
                logger.info("Thinking mode detected — disabled for generation")
                return text
            except TypeError:
                self._supports_thinking = False

        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._supports_thinking:
            kwargs["enable_thinking"] = False
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load(self, backend: str):
        if backend == "mlx":
            self._load_mlx()
        elif backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s via transformers …", self.model_id, device)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=device,
        )
        self._model.eval()

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id

        def _gen(prompt: str) -> str:
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = self._apply_chat_template(messages)
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    top_p=0.9,
                    pad_token_id=pad_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        self._generate_fn = _gen

    def _load_vllm(self):
        from vllm import LLM, SamplingParams  # noqa: F811

        logger.info("Loading %s via vLLM (FP8 dynamic) …", self.model_id)

        self._model = LLM(
            model=self.model_id,
            quantization="fp8",
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )
        self._tokenizer = self._model.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.9,
            max_tokens=self.max_new_tokens,
            seed=self.seed,
        )

        def _gen(prompt: str) -> str:
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = self._apply_chat_template(messages)
            outputs = self._model.generate([text], self._sampling_params)
            return outputs[0].outputs[0].text.strip()

        self._generate_fn = _gen

    def _load_mlx(self):
        import mlx.core as mx
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load
        from mlx_lm.sample_utils import make_sampler

        logger.info("Loading %s via mlx-lm …", self.model_id)
        mx.random.seed(self.seed)
        self._model, self._tokenizer = mlx_load(self.model_id)
        sampler = make_sampler(temp=self.temperature, top_p=0.9)

        def _gen(prompt: str) -> str:
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = self._apply_chat_template(messages)
            return mlx_generate(
                self._model,
                self._tokenizer,
                prompt=text,
                max_tokens=self.max_new_tokens,
                sampler=sampler,
            ).strip()

        self._generate_fn = _gen

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_phrase(
        self, tier: int, must_include: list[str], content_type: str = "poster"
    ) -> str:
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

    def generate_phrases_batch(
        self,
        items: list[tuple[int, list[str], str]],
    ) -> list[str]:
        """Batch-generate phrases.

        *items* = [(tier, must_include, content_type), …].
        Uses native batching for the vLLM backend; falls back to
        sequential generation for transformers / mlx.
        """
        if self.backend == "vllm":
            return self._batch_vllm(items)
        return [self.generate_phrase(tier, mi, ct) for tier, mi, ct in items]

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _batch_vllm(self, items):
        """Native vLLM batch generation."""
        prompts = []
        for tier, must_include, ct in items:
            template = LLM_PHRASE_PROMPTS[tier]
            user_prompt = template.format(
                content_type_ru=CONTENT_TYPE_RU.get(ct, ct),
                words=", ".join(must_include),
            )
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompts.append(self._apply_chat_template(messages))
        outputs = self._model.generate(prompts, self._sampling_params)
        return [self._clean_phrase(o.outputs[0].text) for o in outputs]

    @staticmethod
    def _clean_phrase(text: str) -> str:
        """Strip quotes, think-tags, and stray whitespace from LLM output."""
        # Safety net: remove <think>…</think> blocks if any leak through
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip().strip("«»\"'")
        # Collapse internal whitespace
        text = re.sub(r"[ \t]+", " ", text)
        return text
