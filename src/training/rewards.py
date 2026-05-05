"""
Reward models for training.

QwenYesProbReward: P(yes) from Qwen3.5-9B VLM as differentiable text correctness signal.
OcrCerEntropyReward: CER + CTC entropy from PaddleOCR v5 as non-differentiable reward.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


class QwenYesProbReward:
    """
    Differentiable reward based on Qwen3.5 VLM yes-token probability.
    Computes P(yes | image, "does this contain '{target_text}'?").

    The image is passed as a raw tensor (B, 3, H, W) in [0, 1] range,
    preprocessed to match the VLM's expected input format with gradients retained.
    """

    PROMPT_TEMPLATE = (
        "Carefully examine each character in this image one by one. "
        'Does this image contain the text "{target_text}" with every single '
        "character rendered accurately and correctly? "
        "Respond with only 'yes' or 'no'."
    )

    SYSTEM_PROMPT = (
        "You are a precise image analysis tool. "
        "Answer ONLY with a single word: 'Yes' or 'No'. Do not explain."
    )

    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B", device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        globals()["torch"] = torch

        self.device = device
        print(f"Loading VLM reward model: {model_id} (4-bit quantized) ...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map=device,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Precompute yes/no token IDs
        tokenizer = self.processor.tokenizer
        yes_variants = ["yes", "Yes", "YES", "да", "Да", "ДА"]
        no_variants = ["no", "No", "NO", "нет", "Нет", "НЕТ"]
        self.yes_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in yes_variants]
        self.no_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in no_variants]
        print(f"  yes token IDs: {self.yes_ids}")
        print(f"  no  token IDs: {self.no_ids}")

    def _build_inputs(self, target_text: str, image_tensor: torch.Tensor):
        """
        Build VLM inputs from a target text and an image tensor.
        image_tensor: (3, H, W) float in [0, 1]
        Returns processor inputs ready for the model.
        """
        prompt_text = self.PROMPT_TEMPLATE.format(target_text=target_text)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        # Convert tensor to PIL for the processor's text tokenization
        img_pil = Image.fromarray(
            (image_tensor.detach().cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        )

        inputs = self.processor(
            text=[text], images=[img_pil], return_tensors="pt", padding=True,
        )
        return inputs

    def score_single(self, image_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        """
        Compute P(yes) for a single image.
        image_tensor: (3, H, W) float [0, 1], WITH gradient graph attached.
        Returns: scalar tensor (differentiable through the model's pixel_values input).
        """
        inputs = self._build_inputs(target_text, image_tensor)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Replace pixel_values with our differentiable tensor
        # The processor created pixel_values from the PIL image; we need to
        # substitute in a version that's connected to the grad graph.
        # We use the processor's image_processor to get the right normalization
        # but apply it to our differentiable tensor.
        pv_shape = inputs["pixel_values"].shape
        pv_dtype = inputs["pixel_values"].dtype

        # Resize the image tensor to match what the processor expects
        diff_pixels = self._preprocess_differentiable(image_tensor, pv_shape, pv_dtype)
        inputs["pixel_values"] = diff_pixels

        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token position

        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = sum(probs[tid] for tid in self.yes_ids)
        p_no = sum(probs[tid] for tid in self.no_ids)
        total = p_yes + p_no
        reward = p_yes / (total + 1e-8)

        return reward

    def _preprocess_differentiable(
        self, image_tensor: torch.Tensor, target_shape: torch.Size, target_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Apply Qwen2VL image preprocessing differentiably.

        Qwen2VL packs images into patches of shape:
          (T_grid * H_grid * W_grid, temporal_patch_size * patch_size * patch_size * C)
        For a single 512x512 image: (1024, 1536) with image_grid_thw = [1, 32, 32].

        image_tensor: (3, H, W) float [0, 1]
        Returns: pixel_values matching target_shape, with grad graph intact.
        """
        C, H, W = image_tensor.shape
        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2

        # Pad to multiples of patch_size if needed
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        img = image_tensor
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))

        _, H_pad, W_pad = img.shape
        H_grid = H_pad // patch_size
        W_grid = W_pad // patch_size

        # Normalize: [0,1] → [-1,1] (Qwen3.5 uses mean=0.5, std=0.5)
        img = img * 2.0 - 1.0

        # Duplicate frame for temporal dimension: (C, H, W) → (tps, C, H, W)
        grid_t = 1
        img = img.unsqueeze(0).expand(temporal_patch_size, C, H_pad, W_pad)

        # Reshape into patches with merge_size ordering:
        # (tps, C, H_pad, W_pad) → (gt, tps, C, gh//ms, ms, ps, gw//ms, ms, ps)
        img = img.reshape(
            grid_t,
            temporal_patch_size,
            C,
            H_grid // merge_size,
            merge_size,
            patch_size,
            W_grid // merge_size,
            merge_size,
            patch_size,
        )
        # Permute to: (gt, gh//ms, gw//ms, ms, ms, C, tps, ps, ps)
        img = img.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        # Flatten to: (num_patches, C * tps * ps * ps)
        img = img.reshape(
            grid_t * H_grid * W_grid,
            C * temporal_patch_size * patch_size * patch_size,
        )

        assert img.shape == target_shape, (
            f"Differentiable pixel_values shape {img.shape} != expected {target_shape}"
        )
        return img.to(dtype=target_dtype, device=self.device)

    def score_batch(self, images: torch.Tensor, target_texts: list[str]) -> torch.Tensor:
        """
        Score a batch of images. Returns (B,) tensor of P(yes) rewards.
        images: (B, 3, H, W) float [0, 1]
        """
        import torch

        rewards = []
        for i in range(images.shape[0]):
            r = self.score_single(images[i], target_texts[i])
            rewards.append(r)
        return torch.stack(rewards)


# ── OCR CER + Entropy reward ───────────────────────────────────────────────
# Patch must be applied lazily before PaddleOCR is imported, not at module import
# time. This keeps default pytest collection import-safe without OCR extras.

_raw_preds_store: dict = {}
_original_ctc_call = None
_ctc_patch_applied = False


def _patched_ctc_call(self, pred, return_word_box=False, **kwargs):
    preds = np.array(pred[0])  # (batch, T, vocab)
    # Accumulate all decoder calls per image (multi-region detections produce
    # multiple decoder calls). Score code clears the list before each image.
    _raw_preds_store.setdefault("calls", []).append(preds)
    _raw_preds_store["last"] = preds
    return _original_ctc_call(self, pred, return_word_box=return_word_box, **kwargs)


def _ensure_ctc_capture_patch() -> None:
    """Install the PaddleOCR CTC capture patch when OCR scoring is requested."""
    global _ctc_patch_applied, _original_ctc_call

    if _ctc_patch_applied:
        return

    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    import paddlex.inference.models.text_recognition.processors as proc_mod

    _original_ctc_call = proc_mod.CTCLabelDecode.__call__
    proc_mod.CTCLabelDecode.__call__ = _patched_ctc_call
    _ctc_patch_applied = True

# Latin → Cyrillic homoglyph map for visually identical glyphs.
# Applied before CER so that OCR decoding "TELEFONA" vs target "ТЕЛЕФОНА"
# only penalises genuinely different characters (Л/L, Ф/F), not visual matches.
_LATIN_TO_CYRILLIC: dict[str, str] = {
    # uppercase
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
    # lowercase
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х",
}
_HOMOGLYPH_TABLE = str.maketrans(_LATIN_TO_CYRILLIC)


def _normalize_homoglyphs(text: str) -> str:
    """Replace Latin homoglyphs with their Cyrillic equivalents."""
    return text.translate(_HOMOGLYPH_TABLE)


def _char_error_rate(hypothesis_lines, reference_words) -> float:
    ref = _normalize_homoglyphs("".join(reference_words)).lower()
    hyp = _normalize_homoglyphs("".join(hypothesis_lines)).lower()
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        ndp = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            ndp[j] = min(ndp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost)
        dp = ndp
    return dp[m] / max(len(ref), 1)


def _ctc_entropy_stats(preds_np) -> dict:
    row_sums = preds_np.sum(axis=-1)
    is_probs = bool(np.allclose(row_sums, 1.0, atol=0.05))
    if is_probs:
        probs = preds_np
    else:
        exp = np.exp(preds_np - preds_np.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)

    blank_id = 0
    pred_ids = probs.argmax(axis=-1)
    non_blank = pred_ids != blank_id
    entropy = -(probs * np.log(probs + 1e-9)).sum(axis=-1)

    if non_blank.sum() == 0:
        return {"entropy": float("nan"), "min_p": float("nan"), "frac_unc": float("nan")}

    top_p = probs.max(axis=-1)[non_blank]
    return {
        "entropy": float(entropy[non_blank].mean()),
        "min_p": float(top_p.min()),
        "frac_unc": float((top_p < 0.5).mean()),
    }


class OcrCerEntropyReward:
    """OCR reward matching the validated April 2026 CTC analysis.

    R_OCR = (1 - CER) * exp(-λ * H_nb)

    where H_nb is the mean entropy over non-blank CTC frames captured via
    the module-level CTC decoder monkey-patch.
    """

    def __init__(
        self,
        lang: str = "ru",
        device: str = "cpu",
        entropy_lambda: float = 1.0,
        text_recognition_model_name: str = "eslav_PP-OCRv5_mobile_rec",
        text_detection_model_name: str = "PP-OCRv5_mobile_det",
        text_det_limit_side_len: int = 1280,
        text_det_limit_type: str = "max",
        text_det_thresh: float = 0.2,
        text_det_box_thresh: float = 0.3,
    ):
        _ensure_ctc_capture_patch()

        from paddleocr import PaddleOCR

        self.entropy_lambda = entropy_lambda
        print(
            f"Loading OCR reward model: {text_recognition_model_name} "
            f"(det={text_detection_model_name}, device={device}, "
            f"side_len={text_det_limit_side_len}/{text_det_limit_type})"
        )
        self.ocr = PaddleOCR(
            text_recognition_model_name=text_recognition_model_name,
            text_detection_model_name=text_detection_model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
        )

    def score(self, image_path: str, target_text: str) -> dict:
        """Score a single image using CER + entropy over raw CTC predictions."""
        _raw_preds_store.clear()
        result = self.ocr.ocr(image_path)
        res = result[0] if isinstance(result, list) else result

        rec_texts = res["rec_texts"]
        rec_scores = res["rec_scores"]
        official_conf = float(np.mean(rec_scores)) if rec_scores else float("nan")

        # Concatenate all batches across decoder calls into one (N, T, V) array.
        calls = _raw_preds_store.get("calls", [])
        if calls:
            batches = [c if c.ndim == 3 else c[np.newaxis, ...] for c in calls]
            # Pad to the max T length so we can concat along batch dim.
            max_t = max(b.shape[1] for b in batches)
            padded = []
            for b in batches:
                if b.shape[1] < max_t:
                    pad_width = ((0, 0), (0, max_t - b.shape[1]), (0, 0))
                    b = np.pad(b, pad_width, mode="edge")
                padded.append(b)
            raw = np.concatenate(padded, axis=0)
            stats = [_ctc_entropy_stats(raw[i]) for i in range(raw.shape[0])]
            entropies = [s["entropy"] for s in stats if not math.isnan(s["entropy"])]
            min_ps = [s["min_p"] for s in stats if not math.isnan(s["min_p"])]
            frac_uncs = [s["frac_unc"] for s in stats if not math.isnan(s["frac_unc"])]
            mean_entropy = float(np.mean(entropies)) if entropies else float("nan")
            min_p = float(np.mean(min_ps)) if min_ps else float("nan")
            frac_unc = float(np.mean(frac_uncs)) if frac_uncs else float("nan")
        else:
            mean_entropy = float("nan")
            min_p = float("nan")
            frac_unc = float("nan")

        cer = min(1.0, _char_error_rate(rec_texts, target_text.split()))
        reward_ocr = (
            (1.0 - cer) * math.exp(-self.entropy_lambda * mean_entropy)
            if not math.isnan(mean_entropy)
            else (1.0 - cer)
        )

        return {
            "rec_texts": rec_texts,
            "official_conf": official_conf,
            "cer": cer,
            "entropy": mean_entropy,
            "min_p": min_p,
            "frac_unc": frac_unc,
            "reward_ocr": float(reward_ocr),
            "ocr_detected": " ".join(rec_texts),
        }

    def score_pil(self, image: Image.Image, target_text: str) -> dict:
        """Score a PIL image by saving it to a temporary PNG."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image.save(handle, format="PNG")
            tmp_path = handle.name
        try:
            return self.score(tmp_path, target_text)
        finally:
            os.unlink(tmp_path)
