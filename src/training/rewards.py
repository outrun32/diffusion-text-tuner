"""
Reward models for ReFL training.

QwenYesProbReward: P(yes) from Qwen3.5-9B VLM as differentiable text correctness signal.
Gradient flows: pixels → VLM image encoder → logits → softmax → P(yes).
"""

import torch
from PIL import Image


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
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

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
        rewards = []
        for i in range(images.shape[0]):
            r = self.score_single(images[i], target_texts[i])
            rewards.append(r)
        return torch.stack(rewards)
