"""Import-safe FLUX image generation pipeline implementation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for the generated-image pipeline."""

    prompts: Path
    output_dir: Path = Path("outputs/generated")
    model_id: str = "black-forest-labs/FLUX.2-klein-base-4B"
    lora_path: str | None = None
    versions_per_prompt: int = 5
    batch_size: int = 4
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    resolution: int = 512
    seed: int = 42
    start_idx: int = 0
    end_idx: int | None = None
    save_latents: bool = True
    save_png: bool = True


@dataclass(frozen=True)
class GenerationPaths:
    """Filesystem paths used by the generated-image pipeline."""

    output_dir: Path
    latents_dir: Path
    text_embeds_dir: Path
    images_dir: Path
    manifest_path: Path


def load_prompt_records(
    prompts_path: Path | str,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> list[dict[str, Any]]:
    """Load prompt JSONL records and apply the CLI-compatible index slice."""
    records: list[dict[str, Any]] = []
    with Path(prompts_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            _ = record["prompt"]
            records.append(record)
    return records[start_idx:end_idx]


def resolve_generation_paths(output_dir: Path | str) -> GenerationPaths:
    """Resolve deterministic generated-image output paths without side effects."""
    root = Path(output_dir)
    return GenerationPaths(
        output_dir=root,
        latents_dir=root / "latents",
        text_embeds_dir=root / "text_embeds",
        images_dir=root / "images",
        manifest_path=root / "manifest.json",
    )


def plan_generation_seed(
    *,
    seed: int,
    prompt_index: int,
    versions_per_prompt: int,
    version: int,
) -> int:
    """Return the deterministic per-prompt/version seed used by the original script."""
    return seed + prompt_index * versions_per_prompt + version


def _write_manifest(config: GenerationConfig, paths: GenerationPaths, num_prompts: int) -> None:
    manifest = {
        "model_id": config.model_id,
        "lora_path": config.lora_path,
        "versions_per_prompt": config.versions_per_prompt,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "resolution": config.resolution,
        "num_prompts": num_prompts,
        "start_idx": config.start_idx,
    }
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_generation(config: GenerationConfig) -> None:
    """Generate prompt variants, latents, text embeddings, images, and a manifest."""
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from tqdm import tqdm

    from src.training.flux2_utils import encode_image

    paths = resolve_generation_paths(config.output_dir)
    paths.latents_dir.mkdir(parents=True, exist_ok=True)
    paths.text_embeds_dir.mkdir(parents=True, exist_ok=True)
    paths.images_dir.mkdir(parents=True, exist_ok=True)

    records = load_prompt_records(
        config.prompts,
        start_idx=config.start_idx,
        end_idx=config.end_idx,
    )
    logger.info(
        "Loaded %d prompts (indices %d–%d)",
        len(records),
        config.start_idx,
        config.start_idx + len(records) - 1,
    )

    logger.info("Loading pipeline: %s", config.model_id)
    pipe = Flux2KleinPipeline.from_pretrained(config.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    if config.lora_path:
        logger.info("Loading LoRA: %s", config.lora_path)
        try:
            pipe.load_lora_weights(config.lora_path)
        except ValueError as exc:
            logger.warning(
                "Diffusers LoRA loading failed; trying PEFT adapter format: %s",
                exc,
            )
            from peft import PeftModel

            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer,
                config.lora_path,
            ).to("cuda")

    vae = pipe.vae

    for rec_idx, record in enumerate(tqdm(records, desc="Generating")):
        global_idx = config.start_idx + rec_idx
        prompt_id = f"{global_idx:06d}"
        prompt = record["prompt"]
        target_text = record.get("target_text", "")

        prompt_latent_dir = paths.latents_dir / prompt_id
        prompt_image_dir = paths.images_dir / prompt_id
        prompt_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_image_dir.mkdir(parents=True, exist_ok=True)

        embed_path = paths.text_embeds_dir / f"{prompt_id}.pt"
        if not embed_path.exists():
            with torch.no_grad():
                prompt_embeds = pipe._get_qwen3_prompt_embeds(
                    text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer,
                    prompt=[prompt],
                    device="cuda",
                    max_sequence_length=512,
                    hidden_states_layers=(9, 18, 27),
                )
            torch.save(
                {
                    "prompt_embeds": prompt_embeds[0].cpu(),
                    "target_text": target_text,
                    "prompt": prompt,
                },
                embed_path,
            )

        for version in range(config.versions_per_prompt):
            latent_path = prompt_latent_dir / f"v{version}.pt"
            image_path = prompt_image_dir / f"v{version}.png"

            if latent_path.exists() and image_path.exists():
                continue

            generation_seed = plan_generation_seed(
                seed=config.seed,
                prompt_index=global_idx,
                versions_per_prompt=config.versions_per_prompt,
                version=version,
            )
            generator = torch.Generator(device="cuda").manual_seed(generation_seed)

            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    height=config.resolution,
                    width=config.resolution,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
                    output_type="pil",
                )

            pil_image = output.images[0]

            if config.save_png:
                pil_image.save(image_path)

            if config.save_latents:
                img_array = np.asarray(pil_image.convert("RGB"), dtype="uint8")
                img_tensor = (
                    torch.from_numpy(img_array)
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .div(255.0)
                    .to("cuda", dtype=torch.bfloat16)
                )
                latent = encode_image(img_tensor, vae)
                torch.save({"latent": latent[0].cpu()}, latent_path)

    _write_manifest(config, paths, len(records))

    logger.info(
        "Done. Generated %d × %d = %d images → %s",
        len(records),
        config.versions_per_prompt,
        len(records) * config.versions_per_prompt,
        config.output_dir,
    )
