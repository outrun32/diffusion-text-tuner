"""
FLUX2Klein utility functions extracted from the diffusers pipeline.

Handles latent packing/unpacking, BN normalization, text encoding precomputation,
and VAE decoding — all adapted for use in the ReFL training loop.
"""

import json
import os
from pathlib import Path

import torch
from tqdm import tqdm


# ── Latent packing / unpacking ──────────────────────────────────────────────


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*4, H//2, W//2)"""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(B, C * 4, H // 2, W // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C*4, H//2, W//2) -> (B, C, H, W)"""
    B, C4, H, W = latents.shape
    latents = latents.reshape(B, C4 // 4, 2, 2, H, W)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(B, C4 // 4, H * 2, W * 2)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)"""
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    """(B, seq, C) + (B, seq, 4) -> (B, C, H, W)"""
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1
        flat_ids = h_ids * w + w_ids
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """Generate 4D position IDs (T,H,W,L) for patchified latents (B, C, H, W)."""
    B, _, H, W = latents.shape
    t = torch.arange(1)
    h = torch.arange(H)
    w = torch.arange(W)
    l = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    return latent_ids.unsqueeze(0).expand(B, -1, -1)


def prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    """Generate 4D position IDs for text embeddings (B, L, D) -> (B, L, 4)."""
    B, L, _ = x.shape
    out_ids = []
    for _ in range(B):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids)


# ── BN normalization / denormalization ──────────────────────────────────────


def bn_normalize(latents: torch.Tensor, vae) -> torch.Tensor:
    """Apply BN normalization to patchified latents."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    return (latents - mean) / std


def bn_denormalize(latents: torch.Tensor, vae) -> torch.Tensor:
    """Undo BN normalization on patchified latents."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    return latents * std + mean


# ── VAE encode ──────────────────────────────────────────────────────────────


def encode_image(image: torch.Tensor, vae) -> torch.Tensor:
    """Encode pixel image to patchified, BN-normalized latent.

    Args:
        image: (B, 3, H, W) float tensor in [0, 1]
        vae: FLUX VAE with BN layer

    Returns:
        (B, C*4, H//16, W//16) patchified, BN-normalized latent
    """
    # Scale to [-1, 1]
    x = 2.0 * image - 1.0
    with torch.no_grad():
        latents = vae.encode(x).latent_dist.sample()  # (B, C, H//8, W//8)
    latents = patchify_latents(latents)   # (B, C*4, H//16, W//16)
    latents = bn_normalize(latents, vae)  # BN normalize
    return latents


# ── VAE decode ──────────────────────────────────────────────────────────────


def decode_latents(packed_latents: torch.Tensor, latent_ids: torch.Tensor, vae) -> torch.Tensor:
    """
    Decode packed latents to pixel images.
    packed_latents: (B, seq, C) from transformer output
    latent_ids: (B, seq, 4) position IDs
    Returns: (B, 3, H, W) float tensor in [0, 1]
    """
    # Unpack to spatial
    latents = unpack_latents_with_ids(packed_latents, latent_ids)  # (B, C, H, W) patchified
    # BN denormalize
    latents = bn_denormalize(latents, vae)
    # Unpatchify
    latents = unpatchify_latents(latents)  # (B, 32, H*2, W*2)
    # VAE decode
    images = vae.decode(latents, return_dict=False)[0]  # (B, 3, H_full, W_full)
    # Clamp to [0, 1]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


# ── Text embedding precomputation ──────────────────────────────────────────


def precompute_text_embeddings(
    prompts_path: str,
    output_dir: str,
    model_id: str = "black-forest-labs/FLUX.2-klein-4B-Base",
    max_sequence_length: int = 512,
    hidden_states_layers: tuple[int, ...] = (9, 18, 27),
    batch_size: int = 4,
    device: str = "cuda",
):
    """
    Precompute and save text embeddings from the Qwen3 text encoder.
    This avoids loading the 8B text encoder during training.
    """
    from diffusers import Flux2KleinPipeline

    os.makedirs(output_dir, exist_ok=True)

    # Load just the text encoder parts
    print(f"Loading pipeline for text encoding: {model_id}")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer

    # Load prompts
    records = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Encoding {len(records)} prompts...")
    text_encoder.eval()

    for i in tqdm(range(0, len(records), batch_size), desc="Text encoding"):
        batch = records[i : i + batch_size]
        prompts = [r["prompt"] for r in batch]

        with torch.no_grad():
            prompt_embeds = pipe._get_qwen3_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompts,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
            )

        for j, record in enumerate(batch):
            idx = i + j
            embed = prompt_embeds[j].cpu()
            save_path = os.path.join(output_dir, f"{idx:06d}.pt")
            torch.save(
                {
                    "prompt_embeds": embed,
                    "target_text": record.get("target_text", ""),
                    "prompt": record["prompt"],
                },
                save_path,
            )

    # Free text encoder
    del text_encoder, pipe
    torch.cuda.empty_cache()

    print(f"Saved {len(records)} embeddings to {output_dir}")
    return len(records)
