# Diffusion Text Tuner — Project Plan

## Goal
Framework for fine-tuning diffusion models to render non-Latin text (starting with Russian/Cyrillic).
Diploma project. Intended to become an open-source toolkit.

## Base Model
- FLUX.2 Klein 4B Base (Apache 2.0, ~13GB VRAM, undistilled — best for LoRA)
- Text encoder: Qwen3 8B (already multilingual -> understands Russian prompts)
- Bottleneck is visual glyph rendering, not prompt understanding
- Alternative: FLUX.2 Klein 9B Base (better quality, but Non-Commercial License)

## Training: LoRA only
- ~20-50M trainable params (rank 16-64), ~100-200MB adapter file
- Full model stays frozen, only LoRA matrices updated

## Reward Stack
- Perceptual text: DINOv3-ViT-B (rendered ref comparison), differentiable
- Alignment: SigLIP2 So400m-patch14-384, differentiable
- Aesthetic: HPSv2.1 (CLIP-ViT-H-14 + MLP), differentiable
- Classical OCR: PaddleOCR cyrillic_PP-OCRv3_mobile_rec, non-differentiable

## OCR findings summary
- PaddleOCR v3: honest char-level reading, good classical reward signal
- PaddleOCR v5: language-model correction bias, not suitable as primary reward
- TrOCR Cyrillic: hallucinates Church Slavonic on printed text, rejected
- Qwen3.5-4B yes-prob: best discrimination (0.071 broken vs 0.922 correct; delta 0.85)

## Two-Stage Training
1) Stage 1 ReFL
- Rewards: SigLIP2 + HPSv2.1 + DINOv3 perceptual text
- Purpose: scene fidelity + aesthetics + visual text shape

2) Stage 2 Flow-GRPO
- Rewards include OCR and non-differentiable signals
- Purpose: fine-tune text rendering accuracy

## Hardware
- Local dev: RTX 5090 (32GB)
- Full-scale: cHARISMa cluster (H100/A100)

## Dataset
- Synthetic prompts only
- 5-10k prompts curriculum (chars -> words -> phrases)
- Russian text distribution coverage

## Novelty
- Differentiable rendered-reference text reward
- VLM OCR bias identified + mitigation via yes/no probability
- Multilingual curriculum training
- ReFL -> GRPO mixed reward pipeline
