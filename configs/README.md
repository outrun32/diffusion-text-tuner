# Configuration index

Configuration files are source contracts, not proof that an experiment ran. A comparison-grade run
also needs its immutable config snapshot, input hashes, model revision, seed, outputs, and run
manifest.

## Compatibility entry points

- `configs/sft.json`, `configs/dpo.json`, and `configs/masked_sft.json` keep the established trainer
  commands runnable. They pin the current base-model commit but do not reconstruct missing historical
  run provenance.
- New research variants belong under `configs/experiments/`; see
  [`configs/experiments/README.md`](experiments/README.md).

## Prompt generation

- `configs/prompts/simple.json`: deterministic, no-LLM smoke/data-contract sample.
- `configs/prompts/full.json`: broad prompt distribution used by the public 15k quality report.
- `configs/prompts/curriculum.json`: staged script/content/style curriculum with provenance fields.

## Accelerate launch profiles

- `configs/accelerate/single_gpu.yaml`: one-GPU local or single-node launch.
- `configs/accelerate/multi_gpu.yaml`: four-process distributed launch used by the default SLURM
  SFT/DPO wrappers.
- `configs/accelerate/multi_gpu_8x.yaml`: explicit eight-process profile for a reviewed eight-GPU
  allocation; do not use it merely because a node exposes fewer or more devices.

FLUX generation and every trainer remain Linux/CUDA workflows. These files do not add an MLX/MPS
training backend.

## Historical and corrected experiment records

Under `configs/experiments/{sft,dpo}/`, files ending in `_final.json` preserve settings named in the
defense-era experiment layout. Missing raw rows, checkpoints, and run manifests mean they are
historical config records only.

The DPO files ending in `_fixed_safe.json` preserve corrected objective/pair settings after the DPO
sign and safety audit:

- `configs/experiments/dpo/dpo_vlm_fixed_safe.json`
- `configs/experiments/dpo/dpo_ocr_fixed_safe.json`
- `configs/experiments/dpo/dpo_product_fixed_safe.json`

Use them only for future pinned reruns with materialized preference pairs and fresh manifests. They
must not be presented as completed experimental results.
