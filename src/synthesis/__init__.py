"""Import-safe synthesis pipeline seams."""

from src.synthesis.dataset_builder import (
    SynthesisBuildConfig,
    bake_latents_phase,
    build_dataset,
    collate_records,
    encode_text_phase,
    fan_out,
    render_phase,
    write_anyword_json,
    write_masked_index,
)

__all__ = [
    "SynthesisBuildConfig",
    "bake_latents_phase",
    "build_dataset",
    "collate_records",
    "encode_text_phase",
    "fan_out",
    "render_phase",
    "write_anyword_json",
    "write_masked_index",
]
