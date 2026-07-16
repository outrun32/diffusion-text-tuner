# Contributing

Use Python 3.11 and the committed lockfile. `make check` also calls ShellCheck, so install `uv` and
`shellcheck` before running it (`brew install uv shellcheck` on macOS):

```bash
uv sync --frozen --group dev --extra lint --extra plotting --extra analysis
make check
```

On Apple Silicon, add `--extra mlx` and run `make mac-check`. Do not run or advertise FLUX training
on MLX/MPS; training remains a Linux/CUDA workflow.

Keep reusable code under `src/`, CLI parsing under `scripts/`, and default tests free of model
downloads or accelerator requirements. New metrics need a versioned formula, a sidecar, and a test
that distinguishes them from prior thesis metrics.

Never commit generated checkpoints, tensors, score dumps, chat exports, editor state, private
manifests, or credentials. Small reviewed fixtures and public evidence files must include provenance
and hashes.

Unless explicitly marked otherwise, an intentionally submitted contribution is licensed under the
repository's [Apache License 2.0](LICENSE).
