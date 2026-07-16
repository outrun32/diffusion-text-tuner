#!/usr/bin/env bash
# Validate and merge canonical score shards.

set -euo pipefail

OUTPUT_DIR="${1:-outputs/generated}"
EXPECTED_SHARDS="${2:?usage: merge_scores.sh OUTPUT_DIR EXPECTED_SHARDS [--delete]}"
DELETE_FLAG="${3:-}"

args=(
    --input-dir "$OUTPUT_DIR"
    --output "$OUTPUT_DIR/scores.csv"
    --expected-shards "$EXPECTED_SHARDS"
)
if [[ "$DELETE_FLAG" == "--delete" ]]; then
    args+=(--delete)
elif [[ -n "$DELETE_FLAG" ]]; then
    echo "unknown option: $DELETE_FLAG" >&2
    exit 2
fi

python -m scripts.merge_score_shards "${args[@]}"
