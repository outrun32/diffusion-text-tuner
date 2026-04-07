#!/bin/bash
# Merge per-shard score CSVs into a single scores.csv.
#
# Usage:
#   bash scripts/cluster/merge_scores.sh [output_dir]
#
# Default output_dir: outputs/generated

set -euo pipefail

OUTPUT_DIR="${1:-outputs/generated}"
MERGED="$OUTPUT_DIR/scores.csv"

echo "Merging shard CSVs from $OUTPUT_DIR..."

# Find all shard files
SHARDS=("$OUTPUT_DIR"/scores_shard*.csv)
if [ ${#SHARDS[@]} -eq 0 ]; then
    echo "No shard files found in $OUTPUT_DIR"
    exit 1
fi

echo "Found ${#SHARDS[@]} shard files"

# Write header from first shard, then data from all shards
head -1 "${SHARDS[0]}" > "$MERGED"
for shard in "${SHARDS[@]}"; do
    tail -n +2 "$shard" >> "$MERGED"
done

TOTAL=$(( $(wc -l < "$MERGED") - 1 ))
echo "Merged $TOTAL scores → $MERGED"

# Optional: remove shard files
read -p "Delete shard files? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -v "${SHARDS[@]}"
fi
