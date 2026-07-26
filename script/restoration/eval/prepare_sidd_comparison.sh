#!/bin/bash
# Organize SIDD Small sRGB dataset into comparison-compatible structure.
# Creates symlinks (no file duplication) under comparison_realsr/datasets/SIDD/.
#
# Source: datasets/SIDD_Small_sRGB_Only/Data/<scene_dir>/{GT_SRGB_*.PNG, NOISY_SRGB_*.PNG}
# Target: comparison_realsr/datasets/SIDD/{clean,degraded_1x}/<scene_instance_number>.PNG

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SRC_DIR="$PROJECT_ROOT/datasets/SIDD_Small_sRGB_Only/Data"
DST_DIR="$PROJECT_ROOT/comparison_realsr/datasets/SIDD"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Source directory not found: $SRC_DIR"
    exit 1
fi

echo "Source:      $SRC_DIR"
echo "Destination: $DST_DIR"
echo ""

clean_dir="$DST_DIR/clean"
degraded_dir="$DST_DIR/degraded_1x"

mkdir -p "$clean_dir" "$degraded_dir"

count=0

for scene_dir in "$SRC_DIR"/*/; do
    dirname=$(basename "$scene_dir")
    # Extract scene instance number (first 4 digits)
    instance="${dirname%%_*}"

    gt_file=$(find "$scene_dir" -maxdepth 1 -name "GT_SRGB_*.PNG" -type f | head -1)
    noisy_file=$(find "$scene_dir" -maxdepth 1 -name "NOISY_SRGB_*.PNG" -type f | head -1)

    if [ -z "$gt_file" ] || [ -z "$noisy_file" ]; then
        echo "WARNING: Missing GT or NOISY in $dirname, skipping"
        continue
    fi

    ln -sf "$gt_file" "$clean_dir/${instance}.PNG"
    ln -sf "$noisy_file" "$degraded_dir/${instance}.PNG"
    count=$((count + 1))
done

echo "Done. $count pairs linked under: $DST_DIR"
