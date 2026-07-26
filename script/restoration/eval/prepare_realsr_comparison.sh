#!/bin/bash
# Reorganize RealSR-TEST dataset into comparison-compatible structure.
# Creates symlinks (no file duplication) under comparison/RealSR/.
#
# Source: datasets/RealSR-TEST/{Canon,Nikon}/{2,3,4}/{Camera}_{NNN}_{HR,LR{scale}}.png
# Target: comparison/RealSR/RealSR-{Camera}-{scale}/{clean,degraded_1x}/{Camera}_{NNN}.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SRC_DIR="$PROJECT_ROOT/datasets/RealSR-TEST"
DST_DIR="$PROJECT_ROOT/comparison/RealSR"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Source directory not found: $SRC_DIR"
    exit 1
fi

echo "Source:      $SRC_DIR"
echo "Destination: $DST_DIR"
echo ""

for camera in Canon Nikon; do
    for scale in 2 3 4; do
        subset="RealSR-${camera}-${scale}"
        clean_dir="$DST_DIR/$subset/clean"
        degraded_dir="$DST_DIR/$subset/degraded_1x"

        mkdir -p "$clean_dir" "$degraded_dir"

        src_folder="$SRC_DIR/$camera/$scale"
        count=0

        for hr_file in "$src_folder"/${camera}_*_HR.png; do
            basename=$(basename "$hr_file")
            # Canon_001_HR.png -> Canon_001.png
            name="${basename/_HR/}"

            lr_file="$src_folder/${basename/_HR.png/_LR${scale}.png}"

            if [ ! -f "$lr_file" ]; then
                echo "WARNING: Missing LR file: $lr_file"
                continue
            fi

            ln -sf "$hr_file" "$clean_dir/$name"
            ln -sf "$lr_file" "$degraded_dir/$name"
            count=$((count + 1))
        done

        echo "$subset: $count pairs linked"
    done
done

echo ""
echo "Done. Structure created under: $DST_DIR"
