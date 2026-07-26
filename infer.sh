#!/bin/bash
# IRis — Quick inference wrapper
# Usage: ./infer.sh <input_dir> <output_dir>
#
# Runs the IRis restoration pipeline with recommended defaults:
#   checkpoint : checkpoints/002_re_015000
#   denoise_steps : 5
#   ensemble_size : 1
#   fp16 : enabled
#
# All other parameters use run.py defaults.
# To override any parameter, call run.py directly.

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    echo "  input_dir  : directory containing degraded images (jpg/jpeg/png)"
    echo "  output_dir : directory where restored images will be saved"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CHECKPOINT="checkpoints/002_re_015000"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/script/hybrid_controlnet_restoration/run.py" \
    --checkpoint "$CHECKPOINT" \
    --input_rgb_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --denoise_steps 5 \
    --fp16
