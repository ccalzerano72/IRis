#!/bin/bash
# Quick restoration script — auto-detects checkpoint type
# Usage:
#   ./restore.sh <image_or_directory> [--steps N] [--ensemble N] [--checkpoint PATH]
#
# Examples:
#   ./restore.sh photo.png
#   ./restore.sh ./my_images/
#   ./restore.sh photo.png --steps 5 --ensemble 10
#   ./restore.sh photo.png --checkpoint checkpoints/009_re_015000/latest

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Use cached HuggingFace models (avoid network auth issues with gated repos)
export HF_HUB_OFFLINE=1

# --- Mandatory argument: image or directory ---
INPUT="$1"
if [ -z "$INPUT" ]; then
    echo "Usage: ./restore.sh <image_or_directory> [--steps N] [--ensemble N] [--checkpoint PATH]"
    exit 1
fi
shift

# --- Defaults ---
STEPS=10
ENSEMBLE=1
CHECKPOINT="checkpoints/002_re_015000/latest"

# --- Parse optional arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps) STEPS="$2"; shift 2 ;;
        --ensemble) ENSEMBLE="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Resolve input (single file vs directory) ---
INPUT="$(realpath "$INPUT")"

if [ -f "$INPUT" ]; then
    # Single image: create temp dir with symlink, output next to original
    TMPDIR=$(mktemp -d)
    ln -s "$INPUT" "$TMPDIR/$(basename "$INPUT")"
    INPUT_DIR="$TMPDIR"
    OUTPUT_DIR="$(dirname "$INPUT")/restored"
    CLEANUP_TMP=1
elif [ -d "$INPUT" ]; then
    INPUT_DIR="$INPUT"
    OUTPUT_DIR="$INPUT/restored"
    CLEANUP_TMP=0
else
    echo "Error: '$INPUT' is not a valid file or directory."
    exit 1
fi

# --- Resolve checkpoint path ---
cd "$SCRIPT_DIR"
CKPT_ABS_PATH="$(realpath "$CHECKPOINT")"

# --- Auto-detect checkpoint type ---
if [ -f "$CKPT_ABS_PATH/hybrid_003_config.json" ]; then
    CKPT_TYPE="hybrid_003"
elif [ -f "$CKPT_ABS_PATH/hybrid_config.json" ]; then
    CKPT_TYPE="hybrid_002"
elif [ -d "$CKPT_ABS_PATH/controlnet" ] && [ ! -d "$CKPT_ABS_PATH/unet" ]; then
    CKPT_TYPE="controlnet"
elif [ -d "$CKPT_ABS_PATH/unet" ]; then
    CKPT_TYPE="marigold"
else
    echo "Error: Cannot detect checkpoint type at $CKPT_ABS_PATH"
    exit 1
fi

echo "Checkpoint type: $CKPT_TYPE"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Steps: $STEPS | Ensemble: $ENSEMBLE"

# --- Run restoration with correct script ---
if [ "$CKPT_TYPE" = "hybrid_003" ]; then
    python script/hybrid_controlnet_restoration/run.py \
        --checkpoint "$CKPT_ABS_PATH" \
        --input_rgb_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --denoise_steps "$STEPS" \
        --ensemble_size "$ENSEMBLE" \
        --processing_res 0 \
        --half_precision

elif [ "$CKPT_TYPE" = "hybrid_002" ]; then
    BASE_CKPT=$(python -c "import json; print(json.load(open('$CKPT_ABS_PATH/hybrid_config.json'))['base_checkpoint_path'])")
    BASE_CKPT_ABS=$(realpath "$BASE_CKPT")
    python script/hybrid_controlnet_restoration/run.py \
        --base_checkpoint "$BASE_CKPT_ABS" \
        --controlnet_checkpoint "$CKPT_ABS_PATH" \
        --input_rgb_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --denoise_steps "$STEPS" \
        --ensemble_size "$ENSEMBLE" \
        --processing_res 0 \
        --half_precision

elif [ "$CKPT_TYPE" = "controlnet" ]; then
    python script/controlnet_restoration/run.py \
        --checkpoint "$CKPT_ABS_PATH" \
        --input_rgb_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --denoise_steps "$STEPS" \
        --ensemble_size "$ENSEMBLE" \
        --processing_res 0 \
        --half_precision

elif [ "$CKPT_TYPE" = "marigold" ]; then
    python script/restoration/run.py \
        --checkpoint "$CKPT_ABS_PATH" \
        --input_rgb_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --denoise_steps "$STEPS" \
        --ensemble_size "$ENSEMBLE" \
        --processing_res 0 \
        --half_precision
fi

# --- Cleanup temp dir if used ---
if [ "$CLEANUP_TMP" -eq 1 ]; then
    rm -rf "$TMPDIR"
fi

echo ""
echo "Done. Output in: $OUTPUT_DIR"
