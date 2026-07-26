#!/usr/bin/env bash
# StableSR Inference for Evaluation
# Runs StableSR (768v model) super-resolution on test images for comparison
# with Marigold/DiffBIR/DFPIR/Restormer/Real-ESRGAN.
#
# StableSR is a diffusion-based blind super-resolution model that uses a
# pretrained Stable Diffusion v2 backbone with a VQGAN decoder.
#
# Uses the NON-TILING canvas script (sr_val_ddim_text_T_negativeprompt_canvas.py)
# with --n_samples 1 to process images one at a time. This avoids:
#   1. Tiling artifacts on small images (visible grid pattern)
#   2. Mixed-resolution batching crash (tensor size mismatch)
#
# The canvas script internally upscales small images so the short side is at
# least input_size (768), processes at that resolution, then the output is
# at upscale * original_resolution.
#
# Configurable parameters:
#   ddim_steps  - Number of DDIM sampling steps (default: 20, paper: 200)
#   dec_w       - VQGAN decoder weight (0.0 = pure diffusion, 0.5 = balanced)
#   upscale     - Scale factor (default: 4, standard SR setting)
#   colorfix    - Color correction: wavelet (recommended), adain, nofix

set -e
set -x

# Configuration
STABLESR_VENV="${STABLESR_VENV:-external/StableSR/.venv}"
STABLESR_DIR="${STABLESR_DIR:-external/StableSR}"
TAMING_DIR="${TAMING_DIR:-external/taming-transformers}"
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"
subfolder=${1:-"eval"}
input_dir=${2:-"${BASE_DATA_DIR}/restoration_test/degraded"}
clean_dir=${3:-"${BASE_DATA_DIR}/restoration_test/clean"}
max_images=${4:-""}       # Max images to process per run (empty = all)
ddim_steps=${5:-20}       # DDIM sampling steps (paper: 200, practical: 20-50)
dec_w=${6:-0.0}           # VQGAN decoder weight (0.0 = pure diffusion)
upscale=${7:-4}           # Upscale factor (4 = standard SR setting)
colorfix=${8:-"wavelet"}  # Color correction: wavelet, adain, nofix
gpu=${9:-0}

# Build model name from parameters
# Format: stablesr_s{steps}_w{dec_w}
MODEL_NAME="stablesr_s${ddim_steps}_w${dec_w}"

# Get absolute paths
CURRENT_DIR=$(pwd)
INPUT_ABS_DIR=$(realpath "${input_dir}")
CLEAN_ABS_DIR=$(realpath "${clean_dir}")
STABLESR_ABS_DIR=$(realpath "${STABLESR_DIR}")
TAMING_ABS_DIR=$(realpath "${TAMING_DIR}")

# Create output directory
OUTPUT_DIR="output/${subfolder}/${MODEL_NAME}/restored"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_ABS_DIR=$(realpath "${OUTPUT_DIR}")

# Create metrics output directory
METRICS_DIR="output/${subfolder}/metrics"
mkdir -p "${METRICS_DIR}"
METRICS_ABS_DIR=$(realpath "${METRICS_DIR}")

# Check if metrics already exist for this configuration
SUMMARY_FILE="${METRICS_ABS_DIR}/summary_${MODEL_NAME}.txt"
CSV_FILE="${METRICS_ABS_DIR}/metrics_${MODEL_NAME}.csv"

if [ -f "${SUMMARY_FILE}" ] && [ -f "${CSV_FILE}" ]; then
    echo "Metrics already exist for ${MODEL_NAME}. Skipping entire run."
    echo "Summary: ${SUMMARY_FILE}"
    echo "CSV: ${CSV_FILE}"
    exit 0
fi

# Check if input directory exists
if [ ! -d "${INPUT_ABS_DIR}" ]; then
    echo "ERROR: Input directory not found at ${INPUT_ABS_DIR}"
    exit 1
fi

# Check if clean directory exists
if [ ! -d "${CLEAN_ABS_DIR}" ]; then
    echo "ERROR: Clean images directory not found at ${CLEAN_ABS_DIR}"
    exit 1
fi

# Check if StableSR venv exists
if [ ! -d "${STABLESR_VENV}" ]; then
    echo "ERROR: StableSR virtual environment not found at ${STABLESR_VENV}"
    echo "Please create venv in external/StableSR/.venv or set STABLESR_VENV environment variable"
    exit 1
fi

# Check if StableSR directory exists
if [ ! -d "${STABLESR_DIR}" ]; then
    echo "ERROR: StableSR directory not found at ${STABLESR_DIR}"
    exit 1
fi

# Check if taming-transformers directory exists
if [ ! -d "${TAMING_DIR}" ]; then
    echo "ERROR: taming-transformers directory not found at ${TAMING_DIR}"
    echo "Please clone taming-transformers or set TAMING_DIR environment variable"
    exit 1
fi

# Check if StableSR weights exist
if [ ! -f "${STABLESR_ABS_DIR}/weights/stablesr_768v_000139.ckpt" ]; then
    echo "ERROR: StableSR checkpoint not found at ${STABLESR_ABS_DIR}/weights/stablesr_768v_000139.ckpt"
    exit 1
fi
if [ ! -f "${STABLESR_ABS_DIR}/weights/vqgan_cfw_00011.ckpt" ]; then
    echo "ERROR: VQGAN checkpoint not found at ${STABLESR_ABS_DIR}/weights/vqgan_cfw_00011.ckpt"
    exit 1
fi

# Prepare input directory: create a temp dir with image-only symlinks
# to avoid non-image files (e.g. degradation_metadata.json) crashing StableSR.
# If max_images is set, also limit the number of files.
TEMP_INPUT_DIR=$(mktemp -d)
count=0
for img in "${INPUT_ABS_DIR}"/*.png "${INPUT_ABS_DIR}"/*.jpg "${INPUT_ABS_DIR}"/*.jpeg "${INPUT_ABS_DIR}"/*.PNG "${INPUT_ABS_DIR}"/*.JPG; do
    [ -e "$img" ] || continue
    if [ -n "${max_images}" ] && [ "${max_images}" -gt 0 ] 2>/dev/null; then
        if [ ${count} -ge ${max_images} ]; then
            break
        fi
    fi
    ln -s "$img" "${TEMP_INPUT_DIR}/$(basename "$img")"
    count=$((count + 1))
done
EFFECTIVE_INPUT_DIR="${TEMP_INPUT_DIR}"
if [ -n "${max_images}" ] && [ "${max_images}" -gt 0 ] 2>/dev/null; then
    echo "Selected ${count} images for processing (max_images=${max_images})"
else
    echo "Selected ${count} images for processing (image files only)"
fi

# Check which images are already processed in our output dir
EXISTING_COUNT=0
TOTAL_COUNT=0
for img in "${EFFECTIVE_INPUT_DIR}"/*.png "${EFFECTIVE_INPUT_DIR}"/*.jpg "${EFFECTIVE_INPUT_DIR}"/*.jpeg "${EFFECTIVE_INPUT_DIR}"/*.PNG "${EFFECTIVE_INPUT_DIR}"/*.JPG; do
    [ -e "$img" ] || continue
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    basename_img=$(basename "$img")
    name_no_ext="${basename_img%.*}"
    # Wrapper saves as {name}.png directly in output_dir
    restored_img="${OUTPUT_ABS_DIR}/${name_no_ext}.png"

    if [ -f "${restored_img}" ]; then
        EXISTING_COUNT=$((EXISTING_COUNT + 1))
    fi
done

echo "Found ${EXISTING_COUNT}/${TOTAL_COUNT} images already processed in ${OUTPUT_ABS_DIR}"

if [ ${EXISTING_COUNT} -eq ${TOTAL_COUNT} ] && [ ${TOTAL_COUNT} -gt 0 ]; then
    echo "All images already processed. Skipping inference."
else
    # Activate StableSR venv and run inference
    source "${STABLESR_VENV}/bin/activate"

    echo "Running StableSR inference..."
    echo "  Model: StableSR 768v"
    echo "  Input: ${EFFECTIVE_INPUT_DIR}"
    echo "  DDIM steps: ${ddim_steps}"
    echo "  Decoder weight: ${dec_w}"
    echo "  Upscale: ${upscale}"
    echo "  Color fix: ${colorfix}"
    echo "  GPU: ${gpu}"

    # Get absolute path to our wrapper script (run from Marigold root)
    WRAPPER_SCRIPT=$(realpath "script/restoration/eval/infer_stablesr.py")

    # Run our wrapper script from the StableSR directory
    # (required for relative config/weight paths and StableSR module imports).
    # The wrapper fixes the latent-tiling bug by padding images so that
    # both latent dimensions are >= tile_size before calling the canvas sampler.
    pushd "${STABLESR_ABS_DIR}" > /dev/null
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=".:${TAMING_ABS_DIR}" python \
        "${WRAPPER_SCRIPT}" \
        --input_dir "${EFFECTIVE_INPUT_DIR}" \
        --output_dir "${OUTPUT_ABS_DIR}" \
        --config configs/stableSRNew/v2-finetune_text_T_768v.yaml \
        --ckpt weights/stablesr_768v_000139.ckpt \
        --vqgan_ckpt weights/vqgan_cfw_00011.ckpt \
        --ddim_steps ${ddim_steps} \
        --ddim_eta 1.0 \
        --dec_w ${dec_w} \
        --colorfix_type ${colorfix} \
        --scale 7.0 \
        --upscale ${upscale} \
        --seed 42 \
        --input_size 768 \
        --tile_overlap 48 \
        --gpu ${gpu}
    popd > /dev/null

    deactivate

    echo "StableSR inference complete. Results saved to: ${OUTPUT_ABS_DIR}"
fi

# Cleanup temp input directory
if [ -d "${TEMP_INPUT_DIR}" ]; then
    rm -rf "${TEMP_INPUT_DIR}"
fi

# Calculate metrics only if we don't have them yet
if [ ! -f "${SUMMARY_FILE}" ] || [ ! -f "${CSV_FILE}" ]; then
    echo "Calculating metrics..."

    # Check if Marigold venv exists
    if [ ! -d "${MARIGOLD_VENV}" ]; then
        echo "ERROR: Marigold virtual environment not found at ${MARIGOLD_VENV}"
        echo "Please create venv or set MARIGOLD_VENV environment variable"
        exit 1
    fi

    source "${MARIGOLD_VENV}/bin/activate"

    python script/restoration/eval/04_calculate_metrics.py \
        --clean_dir "${CLEAN_ABS_DIR}" \
        --restored_dir "${OUTPUT_ABS_DIR}" \
        --output_dir "${METRICS_ABS_DIR}" \
        --model_name "${MODEL_NAME}"

    deactivate
else
    echo "Metrics already exist. Skipping calculation."
fi

echo "Evaluation complete!"
echo "Inference results: ${OUTPUT_ABS_DIR}"
echo "Metrics results: ${METRICS_ABS_DIR}"
