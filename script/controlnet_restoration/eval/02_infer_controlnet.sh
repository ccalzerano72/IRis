#!/usr/bin/env bash
# ControlNet Restoration Inference for Evaluation
# Runs ControlNet restoration on test images for comparison
# Duplicated from script/restoration/eval/02_infer_marigold.sh
# Only change: calls script/controlnet_restoration/run.py and uses "controlnet_" prefix

set -e
set -x

# Configuration
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"
subfolder=${1:-"eval"}
input_dir=${2:-"${BASE_DATA_DIR}/restoration_test/degraded"}
clean_dir=${3:-"${BASE_DATA_DIR}/restoration_test/clean"}
processing_res=${4:-0}    # Processing resolution: 0=original, 512, 768, 1024
denoise_steps=${5:-10}
ensemble_size=${6:-1}
ckpt=${7:-"checkpoints/controlnet-restoration-latest"}
scheduler=${8:-"ddim"}    # Scheduler type: ddim, lcm
guidance_scale=${9:-1.0}  # CFG scale: 1.0=no CFG, >1.0=apply CFG
max_images=${10:-""}      # Max images to process per run (empty = all)

# Get absolute paths
CURRENT_DIR=$(pwd)
CKPT_ABS_PATH=$(realpath "${ckpt}")
INPUT_ABS_DIR=$(realpath "${input_dir}")
CLEAN_ABS_DIR=$(realpath "${clean_dir}")

# Extract checkpoint parent directory name for output folder
CKPT_PARENT_DIR=$(basename "$(dirname "${CKPT_ABS_PATH}")")

# Create output directory with checkpoint-specific naming
# Include guidance_scale in name only if CFG is used (scale > 1.0)
if (( $(echo "$guidance_scale > 1.0" | bc -l) )); then
    cfg_suffix="_cfg${guidance_scale}"
else
    cfg_suffix=""
fi
OUTPUT_DIR="output/${subfolder}/controlnet_${CKPT_PARENT_DIR}/prediction_${processing_res}_s$(printf "%02d" ${denoise_steps})_e$(printf "%02d" ${ensemble_size})_${scheduler}${cfg_suffix}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_ABS_DIR=$(realpath "${OUTPUT_DIR}")

# Create clean/degraded symlinks in the subfolder root for the compare tab.
# The webserver's get_image_path() looks for {results_folder}/clean/ and /degraded/.
SUBFOLDER_ROOT="output/${subfolder}"
if [ ! -e "${SUBFOLDER_ROOT}/clean" ]; then
    ln -s "${CLEAN_ABS_DIR}" "${SUBFOLDER_ROOT}/clean"
    echo "Created symlink: ${SUBFOLDER_ROOT}/clean -> ${CLEAN_ABS_DIR}"
fi
if [ ! -e "${SUBFOLDER_ROOT}/degraded" ]; then
    ln -s "${INPUT_ABS_DIR}" "${SUBFOLDER_ROOT}/degraded"
    echo "Created symlink: ${SUBFOLDER_ROOT}/degraded -> ${INPUT_ABS_DIR}"
fi

# Create metrics output directory
METRICS_DIR="output/${subfolder}/metrics"
mkdir -p "${METRICS_DIR}"
METRICS_ABS_DIR=$(realpath "${METRICS_DIR}")

# Check if metrics already exist for this configuration
OUTPUT_SUBDIR="controlnet_${CKPT_PARENT_DIR}_prediction_${processing_res}_s$(printf "%02d" ${denoise_steps})_e$(printf "%02d" ${ensemble_size})_${scheduler}${cfg_suffix}"
SUMMARY_FILE="${METRICS_ABS_DIR}/summary_${OUTPUT_SUBDIR}.txt"
CSV_FILE="${METRICS_ABS_DIR}/metrics_${OUTPUT_SUBDIR}.csv"

if [ -f "${SUMMARY_FILE}" ] && [ -f "${CSV_FILE}" ]; then
    echo "Metrics already exist for this configuration. Skipping entire run."
    echo "Summary: ${SUMMARY_FILE}"
    echo "CSV: ${CSV_FILE}"
    exit 0
fi

# Check if venv exists
if [ ! -d "${MARIGOLD_VENV}" ]; then
    echo "ERROR: Virtual environment not found at ${MARIGOLD_VENV}"
    echo "Please create venv or set MARIGOLD_VENV environment variable"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "${CKPT_ABS_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CKPT_ABS_PATH}"
    exit 1
fi

# Check if clean directory exists
if [ ! -d "${CLEAN_ABS_DIR}" ]; then
    echo "ERROR: Clean images directory not found at ${CLEAN_ABS_DIR}"
    exit 1
fi

# Activate venv and run inference
source "${MARIGOLD_VENV}/bin/activate"

echo "Running ControlNet inference..."

# Check which images are already processed
RESTORED_DIR="${OUTPUT_ABS_DIR}/restored"
mkdir -p "${RESTORED_DIR}"

# Count existing and total images
EXISTING_COUNT=0
TOTAL_COUNT=0
for img in "${INPUT_ABS_DIR}"/*.png "${INPUT_ABS_DIR}"/*.jpg "${INPUT_ABS_DIR}"/*.jpeg; do
    [ -e "$img" ] || continue
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    basename_img=$(basename "$img")
    filename="${basename_img%.*}"
    restored_img="${RESTORED_DIR}/${filename}_restored.png"
    
    if [ -f "${restored_img}" ]; then
        EXISTING_COUNT=$((EXISTING_COUNT + 1))
    fi
done

echo "Found ${EXISTING_COUNT}/${TOTAL_COUNT} images already processed"

if [ ${EXISTING_COUNT} -eq ${TOTAL_COUNT} ] && [ ${TOTAL_COUNT} -gt 0 ]; then
    echo "All images already processed. Skipping inference."
else
    # Build max_images flag if set
    MAX_IMAGES_FLAG=""
    if [ -n "${max_images}" ] && [ "${max_images}" -gt 0 ] 2>/dev/null; then
        MAX_IMAGES_FLAG="--max_images ${max_images}"
    fi

    # Run inference using ControlNet run.py
    python script/controlnet_restoration/run.py \
        --checkpoint "${CKPT_ABS_PATH}" \
        --seed 1234 \
        --input_rgb_dir "${INPUT_ABS_DIR}" \
        --processing_res ${processing_res} \
        --denoise_steps ${denoise_steps} \
        --ensemble_size ${ensemble_size} \
        --scheduler ${scheduler} \
        --guidance_scale ${guidance_scale} \
        --half_precision \
        --output_dir "${OUTPUT_ABS_DIR}" \
        ${MAX_IMAGES_FLAG}

    echo "ControlNet inference complete. Results saved to: ${OUTPUT_ABS_DIR}"
fi

# Calculate metrics only if we don't have them yet
if [ ! -f "${SUMMARY_FILE}" ] || [ ! -f "${CSV_FILE}" ]; then
    echo "Calculating metrics..."
    python script/restoration/eval/04_calculate_metrics.py \
        --clean_dir "${CLEAN_ABS_DIR}" \
        --restored_dir "${OUTPUT_ABS_DIR}/restored" \
        --output_dir "${METRICS_ABS_DIR}" \
        --model_name "${OUTPUT_SUBDIR}"
else
    echo "Metrics already exist. Skipping calculation."
fi

deactivate

echo "Evaluation complete!"
echo "Inference results: ${OUTPUT_ABS_DIR}"
echo "Metrics results: ${METRICS_ABS_DIR}"
