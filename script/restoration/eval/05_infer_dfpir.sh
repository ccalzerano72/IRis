#!/usr/bin/env bash
# DFPIR Inference for Evaluation
# Runs DFPIR restoration on test images for comparison with Marigold/DiffBIR

set -e
set -x

# Configuration
DFPIR_VENV="${DFPIR_VENV:-external/DFPIR/.venv}"
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"
subfolder=${1:-"eval"}
input_dir=${2:-"${BASE_DATA_DIR}/restoration_test/degraded"}
clean_dir=${3:-"${BASE_DATA_DIR}/restoration_test/clean"}
degradation=${4:-"general"}  # noise15, noise25, noise50, rain, haze, blur, lowlight, general
checkpoint=${5:-""}          # Empty = auto-detect
gpu=${6:-0}
max_images=${7:-""}          # Max images to process per run (empty = all)

# Get absolute paths
CURRENT_DIR=$(pwd)
INPUT_ABS_DIR=$(realpath "${input_dir}")
CLEAN_ABS_DIR=$(realpath "${clean_dir}")

# Create output directory
OUTPUT_DIR="output/${subfolder}/dfpir_${degradation}/restored"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_ABS_DIR=$(realpath "${OUTPUT_DIR}")

# Create metrics output directory
METRICS_DIR="output/${subfolder}/metrics"
mkdir -p "${METRICS_DIR}"
METRICS_ABS_DIR=$(realpath "${METRICS_DIR}")

# Check if metrics already exist for this configuration
MODEL_NAME="dfpir_${degradation}"
SUMMARY_FILE="${METRICS_ABS_DIR}/summary_${MODEL_NAME}.txt"
CSV_FILE="${METRICS_ABS_DIR}/metrics_${MODEL_NAME}.csv"

if [ -f "${SUMMARY_FILE}" ] && [ -f "${CSV_FILE}" ]; then
    echo "Metrics already exist for this configuration. Skipping entire run."
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

# Check if DFPIR venv exists
if [ ! -d "${DFPIR_VENV}" ]; then
    echo "ERROR: DFPIR virtual environment not found at ${DFPIR_VENV}"
    echo "Please create venv in external/DFPIR/.venv or set DFPIR_VENV environment variable"
    exit 1
fi

# Activate DFPIR venv and run inference
source "${DFPIR_VENV}/bin/activate"

echo "Running DFPIR inference..."
echo "  Input: ${INPUT_ABS_DIR}"
echo "  Output: ${OUTPUT_ABS_DIR}"
echo "  Degradation: ${degradation}"

# Build checkpoint argument
CKPT_ARG=""
if [ -n "${checkpoint}" ]; then
    CKPT_ARG="--checkpoint ${checkpoint}"
fi

# Build max_images flag if set
MAX_IMAGES_FLAG=""
if [ -n "${max_images}" ] && [ "${max_images}" -gt 0 ] 2>/dev/null; then
    MAX_IMAGES_FLAG="--max_images ${max_images}"
fi

# Check which images are already processed
EXISTING_COUNT=0
TOTAL_COUNT=0
for img in "${INPUT_ABS_DIR}"/*.png "${INPUT_ABS_DIR}"/*.jpg "${INPUT_ABS_DIR}"/*.jpeg; do
    [ -e "$img" ] || continue
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    basename_img=$(basename "$img")
    restored_img="${OUTPUT_ABS_DIR}/${basename_img}"
    
    if [ -f "${restored_img}" ]; then
        EXISTING_COUNT=$((EXISTING_COUNT + 1))
    fi
done

echo "Found ${EXISTING_COUNT}/${TOTAL_COUNT} images already processed"

if [ ${EXISTING_COUNT} -eq ${TOTAL_COUNT} ] && [ ${TOTAL_COUNT} -gt 0 ]; then
    echo "All images already processed. Skipping inference."
else
    # Run inference
    python script/restoration/eval/infer_dfpir.py \
        --input_dir "${INPUT_ABS_DIR}" \
        --output_dir "${OUTPUT_ABS_DIR}" \
        --degradation "${degradation}" \
        --gpu ${gpu} \
        ${CKPT_ARG} \
        ${MAX_IMAGES_FLAG}

    echo "DFPIR inference complete. Results saved to: ${OUTPUT_ABS_DIR}"
fi

# Deactivate DFPIR venv
deactivate

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
