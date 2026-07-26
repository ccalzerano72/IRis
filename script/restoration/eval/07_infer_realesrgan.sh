#!/usr/bin/env bash
# Real-ESRGAN Inference for Evaluation
# Runs Real-ESRGAN (x4plus model) restoration on test images for comparison
# with Marigold/DiffBIR/DFPIR/Restormer
#
# Real-ESRGAN is a GAN-based blind super-resolution model. When used with
# --outscale 1, it performs same-resolution blind restoration (denoising,
# deartifacting) without upscaling.
#
# Note: Real-ESRGAN is a single forward pass model (no iterative denoising).
#       The only configurable parameters are tile size and outscale.

set -e
set -x

# Configuration
REALESRGAN_VENV="${REALESRGAN_VENV:-external/Real-ESRGAN/.venv}"
REALESRGAN_DIR="${REALESRGAN_DIR:-external/Real-ESRGAN}"
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"
subfolder=${1:-"eval"}
input_dir=${2:-"${BASE_DATA_DIR}/restoration_test/degraded"}
clean_dir=${3:-"${BASE_DATA_DIR}/restoration_test/clean"}
max_images=${4:-""}  # Max images to process per run (empty = all)
outscale=${5:-1}     # Output scale: 1 = same resolution, 4 = 4x upscale
gpu=${6:-0}
tile=${7:-256}

# Model name includes outscale to distinguish output folders and metrics
MODEL_NAME="realesrgan_x4plus_s${outscale}"

# Get absolute paths
CURRENT_DIR=$(pwd)
INPUT_ABS_DIR=$(realpath "${input_dir}")
CLEAN_ABS_DIR=$(realpath "${clean_dir}")
REALESRGAN_ABS_DIR=$(realpath "${REALESRGAN_DIR}")

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

# Check if Real-ESRGAN venv exists
if [ ! -d "${REALESRGAN_VENV}" ]; then
    echo "ERROR: Real-ESRGAN virtual environment not found at ${REALESRGAN_VENV}"
    echo "Please create venv in external/Real-ESRGAN/.venv or set REALESRGAN_VENV environment variable"
    exit 1
fi

# Check if Real-ESRGAN directory exists
if [ ! -d "${REALESRGAN_DIR}" ]; then
    echo "ERROR: Real-ESRGAN directory not found at ${REALESRGAN_DIR}"
    exit 1
fi

# Prepare input directory: always create a temp dir with image-only symlinks
# to avoid non-image files (e.g. degradation_metadata) crashing Real-ESRGAN.
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
    # Real-ESRGAN with --suffix "" and --ext png saves as {name}.png
    restored_img="${OUTPUT_ABS_DIR}/${name_no_ext}.png"

    if [ -f "${restored_img}" ]; then
        EXISTING_COUNT=$((EXISTING_COUNT + 1))
    fi
done

echo "Found ${EXISTING_COUNT}/${TOTAL_COUNT} images already processed in ${OUTPUT_ABS_DIR}"

if [ ${EXISTING_COUNT} -eq ${TOTAL_COUNT} ] && [ ${TOTAL_COUNT} -gt 0 ]; then
    echo "All images already processed. Skipping inference."
else
    # Activate Real-ESRGAN venv and run inference
    source "${REALESRGAN_VENV}/bin/activate"

    echo "Running Real-ESRGAN inference..."
    echo "  Model: RealESRGAN_x4plus"
    echo "  Input: ${EFFECTIVE_INPUT_DIR}"
    echo "  Output scale: ${outscale}"
    echo "  Tile size: ${tile}"
    echo "  GPU: ${gpu}"

    # Run inference_realesrgan.py from the Real-ESRGAN directory
    # (required for auto-downloading weights to weights/ subfolder)
    pushd "${REALESRGAN_ABS_DIR}" > /dev/null
    CUDA_VISIBLE_DEVICES=${gpu} python inference_realesrgan.py \
        -n RealESRGAN_x4plus \
        -i "${EFFECTIVE_INPUT_DIR}" \
        -o "${OUTPUT_ABS_DIR}" \
        --outscale ${outscale} \
        --suffix "" \
        --ext png \
        --tile ${tile}
    popd > /dev/null

    deactivate

    echo "Real-ESRGAN inference complete. Results saved to: ${OUTPUT_ABS_DIR}"
fi

# Cleanup temp directory
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
