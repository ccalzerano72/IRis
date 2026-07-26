#!/usr/bin/env bash
# Restormer Inference for Evaluation
# Runs Restormer restoration on test images for comparison with Marigold/DiffBIR/DFPIR
#
# Restormer tasks available for denoising:
#   - Real_Denoising: trained on SIDD real-world noise
#   - Gaussian_Color_Denoising: blind gaussian color denoising
#
# Note: Restormer has NO configurable inference parameters beyond tile size.
#       It is a single forward pass model (no iterative denoising steps).

set -e
set -x

# Configuration
RESTORMER_VENV="${RESTORMER_VENV:-external/Restormer/.venv}"
RESTORMER_DIR="${RESTORMER_DIR:-external/Restormer}"
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"
subfolder=${1:-"eval"}
input_dir=${2:-"${BASE_DATA_DIR}/restoration_test/degraded"}
clean_dir=${3:-"${BASE_DATA_DIR}/restoration_test/clean"}
max_images=${4:-""}  # Max images to process per run (empty = all)
task=${5:-"Real_Denoising"}  # Real_Denoising, Gaussian_Color_Denoising
gpu=${6:-0}
tile=${7:-256}

# Validate task
if [ "${task}" != "Real_Denoising" ] && [ "${task}" != "Gaussian_Color_Denoising" ]; then
    echo "ERROR: Invalid task '${task}'. Must be 'Real_Denoising' or 'Gaussian_Color_Denoising'"
    exit 1
fi

# Build model name from task (lowercase with underscores)
TASK_LOWER=$(echo "${task}" | tr '[:upper:]' '[:lower:]')
MODEL_NAME="restormer_${TASK_LOWER}"

# Get absolute paths
CURRENT_DIR=$(pwd)
INPUT_ABS_DIR=$(realpath "${input_dir}")
CLEAN_ABS_DIR=$(realpath "${clean_dir}")
RESTORMER_ABS_DIR=$(realpath "${RESTORMER_DIR}")

# Create output directory
# Restormer saves to {result_dir}/{task}/, so we set result_dir such that
# the final images end up in our desired restored/ directory
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

# Check if Restormer venv exists
if [ ! -d "${RESTORMER_VENV}" ]; then
    echo "ERROR: Restormer virtual environment not found at ${RESTORMER_VENV}"
    echo "Please create venv in external/Restormer/.venv or set RESTORMER_VENV environment variable"
    exit 1
fi

# Check if Restormer directory exists
if [ ! -d "${RESTORMER_DIR}" ]; then
    echo "ERROR: Restormer directory not found at ${RESTORMER_DIR}"
    exit 1
fi

# Prepare input directory: always create a temp dir with image-only symlinks
# to avoid non-image files (e.g. degradation_metadata) crashing Restormer.
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

# Restormer's demo.py saves output to {result_dir}/{task}/.
# We use a temporary result_dir so that images land in {tmp}/{task}/,
# then move them to our desired output directory.
RESTORMER_RESULT_DIR=$(mktemp -d)

# Check which images are already processed in our output dir
EXISTING_COUNT=0
TOTAL_COUNT=0
for img in "${EFFECTIVE_INPUT_DIR}"/*.png "${EFFECTIVE_INPUT_DIR}"/*.jpg "${EFFECTIVE_INPUT_DIR}"/*.jpeg "${EFFECTIVE_INPUT_DIR}"/*.PNG "${EFFECTIVE_INPUT_DIR}"/*.JPG; do
    [ -e "$img" ] || continue
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    basename_img=$(basename "$img")
    name_no_ext="${basename_img%.*}"
    # Restormer always saves as .png
    restored_img="${OUTPUT_ABS_DIR}/${name_no_ext}.png"

    if [ -f "${restored_img}" ]; then
        EXISTING_COUNT=$((EXISTING_COUNT + 1))
    fi
done

echo "Found ${EXISTING_COUNT}/${TOTAL_COUNT} images already processed in ${OUTPUT_ABS_DIR}"

if [ ${EXISTING_COUNT} -eq ${TOTAL_COUNT} ] && [ ${TOTAL_COUNT} -gt 0 ]; then
    echo "All images already processed. Skipping inference."
else
    # Activate Restormer venv and run inference
    source "${RESTORMER_VENV}/bin/activate"

    echo "Running Restormer inference..."
    echo "  Task: ${task}"
    echo "  Input: ${EFFECTIVE_INPUT_DIR}"
    echo "  Tile size: ${tile}"
    echo "  GPU: ${gpu}"

    # Run demo.py from the Restormer directory (required for relative weight paths
    # and run_path('basicsr/models/archs/restormer_arch.py'))
    pushd "${RESTORMER_ABS_DIR}" > /dev/null
    CUDA_VISIBLE_DEVICES=${gpu} python demo.py \
        --task "${task}" \
        --input_dir "${EFFECTIVE_INPUT_DIR}" \
        --result_dir "${RESTORMER_RESULT_DIR}" \
        --tile ${tile} \
        --tile_overlap 32
    popd > /dev/null

    deactivate

    # Move restored images from Restormer's output subdirectory to our output dir
    # Restormer saves to {result_dir}/{task}/
    RESTORMER_OUTPUT="${RESTORMER_RESULT_DIR}/${task}"
    if [ -d "${RESTORMER_OUTPUT}" ]; then
        echo "Moving restored images to ${OUTPUT_ABS_DIR}..."
        for img in "${RESTORMER_OUTPUT}"/*.png; do
            [ -e "$img" ] || continue
            mv "$img" "${OUTPUT_ABS_DIR}/"
        done
    else
        echo "ERROR: Restormer output directory not found at ${RESTORMER_OUTPUT}"
        echo "Contents of result dir:"
        ls -la "${RESTORMER_RESULT_DIR}/"
        exit 1
    fi

    echo "Restormer inference complete. Results saved to: ${OUTPUT_ABS_DIR}"
fi

# Cleanup temp directories
if [ -d "${TEMP_INPUT_DIR}" ]; then
    rm -rf "${TEMP_INPUT_DIR}"
fi
if [ -d "${RESTORMER_RESULT_DIR}" ]; then
    rm -rf "${RESTORMER_RESULT_DIR}"
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
