#!/usr/bin/env bash
# DiffBIR Local Inference for Evaluation
# Runs DiffBIR on test images for comparison with Marigold

set -e
set -x

# Ensure conda is available (needed when launched from non-interactive shells,
# e.g. subprocess.Popen in the web server batch runner)
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
fi

# Configuration
DIFFBIR_DIR="${DIFFBIR_DIR:-external/DiffBIR}"
DIFFBIR_CONDA_ENV="${DIFFBIR_CONDA_ENV:-diffbir}"
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"
subfolder=${1:-"eval"}
input_dir=${2:-"${BASE_DATA_DIR}/restoration_test/degraded"}
strength=${3:-1.0}    # Controllability: 0=creative/quality, 1=faithful/PSNR (default: 1.0)
steps=${4:-10}        # Sampling steps (default: 10)
upscale=${5:-1}       # Upscale factor (1 for same resolution comparison)
task=${6:-"denoise"}  # denoise, sr (super-resolution), face, unaligned_face
clean_dir=${7:-"${BASE_DATA_DIR}/restoration_test/clean"}
max_images=${8:-""}   # Max images to process per run (empty = all)

# Get absolute paths
CURRENT_DIR=$(pwd)
DIFFBIR_ABS_DIR=$(realpath "${DIFFBIR_DIR}")
INPUT_ABS_DIR=$(realpath "${input_dir}")
CLEAN_ABS_DIR=$(realpath "${clean_dir}")

# Create output directory with parameter-specific naming
# Format matches app.py parse_diffbir_method_name: prediction_steps{steps}_strength{strength}_up{upscale}_nocapt
OUTPUT_DIR="output/${subfolder}/diffbir/prediction_steps${steps}_strength${strength}_up${upscale}_nocapt"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_ABS_DIR=$(realpath "${OUTPUT_DIR}")

# Create metrics output directory
METRICS_DIR="output/${subfolder}/metrics"
mkdir -p "${METRICS_DIR}"
METRICS_ABS_DIR=$(realpath "${METRICS_DIR}")

# Build model name for metrics (matches app.py discover_diffbir_methods)
# metrics CSV: metrics_diffbir_{pred_dir_name_without_prediction_}.csv
PRED_SUFFIX="steps${steps}_strength${strength}_up${upscale}_nocapt"
MODEL_NAME="diffbir_${PRED_SUFFIX}"
SUMMARY_FILE="${METRICS_ABS_DIR}/summary_${MODEL_NAME}.txt"
CSV_FILE="${METRICS_ABS_DIR}/metrics_${MODEL_NAME}.csv"

# Check if metrics already exist for this configuration
if [ -f "${SUMMARY_FILE}" ] && [ -f "${CSV_FILE}" ]; then
    echo "Metrics already exist for this configuration. Skipping entire run."
    echo "Summary: ${SUMMARY_FILE}"
    echo "CSV: ${CSV_FILE}"
    exit 0
fi

# Check if DiffBIR is set up
if [ ! -d "${DIFFBIR_ABS_DIR}" ]; then
    echo "ERROR: DiffBIR not found at ${DIFFBIR_ABS_DIR}"
    echo "Please run: bash script/restoration/eval/01_setup_diffbir.sh"
    exit 1
fi

# Check if clean directory exists
if [ ! -d "${CLEAN_ABS_DIR}" ]; then
    echo "ERROR: Clean images directory not found at ${CLEAN_ABS_DIR}"
    exit 1
fi

echo "Running DiffBIR inference..."

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
    # Build max_images arguments
    # DiffBIR doesn't have --max_images natively, so we limit input files
    INFERENCE_INPUT="${INPUT_ABS_DIR}"
    if [ -n "${max_images}" ] && [ "${max_images}" -gt 0 ] 2>/dev/null; then
        # Create a temp directory with symlinks to limit images
        TEMP_INPUT=$(mktemp -d)
        trap "rm -rf ${TEMP_INPUT}" EXIT
        
        COUNT=0
        for img in "${INPUT_ABS_DIR}"/*.png "${INPUT_ABS_DIR}"/*.jpg "${INPUT_ABS_DIR}"/*.jpeg; do
            [ -e "$img" ] || continue
            
            # Skip already-processed images
            basename_img=$(basename "$img")
            restored_img="${OUTPUT_ABS_DIR}/${basename_img}"
            if [ -f "${restored_img}" ]; then
                continue
            fi
            
            ln -s "$img" "${TEMP_INPUT}/$(basename "$img")"
            COUNT=$((COUNT + 1))
            if [ ${COUNT} -ge ${max_images} ]; then
                break
            fi
        done
        
        echo "Limiting to ${COUNT} images (max_images=${max_images})"
        INFERENCE_INPUT="${TEMP_INPUT}"
    fi

    # Run DiffBIR inference using conda run with working directory set to DiffBIR
    # Unset venv variables to prevent interference with conda env
    unset VIRTUAL_ENV
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "\.venv" | tr '\n' ':' | sed 's/:$//')

    conda run -n ${DIFFBIR_CONDA_ENV} --cwd "${DIFFBIR_ABS_DIR}" --no-capture-output \
        python inference.py \
        --version v2.1 \
        --task $task \
        --input "${INFERENCE_INPUT}" \
        --output "${OUTPUT_ABS_DIR}" \
        --seed 1234 \
        --steps ${steps} \
        --upscale ${upscale} \
        --strength ${strength} \
        --captioner none \
        --cleaner_tiled \
        --vae_encoder_tiled \
        --vae_decoder_tiled \
        --cldm_tiled \
        --device cuda

    echo "DiffBIR inference complete. Results saved to: ${OUTPUT_ABS_DIR}"
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
