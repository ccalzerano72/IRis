#!/usr/bin/env bash
# HYPIR Inference for Evaluation
# Runs HYPIR (SD2 LoRA, single-step GAN distilled from diffusion) on test images
# for comparison with Marigold/DiffBIR/StableSR/Restormer/DFPIR/Real-ESRGAN.
#
# HYPIR is a single-step restoration model that uses a LoRA-finetuned SD 2.1 U-Net
# with adversarial training. Unlike iterative diffusion models, it produces output
# in a single forward pass through the U-Net.
#
# Parameters are hardcoded to the authors' recommended defaults (model_t=200,
# coeff_t=200, lora_rank=256, patch_size=512, stride=256).
# Only lq_dir, output subfolder, upscale, clean_dir, and max_images are configurable.

set -e
set -x

# Ensure conda is available
CONDA_SH="${CONDA_SH:-/titaniumtank/ccalzerano/miniconda3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

# Configuration
HYPIR_DIR="${HYPIR_DIR:-external/HYPIR}"
HYPIR_CONDA_ENV="${HYPIR_CONDA_ENV:-hypir}"
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"
subfolder=${1:-"eval"}
input_dir=${2:-"${BASE_DATA_DIR}/restoration_test/degraded"}
clean_dir=${3:-"${BASE_DATA_DIR}/restoration_test/clean"}
upscale=${4:-1}           # 1 = same-resolution restoration, 4 = super-resolution
max_images=${5:-""}       # Max images to process (empty = all)
gpu=${6:-0}

# Hardcoded parameters (authors' recommended defaults)
MODEL_T=200
COEFF_T=200
LORA_RANK=256
PATCH_SIZE=512
STRIDE=256
SEED=1234
WEIGHT_FILE="weights/HYPIR_sd2.pth"
BASE_MODEL="sd-research/stable-diffusion-2-1-base"

# LoRA modules (from authors' README)
LORA_MODULES_LIST=(to_k to_q to_v to_out.0 conv conv1 conv2 conv_shortcut conv_out proj_in proj_out ff.net.2 ff.net.0.proj)
IFS=','
LORA_MODULES="${LORA_MODULES_LIST[*]}"
unset IFS

# Build model name
MODEL_NAME="hypir_sd2_up${upscale}"

# Get absolute paths
CURRENT_DIR=$(pwd)
HYPIR_ABS_DIR=$(realpath "${HYPIR_DIR}")
INPUT_ABS_DIR=$(realpath "${input_dir}")
CLEAN_ABS_DIR=$(realpath "${clean_dir}")

# Create output directory
OUTPUT_DIR="output/${subfolder}/${MODEL_NAME}/restored"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_ABS_DIR=$(realpath "${OUTPUT_DIR}")

# Create metrics output directory
METRICS_DIR="output/${subfolder}/metrics"
mkdir -p "${METRICS_DIR}"
METRICS_ABS_DIR=$(realpath "${METRICS_DIR}")

# Check if metrics already exist
SUMMARY_FILE="${METRICS_ABS_DIR}/summary_${MODEL_NAME}.txt"
CSV_FILE="${METRICS_ABS_DIR}/metrics_${MODEL_NAME}.csv"

if [ -f "${SUMMARY_FILE}" ] && [ -f "${CSV_FILE}" ]; then
    echo "Metrics already exist for ${MODEL_NAME}. Skipping entire run."
    echo "Summary: ${SUMMARY_FILE}"
    echo "CSV: ${CSV_FILE}"
    exit 0
fi

# Check directories
if [ ! -d "${HYPIR_ABS_DIR}" ]; then
    echo "ERROR: HYPIR not found at ${HYPIR_ABS_DIR}"
    echo "Please run: cd external && git clone https://github.com/XPixelGroup/HYPIR.git"
    exit 1
fi

if [ ! -d "${INPUT_ABS_DIR}" ]; then
    echo "ERROR: Input directory not found at ${INPUT_ABS_DIR}"
    exit 1
fi

if [ ! -d "${CLEAN_ABS_DIR}" ]; then
    echo "ERROR: Clean images directory not found at ${CLEAN_ABS_DIR}"
    exit 1
fi

if [ ! -f "${HYPIR_ABS_DIR}/${WEIGHT_FILE}" ]; then
    echo "ERROR: HYPIR weights not found at ${HYPIR_ABS_DIR}/${WEIGHT_FILE}"
    echo "Please download: wget -O ${HYPIR_ABS_DIR}/${WEIGHT_FILE} https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth"
    exit 1
fi

# Check which images are already processed
EXISTING_COUNT=0
TOTAL_COUNT=0
for img in "${INPUT_ABS_DIR}"/*.png "${INPUT_ABS_DIR}"/*.jpg "${INPUT_ABS_DIR}"/*.jpeg; do
    [ -e "$img" ] || continue
    # Skip non-image files (e.g. degradation_metadata.json)
    basename_img=$(basename "$img")
    ext="${basename_img##*.}"
    case "${ext}" in
        png|jpg|jpeg|PNG|JPG|JPEG) ;;
        *) continue ;;
    esac
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    name_no_ext="${basename_img%.*}"
    restored_img="${OUTPUT_ABS_DIR}/${name_no_ext}.png"

    if [ -f "${restored_img}" ]; then
        EXISTING_COUNT=$((EXISTING_COUNT + 1))
    fi
done

echo "Found ${EXISTING_COUNT}/${TOTAL_COUNT} images already processed"

if [ ${EXISTING_COUNT} -eq ${TOTAL_COUNT} ] && [ ${TOTAL_COUNT} -gt 0 ]; then
    echo "All images already processed. Skipping inference."
else
    # Prepare input: create temp dir with image-only symlinks, optionally limited
    INFERENCE_INPUT="${INPUT_ABS_DIR}"
    if [ -n "${max_images}" ] && [ "${max_images}" -gt 0 ] 2>/dev/null; then
        TEMP_INPUT=$(mktemp -d)
        trap "rm -rf ${TEMP_INPUT}" EXIT

        COUNT=0
        for img in "${INPUT_ABS_DIR}"/*.png "${INPUT_ABS_DIR}"/*.jpg "${INPUT_ABS_DIR}"/*.jpeg; do
            [ -e "$img" ] || continue
            basename_img=$(basename "$img")
            ext="${basename_img##*.}"
            case "${ext}" in
                png|jpg|jpeg|PNG|JPG|JPEG) ;;
                *) continue ;;
            esac

            # Skip already-processed
            name_no_ext="${basename_img%.*}"
            restored_img="${OUTPUT_ABS_DIR}/${name_no_ext}.png"
            if [ -f "${restored_img}" ]; then
                continue
            fi

            ln -s "$img" "${TEMP_INPUT}/${basename_img}"
            COUNT=$((COUNT + 1))
            if [ ${COUNT} -ge ${max_images} ]; then
                break
            fi
        done

        echo "Limiting to ${COUNT} images (max_images=${max_images})"
        INFERENCE_INPUT="${TEMP_INPUT}"
    fi

    echo "Running HYPIR inference..."
    echo "  Model: HYPIR-SD2 (single-step GAN)"
    echo "  Input: ${INFERENCE_INPUT}"
    echo "  Upscale: ${upscale}"
    echo "  model_t: ${MODEL_T}, coeff_t: ${COEFF_T}"
    echo "  Patch size: ${PATCH_SIZE}, Stride: ${STRIDE}"
    echo "  Seed: ${SEED}"
    echo "  GPU: ${gpu}"

    # HYPIR saves to {output_dir}/result/{filename}.png
    # We use a temp output dir, then move results to our standard location
    HYPIR_OUTPUT_DIR=$(mktemp -d)

    # Use the conda env's Python directly to avoid conflicts with active venvs.
    # conda run can fail when a virtualenv is active in the parent process.
    HYPIR_PYTHON="${HOME}/miniconda3/envs/${HYPIR_CONDA_ENV}/bin/python"
    if [ ! -f "${HYPIR_PYTHON}" ]; then
        HYPIR_PYTHON="/titaniumtank/ccalzerano/miniconda3/envs/${HYPIR_CONDA_ENV}/bin/python"
    fi
    if [ ! -f "${HYPIR_PYTHON}" ]; then
        echo "ERROR: Cannot find Python for conda env ${HYPIR_CONDA_ENV}"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${HYPIR_PYTHON} "${HYPIR_ABS_DIR}/test.py" \
        --base_model_type sd2 \
        --base_model_path "${BASE_MODEL}" \
        --model_t ${MODEL_T} \
        --coeff_t ${COEFF_T} \
        --lora_rank ${LORA_RANK} \
        --lora_modules ${LORA_MODULES} \
        --weight_path "${HYPIR_ABS_DIR}/${WEIGHT_FILE}" \
        --patch_size ${PATCH_SIZE} \
        --stride ${STRIDE} \
        --lq_dir "${INFERENCE_INPUT}" \
        --scale_by factor \
        --upscale ${upscale} \
        --captioner empty \
        --output_dir "${HYPIR_OUTPUT_DIR}" \
        --seed ${SEED} \
        --device cuda

    # Move results from HYPIR's output structure to our standard location
    # HYPIR saves to {output_dir}/result/*.png
    HYPIR_RESULT_DIR="${HYPIR_OUTPUT_DIR}/result"
    if [ -d "${HYPIR_RESULT_DIR}" ]; then
        for img in "${HYPIR_RESULT_DIR}"/*.png; do
            [ -e "$img" ] || continue
            mv "$img" "${OUTPUT_ABS_DIR}/"
        done
    else
        echo "WARNING: HYPIR result directory not found at ${HYPIR_RESULT_DIR}"
        echo "Checking output dir contents:"
        ls -la "${HYPIR_OUTPUT_DIR}/" 2>/dev/null || true
        # Try moving from output dir directly if no result/ subdirectory
        for img in "${HYPIR_OUTPUT_DIR}"/*.png; do
            [ -e "$img" ] || continue
            mv "$img" "${OUTPUT_ABS_DIR}/"
        done
    fi

    # Cleanup
    rm -rf "${HYPIR_OUTPUT_DIR}"

    echo "HYPIR inference complete. Results saved to: ${OUTPUT_ABS_DIR}"
fi

# Calculate metrics
if [ ! -f "${SUMMARY_FILE}" ] || [ ! -f "${CSV_FILE}" ]; then
    echo "Calculating metrics..."

    if [ ! -d "${MARIGOLD_VENV}" ]; then
        echo "ERROR: Marigold virtual environment not found at ${MARIGOLD_VENV}"
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
