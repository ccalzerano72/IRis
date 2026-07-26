#!/usr/bin/env bash
# Unified Marigold Restoration Inference for Evaluation
#
# Auto-detects checkpoint type (base marigold, controlnet, hybrid-002, hybrid-003)
# and calls the correct run.py script with appropriate arguments.
#
# Replaces the three separate scripts:
#   - 02_infer_marigold.sh    (base marigold)
#   - 02_infer_controlnet.sh  (controlnet)
#   - 02_infer_hybrid.sh      (hybrid-002 and hybrid-003)
#
# Checkpoint type detection:
#   - hybrid_003_config.json present  → hybrid-003 (unet + controlnet in same ckpt)
#   - hybrid_config.json present      → hybrid-002 (separate base + controlnet)
#   - controlnet/ dir, NO unet/       → controlnet (needs separate base ckpt)
#   - unet/ dir, no hybrid configs    → base marigold
#
# Output prefix matches detected type: marigold_, controlnet_, hybrid_

set -e
set -x

# Configuration — same positional args as 02_infer_marigold.sh
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"
subfolder=${1:-"eval"}
input_dir=${2:-"${BASE_DATA_DIR}/restoration_test/degraded"}
clean_dir=${3:-"${BASE_DATA_DIR}/restoration_test/clean"}
processing_res=${4:-0}
denoise_steps=${5:-10}
ensemble_size=${6:-1}
ckpt=${7:-"checkpoints/marigold-restoration-latest"}
scheduler=${8:-"ddim"}
guidance_scale=${9:-1.0}
use_cpu=${10:-""}
max_images=${11:-""}

# Get absolute paths
CURRENT_DIR=$(pwd)
CKPT_ABS_PATH=$(realpath "${ckpt}")
INPUT_ABS_DIR=$(realpath "${input_dir}")
CLEAN_ABS_DIR=$(realpath "${clean_dir}")

# Extract checkpoint parent directory name for output folder.
# Use abspath (cd + pwd) instead of realpath to avoid resolving symlinks.
# If ckpt=.../012_hybrid_030000/latest where 'latest' is a symlink,
# realpath would resolve through it and lose the meaningful parent name.
CKPT_PARENT_DIR=$(basename "$(cd "$(dirname "${ckpt}")" && pwd)")


# =============================================================================
# AUTO-DETECT CHECKPOINT TYPE
# =============================================================================
HYBRID_003_CONFIG="${CKPT_ABS_PATH}/hybrid_003_config.json"
HYBRID_002_CONFIG="${CKPT_ABS_PATH}/hybrid_config.json"

if [ -f "${HYBRID_003_CONFIG}" ]; then
    CKPT_TYPE="hybrid"
    HYBRID_ARCH="003"
    echo "Detected hybrid-003 architecture (hybrid_003_config.json found)"
    echo "Hybrid checkpoint (unet + controlnet): ${CKPT_ABS_PATH}"

    # Verify checkpoint has both unet/ and controlnet/
    if [ ! -d "${CKPT_ABS_PATH}/unet" ]; then
        echo "ERROR: Hybrid-003 checkpoint missing unet/ at ${CKPT_ABS_PATH}"
        exit 1
    fi
    if [ ! -d "${CKPT_ABS_PATH}/controlnet" ]; then
        echo "ERROR: Hybrid-003 checkpoint missing controlnet/ at ${CKPT_ABS_PATH}"
        exit 1
    fi

elif [ -f "${HYBRID_002_CONFIG}" ]; then
    CKPT_TYPE="hybrid"
    HYBRID_ARCH="002"
    echo "Detected hybrid-002 architecture (hybrid_config.json found)"

    # Extract base_checkpoint_path using python (reliable JSON parsing)
    BASE_CKPT_PATH=$(python -c "import json; print(json.load(open('${HYBRID_002_CONFIG}'))['base_checkpoint_path'])")
    if [ -z "${BASE_CKPT_PATH}" ]; then
        echo "ERROR: base_checkpoint_path not found in ${HYBRID_002_CONFIG}"
        exit 1
    fi
    BASE_CKPT_ABS_PATH=$(realpath "${BASE_CKPT_PATH}")

    echo "Hybrid checkpoint (controlnet): ${CKPT_ABS_PATH}"
    echo "Base checkpoint (8ch UNet): ${BASE_CKPT_ABS_PATH}"

    # Verify base checkpoint has unet/
    if [ ! -d "${BASE_CKPT_ABS_PATH}/unet" ]; then
        echo "ERROR: Base checkpoint missing unet/ subdirectory at ${BASE_CKPT_ABS_PATH}"
        exit 1
    fi

elif [ -d "${CKPT_ABS_PATH}/controlnet" ] && [ ! -d "${CKPT_ABS_PATH}/unet" ]; then
    CKPT_TYPE="controlnet"
    echo "Detected ControlNet checkpoint (controlnet/ dir present, no unet/)"
    echo "ControlNet checkpoint: ${CKPT_ABS_PATH}"

elif [ -f "${CKPT_ABS_PATH}/architecture_config.json" ] && \
     python -c "import json,sys; cfg=json.load(open(sys.argv[1])); sys.exit(0 if cfg.get('architecture')=='controlnet_trainable_unet_4ch' else 1)" "${CKPT_ABS_PATH}/architecture_config.json" 2>/dev/null; then
    CKPT_TYPE="controlnet_4ch"
    echo "Detected 4ch trainable UNet + ControlNet checkpoint (architecture_config.json)"
    echo "Checkpoint (unet + controlnet): ${CKPT_ABS_PATH}"

    # Verify checkpoint has both unet/ and controlnet/
    if [ ! -d "${CKPT_ABS_PATH}/unet" ]; then
        echo "ERROR: controlnet_4ch checkpoint missing unet/ at ${CKPT_ABS_PATH}"
        exit 1
    fi
    if [ ! -d "${CKPT_ABS_PATH}/controlnet" ]; then
        echo "ERROR: controlnet_4ch checkpoint missing controlnet/ at ${CKPT_ABS_PATH}"
        exit 1
    fi

elif [ -d "${CKPT_ABS_PATH}/unet" ]; then
    CKPT_TYPE="marigold"
    echo "Detected base Marigold checkpoint (unet/ dir present)"
    echo "Marigold checkpoint: ${CKPT_ABS_PATH}"

else
    echo "ERROR: Cannot detect checkpoint type at ${CKPT_ABS_PATH}"
    echo "Expected one of:"
    echo "  - unet/ directory (base marigold)"
    echo "  - controlnet/ directory without unet/ (controlnet)"
    echo "  - hybrid_003_config.json (hybrid-003)"
    echo "  - hybrid_config.json (hybrid-002)"
    exit 1
fi

echo "Checkpoint type: ${CKPT_TYPE}"


# =============================================================================
# OUTPUT DIRECTORY SETUP
# =============================================================================
# Include guidance_scale in name only if CFG is used (scale > 1.0)
if (( $(echo "$guidance_scale > 1.0" | bc -l) )); then
    cfg_suffix="_cfg${guidance_scale}"
else
    cfg_suffix=""
fi

# Output prefix matches checkpoint type: marigold_, controlnet_, hybrid_
OUTPUT_DIR="output/${subfolder}/${CKPT_TYPE}_${CKPT_PARENT_DIR}/prediction_${processing_res}_s$(printf "%02d" ${denoise_steps})_e$(printf "%02d" ${ensemble_size})_${scheduler}${cfg_suffix}"
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
OUTPUT_SUBDIR="${CKPT_TYPE}_${CKPT_PARENT_DIR}_prediction_${processing_res}_s$(printf "%02d" ${denoise_steps})_e$(printf "%02d" ${ensemble_size})_${scheduler}${cfg_suffix}"
SUMMARY_FILE="${METRICS_ABS_DIR}/summary_${OUTPUT_SUBDIR}.txt"
CSV_FILE="${METRICS_ABS_DIR}/metrics_${OUTPUT_SUBDIR}.csv"

if [ -f "${SUMMARY_FILE}" ] && [ -f "${CSV_FILE}" ]; then
    echo "Metrics already exist for this configuration. Skipping entire run."
    echo "Summary: ${SUMMARY_FILE}"
    echo "CSV: ${CSV_FILE}"
    exit 0
fi


# =============================================================================
# VALIDATION
# =============================================================================
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

echo "Running ${CKPT_TYPE} inference..."

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

    # =========================================================================
    # DISPATCH TO CORRECT run.py BASED ON CHECKPOINT TYPE
    # =========================================================================
    if [ "${CKPT_TYPE}" = "marigold" ]; then
        # Base Marigold: script/restoration/run.py
        # Supports --cpu flag
        CPU_FLAG=""
        if [ "${use_cpu}" = "cpu" ]; then
            CPU_FLAG="--cpu"
            echo "Forcing CPU inference (use_cpu=cpu)"
        fi

        if [ "${use_cpu}" = "cpu" ]; then
            CUDA_VISIBLE_DEVICES="" python script/restoration/run.py \
                --checkpoint "${CKPT_ABS_PATH}" \
                --seed 1234 \
                --input_rgb_dir "${INPUT_ABS_DIR}" \
                --processing_res ${processing_res} \
                --denoise_steps ${denoise_steps} \
                --ensemble_size ${ensemble_size} \
                --scheduler ${scheduler} \
                --guidance_scale ${guidance_scale} \
                --half_precision \
                --cpu \
                --output_dir "${OUTPUT_ABS_DIR}" \
                ${MAX_IMAGES_FLAG}
        else
            python script/restoration/run.py \
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
        fi

    elif [ "${CKPT_TYPE}" = "controlnet" ]; then
        # ControlNet: script/controlnet_restoration/run.py
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

    elif [ "${CKPT_TYPE}" = "controlnet_4ch" ]; then
        # 4ch trainable UNet + ControlNet (re_004 ablation)
        # Uses same pipeline as controlnet but loads UNet from checkpoint
        python script/controlnet_restoration/run.py \
            --checkpoint "${CKPT_ABS_PATH}" \
            --load_unet_from_checkpoint \
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

    elif [ "${CKPT_TYPE}" = "hybrid" ]; then
        # Hybrid: script/hybrid_controlnet_restoration/run.py
        # 003 uses --checkpoint, 002 uses --base_checkpoint + --controlnet_checkpoint
        if [ "${HYBRID_ARCH}" = "003" ]; then
            python script/hybrid_controlnet_restoration/run.py \
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
        else
            python script/hybrid_controlnet_restoration/run.py \
                --base_checkpoint "${BASE_CKPT_ABS_PATH}" \
                --controlnet_checkpoint "${CKPT_ABS_PATH}" \
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
        fi
    fi

    echo "${CKPT_TYPE} inference complete. Results saved to: ${OUTPUT_ABS_DIR}"
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
echo "Checkpoint type: ${CKPT_TYPE}"
echo "Inference results: ${OUTPUT_ABS_DIR}"
echo "Metrics results: ${METRICS_ABS_DIR}"
