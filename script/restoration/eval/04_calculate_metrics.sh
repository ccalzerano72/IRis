#!/usr/bin/env bash
# Calculate comprehensive metrics for restoration evaluation
# Metrics: PSNR, SSIM, LPIPS, ARNIQA, BRISQUE, NIQE, MANIQA, MUSIQ

set -e
set -x

# Configuration
MARIGOLD_VENV="${MARIGOLD_VENV:-.venv}"

# Default values
clean_dir=""
restored_dir=""
model_name=""
output_dir="output/eval/metrics"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean_dir)
            clean_dir="$2"
            shift 2
            ;;
        --restored_dir)
            restored_dir="$2"
            shift 2
            ;;
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --clean_dir <dir> --restored_dir <dir> --model_name <name> [--output_dir <dir>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${clean_dir}" ]; then
    echo "ERROR: --clean_dir is required"
    echo "Usage: $0 --clean_dir <dir> --restored_dir <dir> --model_name <name> [--output_dir <dir>]"
    exit 1
fi

if [ -z "${restored_dir}" ]; then
    echo "ERROR: --restored_dir is required"
    echo "Usage: $0 --clean_dir <dir> --restored_dir <dir> --model_name <name> [--output_dir <dir>]"
    exit 1
fi

if [ -z "${model_name}" ]; then
    echo "ERROR: --model_name is required"
    echo "Usage: $0 --clean_dir <dir> --restored_dir <dir> --model_name <name> [--output_dir <dir>]"
    exit 1
fi

# Get absolute paths
CLEAN_ABS_DIR=$(realpath "${clean_dir}")
RESTORED_ABS_DIR=$(realpath "${restored_dir}")

# Create output directory and get absolute path
mkdir -p "${output_dir}"
OUTPUT_ABS_DIR=$(realpath "${output_dir}")

# Check if directories exist
if [ ! -d "${CLEAN_ABS_DIR}" ]; then
    echo "ERROR: Clean directory not found at ${CLEAN_ABS_DIR}"
    exit 1
fi

if [ ! -d "${RESTORED_ABS_DIR}" ]; then
    echo "ERROR: Restored directory not found at ${RESTORED_ABS_DIR}"
    exit 1
fi

# Check if venv exists
if [ ! -d "${MARIGOLD_VENV}" ]; then
    echo "ERROR: Virtual environment not found at ${MARIGOLD_VENV}"
    echo "Please create venv or set MARIGOLD_VENV environment variable"
    exit 1
fi

# Activate venv and run metrics calculation
source "${MARIGOLD_VENV}/bin/activate"

python script/restoration/eval/04_calculate_metrics.py \
    --clean_dir "${CLEAN_ABS_DIR}" \
    --restored_dir "${RESTORED_ABS_DIR}" \
    --output_dir "${OUTPUT_ABS_DIR}" \
    --model_name "${model_name}" \
    --device cuda

deactivate

echo ""
echo "Metrics calculation complete!"
echo "Results saved to: ${OUTPUT_ABS_DIR}"
echo "  - metrics_${model_name}.csv (per-image)"
echo "  - summary_${model_name}.txt (statistics)"
