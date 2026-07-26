#!/usr/bin/env bash
# Setup DiffBIR for local inference
# This script clones the DiffBIR repository and downloads pretrained weights

set -e
set -x

# Configuration
DIFFBIR_DIR="${1:-external/DiffBIR}"
WEIGHTS_DIR="${DIFFBIR_DIR}/weights"

echo "=== Setting up DiffBIR ==="
echo "Installation directory: ${DIFFBIR_DIR}"

# Create parent directory if it doesn't exist
mkdir -p "$(dirname "${DIFFBIR_DIR}")"

# Clone DiffBIR repository if not exists
if [ ! -d "${DIFFBIR_DIR}" ]; then
    echo "Cloning DiffBIR repository..."
    git clone https://github.com/XPixelGroup/DiffBIR.git "${DIFFBIR_DIR}"
else
    echo "DiffBIR repository already exists at ${DIFFBIR_DIR}"
fi

# Create weights directory
mkdir -p "${WEIGHTS_DIR}"

# Download pretrained weights from HuggingFace
echo "=== Downloading pretrained weights ==="

# IRControlNet v2.1 (latest, trained on filtered unsplash)
if [ ! -f "${WEIGHTS_DIR}/v2.1.pt" ]; then
    echo "Downloading IRControlNet v2.1..."
    wget -O "${WEIGHTS_DIR}/v2.1.pt" \
        "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.1.pt"
else
    echo "v2.1.pt already exists"
fi

# SwinIR for degradation removal (codeformer degradation)
if [ ! -f "${WEIGHTS_DIR}/codeformer_swinir.ckpt" ]; then
    echo "Downloading SwinIR (codeformer degradation)..."
    wget -O "${WEIGHTS_DIR}/codeformer_swinir.ckpt" \
        "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/codeformer_swinir.ckpt"
else
    echo "codeformer_swinir.ckpt already exists"
fi

# Alternative: SwinIR for Real-ESRGAN degradation
if [ ! -f "${WEIGHTS_DIR}/realesrgan_s4_swinir_100k.pth" ]; then
    echo "Downloading SwinIR (Real-ESRGAN degradation)..."
    wget -O "${WEIGHTS_DIR}/realesrgan_s4_swinir_100k.pth" \
        "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth"
else
    echo "realesrgan_s4_swinir_100k.pth already exists"
fi

echo ""
echo "=== DiffBIR Setup Complete ==="
echo "Repository: ${DIFFBIR_DIR}"
echo "Weights: ${WEIGHTS_DIR}"
echo ""
echo "To install DiffBIR dependencies, run:"
echo "  cd ${DIFFBIR_DIR} && pip install -r requirements.txt"
echo ""
echo "Note: DiffBIR requires PyTorch 2.2.2+ for memory-efficient attention"
echo "      or PyTorch 1.13.1 with xformers 0.0.16"
