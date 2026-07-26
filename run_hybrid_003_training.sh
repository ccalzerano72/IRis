#!/bin/bash

# --------------------------------------------------------------------------
# Hybrid-003: Joint UNet + ControlNet + ARNIQA Training
# Based on run_hybrid_training.sh pattern
# --------------------------------------------------------------------------

# Activate virtual environment
source .venv/bin/activate

# Configuration
EXPERIMENT_NAME="hybrid_re_008_linearSched"
OUTPUT_DIR="output/${EXPERIMENT_NAME}"
LOG_FILE="training_${EXPERIMENT_NAME}.log"

# Check if output directory already exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "WARNING: Output directory already exists: $OUTPUT_DIR"
    read -p "Do you want to delete it and start fresh? (y/n): " choice
    case "$choice" in
        y|Y )
            echo "Removing $OUTPUT_DIR..."
            rm -rf "$OUTPUT_DIR"
            echo "Removed."
            ;;
        * )
            echo "Aborting. Rename or remove the existing directory first."
            exit 1
            ;;
    esac
fi

# Clear GPU memory first
echo "Clearing GPU memory..."
python clear_gpu_memory.py

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start training
echo "Starting training..."
nohup python script/hybrid_controlnet_restoration_003/train.py \
    --config config/train_marigold_hybrid_controlnet_restoration_003.yaml \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo $PID > training.pid

# Wait a moment for startup
sleep 5

# Follow the log
tail -f "$LOG_FILE"
