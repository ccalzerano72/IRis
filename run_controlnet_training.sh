#!/bin/bash

# Configuration
EXPERIMENT_NAME="controlnet_restoration_001"
OUTPUT_DIR="output/${EXPERIMENT_NAME}"
LOG_FILE="training_${EXPERIMENT_NAME}.log"

# Clear GPU memory first
echo "Clearing GPU memory..."
python clear_gpu_memory.py

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start training
echo "Starting training..."
nohup python script/controlnet_restoration/train.py \
    --config config/train_marigold_controlnet_restoration.yaml \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo $PID > training.pid

# Wait a moment for startup
sleep 5

# Follow the log
tail -f "$LOG_FILE"
