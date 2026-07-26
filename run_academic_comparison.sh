#!/bin/bash

# --------------------------------------------------------------------------
# Academic Comparison: Run all baseline models on all datasets
# Launches run_academic_comparison.py with nohup so it survives shell close.
#
# Usage:
#   bash run_academic_comparison.sh          # Full run (all images)
#   bash run_academic_comparison.sh --test   # Test mode (2 images per run)
# --------------------------------------------------------------------------

# Activate virtual environment
source .venv/bin/activate

# Configuration
LOG_FILE="nocn_academic_comparison.log"
EXTRA_ARGS="$@"

# If --test flag is passed, use a separate log file
for arg in "$@"; do
    if [ "$arg" = "--test" ]; then
        LOG_FILE="academic_comparison_test.log"
        break
    fi
done

# Clear GPU memory first
echo "Clearing GPU memory..."
python clear_gpu_memory.py

# Start comparison
echo "Starting academic comparison..."
echo "  Log file: ${LOG_FILE}"
echo "  Extra args: ${EXTRA_ARGS}"
setsid nohup python script/restoration/eval/run_academic_comparison.py \
# setsid nohup python script/restoration/eval/run_realsr_comparison.py \
    ${EXTRA_ARGS} \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Comparison started with PID: $PID"
echo $PID > comparison.pid

# Wait a moment for startup
sleep 3

# Follow the log
tail -f "$LOG_FILE"
