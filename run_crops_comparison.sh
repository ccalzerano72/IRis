#!/bin/bash

# --------------------------------------------------------------------------
# Crops Comparison: Run all baseline models on all degradation types
# Launches run_crops_comparison.py with nohup so it survives shell close.
#
# Input layout:
#   comparison_crops/input/clean/          <- reference images (shared)
#   comparison_crops/input/degraded/*/     <- one subdir per degradation type
#
# Output layout:
#   comparison_crops/output/<degradation>/<model>/restored/
#   comparison_crops/output/<degradation>/metrics/metrics_<model>.csv
#   comparison_crops/output/<degradation>/metrics/summary_<model>.txt
#
# Usage:
#   bash run_crops_comparison.sh                    # Full run (all images)
#   bash run_crops_comparison.sh --test             # Test mode (2 images per run)
#   bash run_crops_comparison.sh --input_dir path   # Custom input directory
# --------------------------------------------------------------------------

# Activate virtual environment
source .venv/bin/activate

# Configuration
LOG_FILE="crops_comparison.log"
EXTRA_ARGS="$@"

# If --test flag is passed, use a separate log file
for arg in "$@"; do
    if [ "$arg" = "--test" ]; then
        LOG_FILE="crops_comparison_test.log"
        break
    fi
done

# Clear GPU memory first
echo "Clearing GPU memory..."
python clear_gpu_memory.py

# Start comparison
echo "Starting crops comparison..."
echo "  Log file: ${LOG_FILE}"
echo "  Extra args: ${EXTRA_ARGS}"
setsid nohup python script/restoration/eval/run_crops_comparison.py \
    ${EXTRA_ARGS} \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Comparison started with PID: $PID"
echo $PID > crops_comparison.pid

# Wait a moment for startup
sleep 3

# Follow the log
tail -f "$LOG_FILE"
