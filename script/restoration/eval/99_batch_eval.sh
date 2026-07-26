#!/usr/bin/env bash
# Batch Evaluation Script for Marigold Restoration
# Runs multiple parameter combinations for comprehensive evaluation

set -e

# Check if required parameters are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <subfolder> <input_dir> <clean_dir> <checkpoint> [scheduler] [--cpu]"
    echo ""
    echo "Arguments:"
    echo "  subfolder   - Output subfolder name"
    echo "  input_dir   - Path to degraded input images"
    echo "  clean_dir   - Path to clean reference images"
    echo "  checkpoint  - Path to model checkpoint"
    echo "  scheduler   - Scheduler type: ddim, lcm, heun (default: ddim)"
    echo "  --cpu       - Force CPU inference (slow but works when GPU is full)"
    echo ""
    echo "Example:"
    echo "  $0 eval_batch input/degraded input/clean checkpoints/013_best_008000/latest"
    echo "  $0 eval_batch input/degraded input/clean checkpoints/013_best_008000/latest lcm"
    echo "  $0 eval_batch input/degraded input/clean checkpoints/013_best_008000/latest ddim --cpu"
    echo ""
    echo "This will run all combinations of:"
    echo "  - Processing resolution: 0, 768"
    echo "  - Denoise steps: 1, 5, 10, 20, 50"
    echo "  - Ensemble size: 1, 10"
    echo "  Total: 20 runs"
    exit 1
fi

# Parameters
SUBFOLDER=$1
INPUT_DIR=$2
CLEAN_DIR=$3
CHECKPOINT=$4

# Check for --cpu flag and scheduler in arguments 5+
USE_CPU=""
SCHEDULER="ddim"  # default

for arg in "${@:5}"; do
    if [ "$arg" = "--cpu" ]; then
        USE_CPU="cpu"
    elif [ "$arg" = "ddim" ] || [ "$arg" = "lcm" ] || [ "$arg" = "heun" ]; then
        SCHEDULER="$arg"
    fi
done

# Parameter arrays
PROCESSING_RES=(0)
DENOISE_STEPS=(1 5 10 25)
ENSEMBLE_SIZES=(1 5)
GUIDANCE_SCALES=(1.0)

# Calculate total runs
TOTAL_RUNS=$((${#PROCESSING_RES[@]} * ${#DENOISE_STEPS[@]} * ${#ENSEMBLE_SIZES[@]} * ${#GUIDANCE_SCALES[@]}))

echo "=========================================="
echo "BATCH EVALUATION - MARIGOLD RESTORATION"
echo "=========================================="
echo "Subfolder: ${SUBFOLDER}"
echo "Input dir: ${INPUT_DIR}"
echo "Clean dir: ${CLEAN_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Scheduler: ${SCHEDULER}"
echo "CPU mode: ${USE_CPU:-"no (using GPU)"}"
echo ""
echo "Parameter combinations:"
echo "  Processing resolution: ${PROCESSING_RES[*]}"
echo "  Denoise steps: ${DENOISE_STEPS[*]}"
echo "  Ensemble sizes: ${ENSEMBLE_SIZES[*]}"
echo "  Guidance scales (CFG): ${GUIDANCE_SCALES[*]}"
echo ""
echo "Total runs: ${TOTAL_RUNS}"
echo "=========================================="

# Confirm before starting
read -p "Continue with batch evaluation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run counter
RUN_COUNT=0
START_TIME=$(date +%s)

# Quadruple nested loop for all combinations
for res in "${PROCESSING_RES[@]}"; do
    for steps in "${DENOISE_STEPS[@]}"; do
        for ensemble in "${ENSEMBLE_SIZES[@]}"; do
            for cfg in "${GUIDANCE_SCALES[@]}"; do
                RUN_COUNT=$((RUN_COUNT + 1))
                
                echo ""
                echo "=========================================="
                echo "RUN ${RUN_COUNT}/${TOTAL_RUNS}"
                echo "Resolution: ${res}, Steps: ${steps}, Ensemble: ${ensemble}, Scheduler: ${SCHEDULER}, CFG: ${cfg}"
                echo "=========================================="
                
                # Record start time for this run
                RUN_START=$(date +%s)
                
                # Run the evaluation
                bash script/restoration/eval/02_infer_marigold.sh \
                    "${SUBFOLDER}" \
                    "${INPUT_DIR}" \
                    "${CLEAN_DIR}" \
                    "${res}" \
                    "${steps}" \
                    "${ensemble}" \
                    "${CHECKPOINT}" \
                    "${SCHEDULER}" \
                    "${cfg}" \
                    "${USE_CPU}"
                
                # Calculate run time
                RUN_END=$(date +%s)
                RUN_TIME=$((RUN_END - RUN_START))
                
                echo "Run ${RUN_COUNT} completed in ${RUN_TIME}s"
                
                # Estimate remaining time
                if [ $RUN_COUNT -lt $TOTAL_RUNS ]; then
                    ELAPSED=$((RUN_END - START_TIME))
                    AVG_TIME=$((ELAPSED / RUN_COUNT))
                    REMAINING_RUNS=$((TOTAL_RUNS - RUN_COUNT))
                    ETA=$((REMAINING_RUNS * AVG_TIME))
                    
                    echo "Estimated time remaining: ${ETA}s ($(($ETA / 60))m)"
                fi
            done
        done
    done
done

# Final summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "BATCH EVALUATION COMPLETE"
echo "=========================================="
echo "Total runs: ${TOTAL_RUNS}"
echo "Total time: ${TOTAL_TIME}s ($(($TOTAL_TIME / 60))m)"
echo "Average time per run: $((TOTAL_TIME / TOTAL_RUNS))s"
echo ""
echo "Results saved in: output/${SUBFOLDER}/"
echo "Metrics saved in: output/${SUBFOLDER}/metrics/"
echo "=========================================="