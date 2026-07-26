# Restoration Model Evaluation Framework

Evaluation framework for comparing Marigold-based restoration model against DiffBIR (ECCV 2024).

## Structure

```
script/restoration/eval/
├── 01_setup_diffbir.sh           # Setup DiffBIR repository and weights
├── 02_infer_marigold.sh          # Run Marigold restoration inference
├── 03_infer_diffbir.sh           # Run DiffBIR inference
├── 04_calculate_metrics.py       # Calculate comprehensive metrics
├── 05_compare_results.py         # Compare and visualize results
└── README.md                     # This file
```

## Prerequisites

- Python 3.10+
- PyTorch 2.2.2+ (for DiffBIR memory-efficient attention)
- CUDA-capable GPU with sufficient VRAM (recommended: 24GB for DiffBIR)

## Quick Start

### Step 1: Setup DiffBIR

```bash
# Clone DiffBIR and download pretrained weights
bash script/restoration/eval/01_setup_diffbir.sh

# Install DiffBIR dependencies (in separate venv recommended)
cd external/DiffBIR
pip install -r requirements.txt
cd ../..
```

### Step 2: Prepare Test Data

Organize your test data as follows:
```
datasets/restoration_test/
├── degraded/          # Degraded input images
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── ground_truth/      # Clean ground truth images
    ├── image001.png
    ├── image002.png
    └── ...
```

### Step 3: Run Inference

```bash
# Set environment variables
export BASE_DATA_DIR=/path/to/datasets
export DIFFBIR_DIR=external/DiffBIR

# Run Marigold restoration
bash script/restoration/eval/02_infer_marigold.sh \
    checkpoints/marigold-restoration-latest \
    eval \
    ${BASE_DATA_DIR}/restoration_test/degraded

# Run DiffBIR
bash script/restoration/eval/03_infer_diffbir.sh \
    eval \
    ${BASE_DATA_DIR}/restoration_test/degraded
```

### Step 4: Calculate Metrics

```bash
python script/restoration/eval/04_calculate_metrics.py \
    --gt_dir ${BASE_DATA_DIR}/restoration_test/ground_truth \
    --pred_dirs output/eval/marigold/prediction/restored \
               output/eval/diffbir/prediction \
    --model_names marigold diffbir \
    --output_dir output/eval/metrics
```

### Step 5: Compare and Visualize

```bash
python script/restoration/eval/05_compare_results.py \
    --metrics_dir output/eval/metrics \
    --gt_dir ${BASE_DATA_DIR}/restoration_test/ground_truth \
    --degraded_dir ${BASE_DATA_DIR}/restoration_test/degraded \
    --image_dirs output/eval/marigold/prediction/restored \
                 output/eval/diffbir/prediction \
    --model_names marigold diffbir \
    --output_dir output/eval/comparison \
    --max_images 20
```

## Output Structure

After running the full pipeline:
```
output/eval/
├── marigold/
│   └── prediction/
│       ├── restored/           # Restored images
│       └── restored_npy/       # NumPy arrays
├── diffbir/
│   └── prediction/             # DiffBIR outputs
├── metrics/
│   ├── per_image_metrics.csv   # Per-image metrics
│   ├── summary_metrics.json    # Aggregated statistics
│   └── comparison_table.txt    # Formatted comparison
└── comparison/
    ├── metrics_comparison.png  # Bar chart
    ├── psnr_vs_lpips.png       # Scatter plot
    ├── comparison_report.txt   # Text summary
    └── side_by_side/           # Visual comparisons
        ├── comparison_image001.png
        └── ...
```

## Metrics Computed

| Metric | Type | Direction | Description |
|--------|------|-----------|-------------|
| PSNR | Distortion | ↑ Higher better | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Distortion | ↑ Higher better | Structural Similarity Index |
| L1 | Distortion | ↓ Lower better | Mean Absolute Error |
| LPIPS (Alex) | Perceptual | ↓ Lower better | Learned Perceptual Similarity |
| LPIPS (VGG) | Perceptual | ↓ Lower better | Learned Perceptual Similarity |

## DiffBIR Configuration

DiffBIR supports different tasks:
- `sr` - Blind super-resolution (default)
- `face` - Face restoration
- `denoise` - Image denoising

Modify `03_infer_diffbir.sh` to change the task:
```bash
bash script/restoration/eval/03_infer_diffbir.sh eval /path/to/images sr
```

## Troubleshooting

### CUDA Out of Memory
- DiffBIR requires significant VRAM. Use tiled sampling by adding `--tiled` flag
- Reduce image resolution or batch size

### DiffBIR Not Found
- Ensure `01_setup_diffbir.sh` completed successfully
- Check that `external/DiffBIR` directory exists
- Verify weights are downloaded in `external/DiffBIR/weights/`

### Mismatched Image Names
- The metrics script handles common naming patterns (_restored, _pred, etc.)
- Ensure GT and prediction images have matching base names

## References

- **DiffBIR**: Lin et al., "DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior", ECCV 2024
- **Marigold**: Ke et al., "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation", CVPR 2024
