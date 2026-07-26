# IRis: Image Restoration via Latent Diffusion

This repository contains the implementation for the Master's thesis **"Blind Image Restoration via Dual-Conditioned Latent Diffusion"**.

The project extends the [Marigold](https://github.com/prs-eth/Marigold) framework to support blind image restoration, demonstrating how diffusion models pretrained for image generation can be repurposed for image enhancement tasks.

## Approach

IRis introduces a hybrid architecture combining:

1. **8-channel UNet**: The standard Stable Diffusion 2 UNet is modified to accept 8 input channels (4 for degraded latent + 4 for noisy latent), enabling direct conditioning on the degraded image in latent space.

2. **ControlNet guidance**: A ControlNet branch processes the degraded image in pixel space, providing structural guidance to the denoising process while preserving fine details.

3. **Joint training**: Both the modified UNet and ControlNet are trained jointly on synthetic degradations (noise, blur, JPEG compression, resize artifacts) applied to high-quality images from a curated subset of LAION-Aesthetics.

The model operates in the latent space of a pretrained VAE encoder/decoder, enabling efficient processing while leveraging the strong image priors learned during Stable Diffusion pretraining.

## Results

Evaluation on two axes: **synthetic degradations** (same pipeline as training) and **real-world degradations** (unseen optical artifacts from RealSR dataset).

### Synthetic Degradations (DIV2K, 100 images)

| Method           | PSNR ↑    | SSIM ↑    | ΔE ↓     | LPIPS ↓   |
| ---------------- | --------- | --------- | -------- | --------- |
| _Degraded input_ | _22.91_   | _0.474_   | _6.36_   | _0.649_   |
| Real-ESRGAN      | 21.78     | 0.563     | 6.18     | 0.389     |
| Restormer        | 23.49     | 0.543     | 5.70     | 0.617     |
| DiffBIR          | 22.13     | 0.568     | 5.56     | 0.318     |
| HyPIR            | 20.13     | 0.498     | 7.26     | 0.362     |
| **IRis (Ours)**  | **24.71** | **0.670** | **4.54** | **0.312** |

### Real-World Degradations (RealSR L4, 100 images)

| Method           | PSNR ↑    | SSIM ↑    | ΔE ↓     | LPIPS ↓   |
| ---------------- | --------- | --------- | -------- | --------- |
| _Degraded input_ | _25.98_   | _0.743_   | _4.21_   | _0.428_   |
| Real-ESRGAN      | 23.24     | 0.698     | 5.12     | 0.387     |
| Restormer        | 25.91     | 0.742     | 4.16     | 0.452     |
| DiffBIR          | 23.00     | 0.643     | 5.05     | 0.354     |
| HyPIR            | 21.00     | 0.635     | 6.48     | 0.320     |
| **IRis (Ours)**  | **26.04** | **0.766** | **4.05** | **0.282** |

IRis ranks **1st on all metrics** on both synthetic and real-world degradations, achieving strong fidelity (PSNR, SSIM, ΔE) while maintaining the best perceptual quality (LPIPS).

### Statistical Significance

All reported comparisons were validated using **Wilcoxon signed-rank tests** (paired, non-parametric). Effect sizes are reported as rank-biserial correlation (_r_<sub>rb</sub>), where values > 0.5 indicate large effects.

**Synthetic degradations (DIV2K, Urban100):**

- All fidelity comparisons are highly significant (_p_ < 10⁻⁵) with large effect sizes (_r_<sub>rb</sub> > 0.9)
- IRis wins on 69–100 out of 100 images depending on metric and competitor
- Exception: LPIPS vs DiffBIR is **not significant** (_p_ > 0.28), indicating comparable perceptual quality; the architectural difference manifests in fidelity, not perception

**Real-world degradations (RealSR):**

- LPIPS advantage over Restormer is significant at all scale factors (_p_ < 10⁻⁸), winning on 43–50 out of 50 images
- PSNR vs Restormer varies by degradation severity:
  - x2 (mild): Restormer wins (_p_ < 10⁻⁹)
  - x3 (moderate): no significant difference (_p_ > 0.30)
  - x4 (severe): IRis wins slightly (_p_ = 0.011)
- Against DiffBIR: both PSNR and LPIPS are significant at all scales (_p_ < 10⁻⁶), confirming simultaneous advantage in fidelity and perception

### Visual Comparisons

Side-by-side qualitative results are available in:

- [Thesis](thesis/thesis.pdf) — Chapter 5 for the full quantitative analysis; Appendix C ("Additional Visual Comparisons") for per-image panels showing clean ground truth, degraded input, Restormer, RealESRGAN, DiffBIR, and IRis with 4× magnified crops and per-image metrics
- [Presentation slides](thesis/slides.pdf) — slide 12 ("Visual Comparison": degraded input, Restormer, DiffBIR, and IRis on the same crop, including a case where DiffBIR hallucinates incorrect texture)

## Installation

```bash
# Clone the repository
git clone https://github.com/ccalzerano72/IRis.git
cd IRis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

For training, install additional dependencies:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

## Checkpoint Download

Download the pretrained checkpoint (~5GB):

```bash
# Option 1: Download from GitHub releases
# Download all parts: 002_re_015000.z01, 002_re_015000.z02, 002_re_015000.z03, 002_re_015000.z04, 002_re_015000.zip

# Extract the multi-part archive
# On Linux/Mac:
cat 002_re_015000.z* > combined.zip
unzip combined.zip -d checkpoints/

# On Windows (using 7-Zip):
# Right-click 002_re_015000.zip -> 7-Zip -> Extract Here

# The extracted checkpoint should have this structure:
# checkpoints/002_re_015000/
#   ├── unet/
#   │   └── diffusion_pytorch_model.safetensors
#   ├── controlnet/
#   │   └── ...
#   └── hybrid_003_config.json
```

## Quick Start: Inference

```bash
python script/hybrid_controlnet_restoration/run.py \
    --checkpoint checkpoints/002_re_015000 \
    --input_rgb_dir path/to/degraded/images \
    --output_dir path/to/output \
    --denoise_steps 5 \
    --fp16
```

### Key Parameters

| Parameter          | Default  | Description                                                 |
| ------------------ | -------- | ----------------------------------------------------------- |
| `--checkpoint`     | required | Path to checkpoint directory                                |
| `--input_rgb_dir`  | required | Directory containing degraded images                        |
| `--output_dir`     | required | Output directory for restored images                        |
| `--denoise_steps`  | None     | Number of denoising steps (default from pipeline config)    |
| `--ensemble_size`  | 1        | Number of predictions to ensemble                           |
| `--processing_res` | None     | Processing resolution (0 = native, None = pipeline default) |
| `--fp16`           | false    | Use half-precision for faster inference                     |
| `--seed`           | random   | Random seed for reproducibility                             |

### Example

```bash
# Restore images with 5 denoising steps
python script/hybrid_controlnet_restoration/run.py \
    --checkpoint checkpoints/002_re_015000 \
    --input_rgb_dir ./input \
    --output_dir ./output \
    --denoise_steps 5 \
    --fp16

# Results will be saved to:
#   ./output/restored/       (PNG images)
#   ./output/restored_npy/   (NumPy arrays)
```

## Training

### Setup

|                   |                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| **Dataset**       | LAION-Aesthetics subset, ~20,000 images                                                          |
| **Crop size**     | 768 × 768 (random crops)                                                                         |
| **Degradation**   | RealESRGAN second-order pipeline: blur → resize → noise → JPEG (two cascaded stages, on-the-fly) |
| **Iterations**    | 15,000                                                                                           |
| **Batch size**    | 16                                                                                               |
| **Hardware**      | NVIDIA L40S (single GPU)                                                                         |
| **Training time** | ~20 hours                                                                                        |

### Running

To train from scratch:

```bash
# Set environment variables
export BASE_DATA_DIR=/path/to/datasets
export BASE_CKPT_DIR=/path/to/checkpoints

# Start training
python script/hybrid_controlnet_restoration_003/train.py \
    --config config/train_marigold_hybrid_controlnet_restoration_003.yaml
```

See `config/train_marigold_hybrid_controlnet_restoration_003.yaml` for training configuration details.

## Project Structure

```
IRis/
├── marigold/                    # Inference pipelines
│   ├── marigold_hybrid_controlnet_arniqa_003_pipeline.py
│   └── marigold_hybrid_controlnet_arniqa_003_pipeline_patched.py
├── src/
│   ├── trainer/                 # Training code
│   ├── dataset/                 # Dataset loaders
│   ├── ARNIQA/                  # Quality-aware conditioning module
│   └── util/                    # Utilities (losses, metrics, etc.)
├── script/
│   ├── hybrid_controlnet_restoration/
│   │   └── run.py              # Inference script
│   ├── hybrid_controlnet_restoration_003/
│   │   └── train.py            # Training script
│   └── restoration/eval/        # Evaluation scripts
├── config/                      # Training configurations
└── checkpoints/                 # Model checkpoints
```

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{iris2026,
    title={Blind Image Restoration via Dual-Conditioned Latent Diffusion},
    author={Carmelo Calzerano},
    school={University of Pisa},
    year={2026}
}
```

This work builds upon Marigold. Please also cite:

```bibtex
@InProceedings{ke2023repurposing,
    title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
    author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}

@misc{ke2025marigold,
    title={Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis},
    author={Bingxin Ke and Kevin Qu and Tianfu Wang and Nando Metzger and Shengyu Huang and Bo Li and Anton Obukhov and Konrad Schindler},
    year={2025},
    eprint={2505.09358},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License

- **Code**: Apache License 2.0 (see `LICENSE.txt`)
- **Model weights**: OpenRAIL++-M License (see `LICENSE-MODEL.txt`)

## Acknowledgments

This project is based on the [Marigold](https://github.com/prs-eth/Marigold) framework by PRS-ETH.
