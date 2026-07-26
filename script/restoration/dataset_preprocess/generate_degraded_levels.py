#!/usr/bin/env python3
"""
Generate degraded images at controlled severity levels using DiffBIR's
two-stage RealESRGAN degradation pipeline.

Unlike generate_degraded_diffbir.py (which uses the full random ranges from
DiffBIR's training config), this script restricts all random parameters to
narrow bands determined by a discrete severity level (0-4):

    Level 0: Barely perceptible degradation
    Level 1: Mild degradation
    Level 2: Moderate degradation
    Level 3: Heavy degradation
    Level 4: Severe degradation

Each image gets a deterministic per-image seed (base_seed + image_index),
so results are reproducible but each image within a level gets a slightly
different degradation realization (different kernel shape, noise pattern, etc.)
sampled from within the level's parameter band.

The two-stage pipeline is preserved (stage1: blur+resize+noise+JPEG,
stage2: blur+resize+noise+JPEG+sinc), with stage2_scale=1.0 (no
super-resolution downscaling, same output resolution as input).

Usage:
    # Generate level 2 degradation
    python generate_degraded_levels.py \\
        --clean_dir /path/to/clean \\
        --output_dir /path/to/degraded_level2 \\
        --level 2

    # Generate all levels at once (creates subdirs level_0/ ... level_4/)
    python generate_degraded_levels.py \\
        --clean_dir /path/to/clean \\
        --output_dir /path/to/output \\
        --level all

    # With GPU acceleration
    python generate_degraded_levels.py \\
        --clean_dir /path/to/clean \\
        --output_dir /path/to/out \\
        --level 3 --device cuda
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

# Add DiffBIR root to sys.path so we can import its modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..', '..')
DIFFBIR_ROOT = os.path.join(PROJECT_ROOT, 'external', 'DiffBIR')
sys.path.insert(0, DIFFBIR_ROOT)

from diffbir.dataset.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    circular_lowpass_kernel,
)
from diffbir.dataset.diffjpeg import DiffJPEG


# =============================================================================
# Degradation level definitions
#
# Each level constrains the random parameter ranges of the two-stage pipeline.
# The pipeline structure (blur -> resize -> noise -> JPEG) x2 + sinc is
# unchanged; only the numeric ranges are narrowed.
# =============================================================================

DEGRADATION_LEVELS = {
    0: {
        "description": "Barely perceptible",
        # Stage 1
        "blur_sigma": [0.2, 0.5],
        "resize_range": [0.9, 1.0],
        "noise_range": [1, 5],
        "jpeg_range": [85, 95],
        # Stage 2
        "blur_sigma2": [0.2, 0.4],
        "resize_range2": [0.9, 1.0],
        "noise_range2": [1, 4],
        "jpeg_range2": [85, 95],
        # Probabilities (reduced randomness for mild levels)
        "sinc_prob": 0.05,
        "sinc_prob2": 0.05,
        "final_sinc_prob": 0.3,
        "second_blur_prob": 0.3,
        "gaussian_noise_prob": 0.7,
        "gaussian_noise_prob2": 0.7,
        "gray_noise_prob": 0.2,
        "gray_noise_prob2": 0.2,
        "poisson_scale_range": [0.05, 0.5],
        "poisson_scale_range2": [0.05, 0.5],
    },
    1: {
        "description": "Mild",
        "blur_sigma": [0.5, 1.2],
        "resize_range": [0.7, 1.0],
        "noise_range": [5, 12],
        "jpeg_range": [65, 85],
        "blur_sigma2": [0.3, 0.8],
        "resize_range2": [0.7, 1.0],
        "noise_range2": [3, 10],
        "jpeg_range2": [65, 85],
        "sinc_prob": 0.08,
        "sinc_prob2": 0.08,
        "final_sinc_prob": 0.5,
        "second_blur_prob": 0.5,
        "gaussian_noise_prob": 0.6,
        "gaussian_noise_prob2": 0.6,
        "gray_noise_prob": 0.3,
        "gray_noise_prob2": 0.3,
        "poisson_scale_range": [0.05, 1.5],
        "poisson_scale_range2": [0.05, 1.2],
    },
    2: {
        "description": "Moderate",
        "blur_sigma": [1.0, 2.0],
        "resize_range": [0.5, 0.9],
        "noise_range": [10, 20],
        "jpeg_range": [45, 70],
        "blur_sigma2": [0.5, 1.2],
        "resize_range2": [0.5, 0.9],
        "noise_range2": [8, 16],
        "jpeg_range2": [45, 70],
        "sinc_prob": 0.1,
        "sinc_prob2": 0.1,
        "final_sinc_prob": 0.7,
        "second_blur_prob": 0.7,
        "gaussian_noise_prob": 0.5,
        "gaussian_noise_prob2": 0.5,
        "gray_noise_prob": 0.4,
        "gray_noise_prob2": 0.4,
        "poisson_scale_range": [0.5, 2.5],
        "poisson_scale_range2": [0.5, 2.0],
    },
    3: {
        "description": "Heavy",
        "blur_sigma": [1.5, 2.5],
        "resize_range": [0.3, 0.7],
        "noise_range": [18, 28],
        "jpeg_range": [30, 50],
        "blur_sigma2": [0.8, 1.5],
        "resize_range2": [0.4, 0.8],
        "noise_range2": [12, 22],
        "jpeg_range2": [30, 50],
        "sinc_prob": 0.12,
        "sinc_prob2": 0.12,
        "final_sinc_prob": 0.8,
        "second_blur_prob": 0.8,
        "gaussian_noise_prob": 0.5,
        "gaussian_noise_prob2": 0.5,
        "gray_noise_prob": 0.4,
        "gray_noise_prob2": 0.4,
        "poisson_scale_range": [1.0, 3.0],
        "poisson_scale_range2": [0.5, 2.5],
    },
    4: {
        "description": "Severe",
        "blur_sigma": [2.0, 3.0],
        "resize_range": [0.2, 0.5],
        "noise_range": [25, 35],
        "jpeg_range": [20, 40],
        "blur_sigma2": [1.0, 1.5],
        "resize_range2": [0.3, 0.6],
        "noise_range2": [18, 28],
        "jpeg_range2": [20, 40],
        "sinc_prob": 0.15,
        "sinc_prob2": 0.15,
        "final_sinc_prob": 0.9,
        "second_blur_prob": 0.9,
        "gaussian_noise_prob": 0.5,
        "gaussian_noise_prob2": 0.5,
        "gray_noise_prob": 0.4,
        "gray_noise_prob2": 0.4,
        "poisson_scale_range": [1.5, 3.5],
        "poisson_scale_range2": [1.0, 3.0],
    },
}


# =============================================================================
# filter2D — copied from generate_degraded_diffbir.py (originally from
# diffbir/dataset/utils.py lines 162-185)
# =============================================================================

def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D.

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


# =============================================================================
# Kernel generation — adapted from generate_degraded_diffbir.py
# generate_realesrgan_kernels(), using level-specific sigma ranges.
# =============================================================================

def generate_kernels_for_level(level_params: dict):
    """Generate blur kernels with level-constrained sigma ranges.

    Uses the same kernel type distribution as the original RealESRGAN pipeline
    (iso, aniso, generalized, plateau variants) but constrains blur_sigma to
    the level's band.

    Returns:
        kernel1: torch.FloatTensor (21, 21)
        kernel2: torch.FloatTensor (21, 21)
        sinc_kernel: torch.FloatTensor (21, 21)
    """
    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                   'plateau_iso', 'plateau_aniso']
    kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    betag_range = [0.5, 4]
    betap_range = [1, 2]

    blur_sigma = level_params["blur_sigma"]
    blur_sigma2 = level_params["blur_sigma2"]
    sinc_prob = level_params["sinc_prob"]
    sinc_prob2 = level_params["sinc_prob2"]
    final_sinc_prob = level_params["final_sinc_prob"]

    kernel_range = [2 * v + 1 for v in range(3, 11)]  # 7 to 21
    pulse_tensor = torch.zeros(21, 21).float()
    pulse_tensor[10, 10] = 1

    # --- Stage 1 kernel ---
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list, kernel_prob, kernel_size,
            blur_sigma, blur_sigma,
            [-math.pi, math.pi],
            betag_range, betap_range,
            noise_range=None,
        )
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # --- Stage 2 kernel ---
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            kernel_list, kernel_prob, kernel_size,
            blur_sigma2, blur_sigma2,
            [-math.pi, math.pi],
            betag_range, betap_range,
            noise_range=None,
        )
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # --- Final sinc kernel ---
    if np.random.uniform() < final_sinc_prob:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    kernel1 = torch.FloatTensor(kernel)
    kernel2_t = torch.FloatTensor(kernel2)

    return kernel1, kernel2_t, sinc_kernel


# =============================================================================
# Two-stage degradation pipeline — adapted from generate_degraded_diffbir.py
# apply_realesrgan_degradation(), using level-specific parameter ranges.
# =============================================================================

@torch.no_grad()
def apply_degradation_for_level(
    img_tensor: torch.Tensor,
    kernel1: torch.Tensor,
    kernel2: torch.Tensor,
    sinc_kernel: torch.Tensor,
    jpeger: DiffJPEG,
    level_params: dict,
    device: str = 'cpu',
) -> torch.Tensor:
    """Apply two-stage RealESRGAN degradation with level-constrained parameters.

    The pipeline structure is identical to apply_realesrgan_degradation() in
    generate_degraded_diffbir.py. Only the numeric ranges are narrowed to the
    level's band.

    Args:
        img_tensor: Clean image, BCHW float32 [0, 1] RGB, shape (1, 3, H, W).
        kernel1: Stage 1 blur kernel, shape (1, 21, 21).
        kernel2: Stage 2 blur kernel, shape (1, 21, 21).
        sinc_kernel: Final sinc kernel, shape (1, 21, 21).
        jpeger: DiffJPEG module instance.
        level_params: Dict of parameters for this degradation level.
        device: Device for computation.

    Returns:
        Degraded image, BCHW float32 [0, 1] RGB, shape (1, 3, H, W).
    """
    # Extract level-specific ranges
    resize_range = level_params["resize_range"]
    noise_range = level_params["noise_range"]
    jpeg_range = level_params["jpeg_range"]
    gaussian_noise_prob = level_params["gaussian_noise_prob"]
    gray_noise_prob = level_params["gray_noise_prob"]
    poisson_scale_range = level_params["poisson_scale_range"]

    second_blur_prob = level_params["second_blur_prob"]
    resize_range2 = level_params["resize_range2"]
    noise_range2 = level_params["noise_range2"]
    jpeg_range2 = level_params["jpeg_range2"]
    gaussian_noise_prob2 = level_params["gaussian_noise_prob2"]
    gray_noise_prob2 = level_params["gray_noise_prob2"]
    poisson_scale_range2 = level_params["poisson_scale_range2"]

    # Resize probs: for lower levels, favor "keep"; for higher, favor "down"
    # Original: [0.2, 0.7, 0.1] = [up, down, keep]
    # We adjust: lower levels have more "keep", higher levels more "down"
    resize_prob = [0.1, 0.5, 0.4]   # stage 1
    resize_prob2 = [0.1, 0.5, 0.4]  # stage 2

    hq = img_tensor.to(device)
    kernel1 = kernel1.to(device)
    kernel2 = kernel2.to(device)
    sinc_kernel = sinc_kernel.to(device)
    if device != 'cpu' or next(jpeger.parameters()).device.type != 'cpu':
        jpeger = jpeger.to(device)

    ori_h, ori_w = hq.size()[2:4]
    stage2_scale = 1.0  # Same resolution output

    # -------------------- Stage 1 -------------------- #
    # Blur
    out = filter2D(hq, kernel1)

    # Random resize (constrained to level's range)
    updown_type = random.choices(["up", "down", "keep"], resize_prob)[0]
    if updown_type == "up":
        scale = np.random.uniform(1, resize_range[1])
    elif updown_type == "down":
        scale = np.random.uniform(resize_range[0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(out, scale_factor=scale, mode=mode)

    # Noise
    if np.random.uniform() < gaussian_noise_prob:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=noise_range,
            clip=True, rounds=False, gray_prob=gray_noise_prob,
        )
    else:
        out = random_add_poisson_noise_pt(
            out, scale_range=poisson_scale_range,
            gray_prob=gray_noise_prob, clip=True, rounds=False,
        )

    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)

    # -------------------- Stage 2 -------------------- #
    # Blur (conditional)
    if np.random.uniform() < second_blur_prob:
        out = filter2D(out, kernel2)

    # Target size (stage2_scale=1.0 means same as original)
    stage2_h = int(ori_h / stage2_scale)
    stage2_w = int(ori_w / stage2_scale)

    # Random resize
    updown_type = random.choices(["up", "down", "keep"], resize_prob2)[0]
    if updown_type == "up":
        scale = np.random.uniform(1, resize_range2[1])
    elif updown_type == "down":
        scale = np.random.uniform(resize_range2[0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(
        out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode
    )

    # Noise
    if np.random.uniform() < gaussian_noise_prob2:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=noise_range2,
            clip=True, rounds=False, gray_prob=gray_noise_prob2,
        )
    else:
        out = random_add_poisson_noise_pt(
            out, scale_range=poisson_scale_range2,
            gray_prob=gray_noise_prob2, clip=True, rounds=False,
        )

    # Final: JPEG + sinc (random order, same as original)
    if np.random.uniform() < 0.5:
        # resize back + sinc filter + JPEG
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
        out = filter2D(out, sinc_kernel)
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG + resize back + sinc filter
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
        out = filter2D(out, sinc_kernel)

    # Clamp and quantize
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

    return lq


# =============================================================================
# Image I/O and utilities
# =============================================================================

def save_image_from_tensor(tensor: torch.Tensor, path: str):
    """Save CHW or BCHW float32 [0, 1] RGB tensor as PNG."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


def find_images(directory: Path) -> list:
    """Find all image files in directory (non-recursive for flat dirs)."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    files = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(f)
    return files


def set_seed(base_seed: int, index: int):
    """Set deterministic seed for given image index."""
    seed = base_seed + index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Main
# =============================================================================

def process_level(
    level: int,
    clean_dir: Path,
    output_dir: Path,
    image_files: list,
    device: str,
    base_seed: int,
    jpeger: DiffJPEG,
):
    """Process all images for a single degradation level."""
    level_params = DEGRADATION_LEVELS[level]
    level_output = output_dir / f"level_{level}"
    level_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Level {level}: {level_params['description']}")
    print(f"Output: {level_output}")
    print(f"{'='*60}")

    processed = 0
    skipped = 0

    for idx, img_path in enumerate(tqdm(
        image_files, desc=f"Level {level} ({level_params['description']})"
    )):
        dest_file = level_output / img_path.with_suffix('.png').name

        # Skip if already exists
        if dest_file.exists():
            processed += 1
            continue

        try:
            set_seed(base_seed, idx)

            # Load image as RGB float32 [0, 1]
            img = Image.open(str(img_path)).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(
                img_np.transpose(2, 0, 1).copy()
            ).float().unsqueeze(0)  # (1, 3, H, W)

            # Generate kernels with level-constrained sigmas
            kernel1, kernel2, sinc_kernel = generate_kernels_for_level(level_params)
            kernel1 = kernel1.unsqueeze(0)
            kernel2 = kernel2.unsqueeze(0)
            sinc_kernel = sinc_kernel.unsqueeze(0)

            # Apply degradation
            lq = apply_degradation_for_level(
                img_tensor, kernel1, kernel2, sinc_kernel,
                jpeger, level_params, device=device,
            )

            # Save
            save_image_from_tensor(lq, str(dest_file))
            processed += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate degraded images at controlled severity levels "
                    "using DiffBIR's two-stage RealESRGAN pipeline."
    )
    parser.add_argument(
        "--clean_dir", type=str, required=True,
        help="Directory containing clean source images",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory. When --level is a number, degraded images are "
             "saved directly here. When --level all, subdirs level_0/...level_4/ "
             "are created.",
    )
    parser.add_argument(
        "--level", type=str, required=True,
        help="Degradation level: 0-4 or 'all' to generate all levels",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for computation (default: cpu)",
    )

    args = parser.parse_args()

    # Parse level argument
    if args.level.lower() == "all":
        levels = list(range(5))
    else:
        try:
            lvl = int(args.level)
            if lvl < 0 or lvl > 4:
                parser.error("Level must be 0-4 or 'all'")
            levels = [lvl]
        except ValueError:
            parser.error("Level must be 0-4 or 'all'")

    clean_dir = Path(args.clean_dir)
    output_dir = Path(args.output_dir)

    if not clean_dir.is_dir():
        print(f"ERROR: Clean directory not found: {clean_dir}")
        sys.exit(1)

    image_files = find_images(clean_dir)
    if not image_files:
        print(f"ERROR: No images found in {clean_dir}")
        sys.exit(1)

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Clean dir: {clean_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Levels: {levels}")
    print(f"Images: {len(image_files)}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # Initialize DiffJPEG
    jpeger = DiffJPEG(differentiable=False).to(device)

    # If single level and output_dir specified, save directly there
    # (don't create level_X subdir). If 'all', create subdirs.
    total_processed = 0
    total_skipped = 0

    for level in levels:
        if len(levels) == 1:
            # Single level: save directly to output_dir
            level_params = DEGRADATION_LEVELS[level]
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nLevel {level}: {level_params['description']}")
            print(f"Output: {output_dir}")

            processed = 0
            skipped = 0

            for idx, img_path in enumerate(tqdm(
                image_files,
                desc=f"Level {level} ({level_params['description']})"
            )):
                dest_file = output_dir / img_path.with_suffix('.png').name
                if dest_file.exists():
                    processed += 1
                    continue

                try:
                    set_seed(args.seed, idx)
                    img = Image.open(str(img_path)).convert("RGB")
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(
                        img_np.transpose(2, 0, 1).copy()
                    ).float().unsqueeze(0)

                    kernel1, kernel2, sinc_kernel = generate_kernels_for_level(
                        level_params
                    )
                    kernel1 = kernel1.unsqueeze(0)
                    kernel2 = kernel2.unsqueeze(0)
                    sinc_kernel = sinc_kernel.unsqueeze(0)

                    lq = apply_degradation_for_level(
                        img_tensor, kernel1, kernel2, sinc_kernel,
                        jpeger, level_params, device=device,
                    )
                    save_image_from_tensor(lq, str(dest_file))
                    processed += 1
                except Exception as e:
                    print(f"\nError processing {img_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    skipped += 1

            total_processed += processed
            total_skipped += skipped
        else:
            # Multiple levels: use subdirs
            p, s = process_level(
                level, clean_dir, output_dir, image_files,
                device, args.seed, jpeger,
            )
            total_processed += p
            total_skipped += s

    # Save metadata
    metadata = {
        "levels_generated": levels,
        "level_definitions": {
            str(k): {
                "description": v["description"],
                "blur_sigma_stage1": v["blur_sigma"],
                "blur_sigma_stage2": v["blur_sigma2"],
                "resize_range_stage1": v["resize_range"],
                "resize_range_stage2": v["resize_range2"],
                "noise_range_stage1": v["noise_range"],
                "noise_range_stage2": v["noise_range2"],
                "jpeg_range_stage1": v["jpeg_range"],
                "jpeg_range_stage2": v["jpeg_range2"],
            }
            for k, v in DEGRADATION_LEVELS.items()
            if k in levels
        },
        "seed": args.seed,
        "stage2_scale": 1.0,
        "pipeline": "two-stage RealESRGAN (from DiffBIR)",
        "num_images": len(image_files),
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "source_dir": str(clean_dir),
    }
    metadata_path = output_dir / "degradation_levels_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nComplete: {total_processed} processed, {total_skipped} skipped")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
