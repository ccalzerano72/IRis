# Generate degraded images using DiffBIR's degradation pipelines.
# Supports two modes:
#   - codeformer: Single-stage (blur -> downsample -> noise -> JPEG) using numpy ops.
#     Replicates CodeformerDataset.__getitem__ from diffbir/dataset/codeformer.py.
#   - realesrgan: Two-stage Real-ESRGAN degradation using PyTorch ops.
#     Replicates RealESRGANDataset + RealESRGANBatchTransform from
#     diffbir/dataset/realesrgan.py and diffbir/dataset/batch_transform.py.
#
# Default parameters match DiffBIR's training configs:
#   - codeformer: configs/train/train_stage1.yaml
#   - realesrgan: configs/train/train_stage2_v2.1.yaml
#
# Usage:
#   python generate_degraded_diffbir.py --clean_dir /path/to/clean --output_dir /path/to/out
#   python generate_degraded_diffbir.py --mode codeformer --clean_dir ... --output_dir ...

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
    random_add_gaussian_noise,
    random_add_jpg_compression,
    circular_lowpass_kernel,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from diffbir.dataset.diffjpeg import DiffJPEG


# Copied from: diffbir/dataset/utils.py lines 162-185
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

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
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


# ============================================================================
# Image I/O helpers
# ============================================================================

def load_image_bgr(path: str) -> np.ndarray:
    """Load image as BGR float32 [0, 1] HWC array (same as DiffBIR convention)."""
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    # RGB -> BGR (DiffBIR internal convention)
    return img_np[..., ::-1].copy()


def save_image_from_bgr(img_bgr: np.ndarray, path: str):
    """Save BGR float32 [0, 1] HWC array as PNG."""
    # BGR -> RGB
    img_rgb = img_bgr[..., ::-1]
    img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_rgb).save(path)


def save_image_from_tensor(tensor: torch.Tensor, path: str):
    """Save CHW float32 [0, 1] RGB tensor as PNG."""
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


# ============================================================================
# Codeformer degradation (numpy-based, single-stage)
# Copied from: diffbir/dataset/codeformer.py CodeformerDataset.__getitem__
# ============================================================================

def apply_codeformer_degradation(
    img_bgr: np.ndarray,
    blur_kernel_size: int = 41,
    kernel_list: list = None,
    kernel_prob: list = None,
    blur_sigma: list = None,
    downsample_range: list = None,
    noise_range: list = None,
    jpeg_range: list = None,
) -> np.ndarray:
    """Apply single-stage Codeformer degradation pipeline.

    Pipeline: blur -> downsample -> gaussian noise -> JPEG -> resize back.
    All operations use numpy/cv2, matching codeformer.py exactly.

    Args:
        img_bgr: Clean image, BGR float32 [0, 1], shape (H, W, 3).
        blur_kernel_size: Kernel size for blur. Default 41 (train_stage1.yaml).
        kernel_list: Kernel types. Default ['iso', 'aniso'].
        kernel_prob: Kernel probabilities. Default [0.5, 0.5].
        blur_sigma: Sigma range [min, max]. Default [0.1, 12].
        downsample_range: Downsample scale range [min, max]. Default [1, 12].
        noise_range: Gaussian noise sigma range [min, max]. Default [0, 15].
        jpeg_range: JPEG quality range [min, max]. Default [30, 100].

    Returns:
        Degraded image, BGR float32 [0, 1], shape (H, W, 3).
    """
    if kernel_list is None:
        kernel_list = ['iso', 'aniso']
    if kernel_prob is None:
        kernel_prob = [0.5, 0.5]
    if blur_sigma is None:
        blur_sigma = [0.1, 12]
    if downsample_range is None:
        downsample_range = [1, 12]
    if noise_range is None:
        noise_range = [0, 15]
    if jpeg_range is None:
        jpeg_range = [30, 100]

    h, w, _ = img_bgr.shape

    # blur
    kernel = random_mixed_kernels(
        kernel_list,
        kernel_prob,
        blur_kernel_size,
        blur_sigma,
        blur_sigma,
        [-math.pi, math.pi],
        noise_range=None,
    )
    img_lq = cv2.filter2D(img_bgr, -1, kernel)

    # downsample
    scale = np.random.uniform(downsample_range[0], downsample_range[1])
    img_lq = cv2.resize(
        img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
    )

    # noise
    if noise_range is not None:
        img_lq = random_add_gaussian_noise(img_lq, noise_range)

    # jpeg compression
    if jpeg_range is not None:
        img_lq = random_add_jpg_compression(img_lq, jpeg_range)

    # resize to original size
    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

    return img_lq


# ============================================================================
# RealESRGAN degradation (PyTorch-based, two-stage)
# Kernel generation from: diffbir/dataset/realesrgan.py RealESRGANDataset.__getitem__
# Degradation pipeline from: diffbir/dataset/batch_transform.py RealESRGANBatchTransform.__call__
# Default params from: configs/train/train_stage2_v2.1.yaml
# ============================================================================

def generate_realesrgan_kernels(
    # Stage 1 kernel params
    kernel_list: list = None,
    kernel_prob: list = None,
    blur_sigma: list = None,
    betag_range: list = None,
    betap_range: list = None,
    sinc_prob: float = 0.1,
    # Stage 2 kernel params
    kernel_list2: list = None,
    kernel_prob2: list = None,
    blur_sigma2: list = None,
    betag_range2: list = None,
    betap_range2: list = None,
    sinc_prob2: float = 0.1,
    # Final sinc
    final_sinc_prob: float = 0.8,
    # Optional pre-created pulse tensor (avoids torch.zeros allocation each call)
    pulse_tensor: torch.Tensor = None,
):
    """Generate blur kernels for two-stage RealESRGAN degradation.

    Copied from RealESRGANDataset.__getitem__ (realesrgan.py lines 157-220).

    Returns:
        kernel1: torch.FloatTensor (21, 21)
        kernel2: torch.FloatTensor (21, 21)
        sinc_kernel: torch.FloatTensor (21, 21)
    """
    if kernel_list is None:
        kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                       'plateau_iso', 'plateau_aniso']
    if kernel_prob is None:
        kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    if blur_sigma is None:
        blur_sigma = [0.2, 3]
    if betag_range is None:
        betag_range = [0.5, 4]
    if betap_range is None:
        betap_range = [1, 2]
    if kernel_list2 is None:
        kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                        'plateau_iso', 'plateau_aniso']
    if kernel_prob2 is None:
        kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    if blur_sigma2 is None:
        blur_sigma2 = [0.2, 1.5]
    if betag_range2 is None:
        betag_range2 = [0.5, 4]
    if betap_range2 is None:
        betap_range2 = [1, 2]

    # kernel size ranges from 7 to 21 (hard-coded in realesrgan.py)
    kernel_range = [2 * v + 1 for v in range(3, 11)]
    # Use pre-created pulse tensor if provided, otherwise create one
    if pulse_tensor is None:
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
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma,
            [-math.pi, math.pi],
            betag_range,
            betap_range,
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
            kernel_list2,
            kernel_prob2,
            kernel_size,
            blur_sigma2,
            blur_sigma2,
            [-math.pi, math.pi],
            betag_range2,
            betap_range2,
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
    kernel2 = torch.FloatTensor(kernel2)

    return kernel1, kernel2, sinc_kernel


@torch.no_grad()
def apply_realesrgan_degradation(
    img_tensor: torch.Tensor,
    kernel1: torch.Tensor,
    kernel2: torch.Tensor,
    sinc_kernel: torch.Tensor,
    jpeger: DiffJPEG,
    device: str = 'cpu',
    # Stage 1 params (from train_stage2_v2.1.yaml batch_transform)
    resize_prob: list = None,
    resize_range: list = None,
    gaussian_noise_prob: float = 0.5,
    noise_range: list = None,
    poisson_scale_range: list = None,
    gray_noise_prob: float = 0.4,
    jpeg_range: list = None,
    # Stage 2 params
    second_blur_prob: float = 0.8,
    stage2_scale: float = 4,
    resize_prob2: list = None,
    resize_range2: list = None,
    gaussian_noise_prob2: float = 0.5,
    noise_range2: list = None,
    poisson_scale_range2: list = None,
    gray_noise_prob2: float = 0.4,
    jpeg_range2: list = None,
) -> torch.Tensor:
    """Apply two-stage RealESRGAN degradation pipeline using PyTorch ops.

    Copied from RealESRGANBatchTransform.__call__ (batch_transform.py lines 142-250).
    Adapted for single-image processing (batch_size=1).

    Args:
        img_tensor: Clean image, BCHW float32 [0, 1] RGB, shape (1, 3, H, W).
        kernel1: Stage 1 blur kernel, shape (1, 21, 21).
        kernel2: Stage 2 blur kernel, shape (1, 21, 21).
        sinc_kernel: Final sinc kernel, shape (1, 21, 21).
        jpeger: DiffJPEG module instance.
        device: Device for computation.

    Returns:
        Degraded image, BCHW float32 [0, 1] RGB, shape (1, 3, H, W).
    """
    if resize_prob is None:
        resize_prob = [0.2, 0.7, 0.1]
    if resize_range is None:
        resize_range = [0.15, 1.5]
    if noise_range is None:
        noise_range = [1, 30]
    if poisson_scale_range is None:
        poisson_scale_range = [0.05, 3]
    if jpeg_range is None:
        jpeg_range = [30, 95]
    if resize_prob2 is None:
        resize_prob2 = [0.3, 0.4, 0.3]
    if resize_range2 is None:
        resize_range2 = [0.3, 1.2]
    if noise_range2 is None:
        noise_range2 = [1, 25]
    if poisson_scale_range2 is None:
        poisson_scale_range2 = [0.05, 2.5]
    if jpeg_range2 is None:
        jpeg_range2 = [30, 95]

    hq = img_tensor.to(device)
    kernel1 = kernel1.to(device)
    kernel2 = kernel2.to(device)
    sinc_kernel = sinc_kernel.to(device)
    # Only move jpeger if not already on target device (avoids redundant
    # nn.Module.to() iteration in DataLoader workers where device='cpu')
    if device != 'cpu' or next(jpeger.parameters()).device.type != 'cpu':
        jpeger = jpeger.to(device)

    ori_h, ori_w = hq.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(hq, kernel1)

    # random resize
    updown_type = random.choices(["up", "down", "keep"], resize_prob)[0]
    if updown_type == "up":
        scale = np.random.uniform(1, resize_range[1])
    elif updown_type == "down":
        scale = np.random.uniform(resize_range[0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(out, scale_factor=scale, mode=mode)

    # add noise
    if np.random.uniform() < gaussian_noise_prob:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=noise_range,
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=poisson_scale_range,
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
        )

    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < second_blur_prob:
        out = filter2D(out, kernel2)

    # stage2 target size
    stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)

    # random resize
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

    # add noise
    if np.random.uniform() < gaussian_noise_prob2:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=noise_range2,
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob2,
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=poisson_scale_range2,
            gray_prob=gray_noise_prob2,
            clip=True,
            rounds=False,
        )

    # JPEG compression + the final sinc filter
    # Two orders (randomly chosen, as in batch_transform.py):
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
        out = filter2D(out, sinc_kernel)

    # clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

    return lq


# ============================================================================
# Main
# ============================================================================

def find_images(directory: Path) -> list:
    """Find all image files in directory recursively."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files_set = set()
    for ext in image_extensions:
        image_files_set.update(directory.rglob(f'*{ext}'))
        image_files_set.update(directory.rglob(f'*{ext.upper()}'))
    return sorted(list(image_files_set))


def random_crop_tensor(img_tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Take a random square crop from a CHW float32 tensor.

    Args:
        img_tensor: Tensor of shape (C, H, W).
        crop_size: Side length of the square crop in pixels.

    Returns:
        Cropped tensor of shape (C, crop_size, crop_size).

    Raises:
        ValueError: If either spatial dimension is smaller than crop_size.
    """
    _, h, w = img_tensor.shape
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Image ({h}x{w}) is smaller than crop_size={crop_size}. "
            "Skipping this image."
        )
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return img_tensor[:, top:top + crop_size, left:left + crop_size].clone()


def set_seed(base_seed: int, index: int, seed_offset: int = 0):
    """Set deterministic seed for given image index."""
    seed = base_seed + index + seed_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Generate degraded images using DiffBIR degradation pipelines"
    )
    parser.add_argument(
        "--mode", type=str, default="realesrgan",
        choices=["codeformer", "realesrgan"],
        help="Degradation mode. 'codeformer' = single-stage (numpy), "
             "'realesrgan' = two-stage (PyTorch). Default: realesrgan"
    )
    parser.add_argument(
        "--clean_dir", type=str, required=True,
        help="Directory containing clean images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save degraded images"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed for deterministic generation. Default: 42"
    )
    parser.add_argument(
        "--seed_offset", type=int, default=0,
        help="Offset added to seed (for generating different degradation sets)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for RealESRGAN mode. Default: cpu"
    )
    # RealESRGAN-specific: stage2_scale controls the downscale factor
    # in the second degradation stage. Default 4 means 4x downscale.
    # Set to 1 for same-resolution degradation (no SR, just restoration).
    parser.add_argument(
        "--stage2_scale", type=float, default=1.0,
        help="RealESRGAN stage2 downscale factor. "
             "Use 1.0 for same-resolution restoration (no SR). "
             "Use 4.0 for 4x super-resolution degradation. Default: 1.0"
    )
    parser.add_argument(
        "--crop_size", type=int, default=None,
        help="If set, take a random square crop of this size (in pixels) from "
             "each image before applying degradation. Images smaller than "
             "crop_size are skipped. Example: --crop_size 768"
    )
    parser.add_argument(
        "--save_clean_dir", type=str, default=None,
        help="If set, save the corresponding clean crop (or full image when "
             "--crop_size is not used) to this directory, mirroring the same "
             "relative path as the degraded output. Useful for thesis figure pairs."
    )

    args = parser.parse_args()

    clean_path = Path(args.clean_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clean_save_path = Path(args.save_clean_dir) if args.save_clean_dir else None
    if clean_save_path:
        clean_save_path.mkdir(parents=True, exist_ok=True)

    image_files = find_images(clean_path)
    print(f"Found {len(image_files)} images in {args.clean_dir}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")
    if args.crop_size:
        print(f"Crop size: {args.crop_size}x{args.crop_size} (random crop)")
    if clean_save_path:
        print(f"Clean crops output: {args.save_clean_dir}")
    print(f"Seed: {args.seed}, offset: {args.seed_offset}")

    if len(image_files) == 0:
        print("No images found, exiting")
        return

    # Initialize DiffJPEG for realesrgan mode
    jpeger = None
    if args.mode == "realesrgan":
        jpeger = DiffJPEG(differentiable=False)
        if args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        jpeger = jpeger.to(device)
        print(f"Device: {device}")

    processed = 0
    skipped = 0

    for idx, img_path in enumerate(tqdm(image_files, desc="Generating degraded images")):
        try:
            set_seed(args.seed, idx, args.seed_offset)

            if args.mode == "codeformer":
                # Load as BGR float32 [0, 1] (Codeformer convention)
                img_bgr = load_image_bgr(str(img_path))

                # Apply codeformer degradation
                img_lq = apply_codeformer_degradation(img_bgr)

                # Save (convert BGR -> RGB internally)
                rel_path = img_path.relative_to(clean_path)
                dest_file = output_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                save_image_from_bgr(img_lq, str(dest_file))

            elif args.mode == "realesrgan":
                # Load as RGB float32 [0, 1]
                img = Image.open(str(img_path)).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                # HWC -> CHW tensor
                img_tensor = torch.from_numpy(
                    img_np.transpose(2, 0, 1).copy()
                ).float()

                # Optional random crop (applied before degradation, same as training)
                if args.crop_size is not None:
                    img_tensor = random_crop_tensor(img_tensor, args.crop_size)

                # Add batch dim -> (1, 3, H, W)
                img_batch = img_tensor.unsqueeze(0)

                # Generate kernels
                kernel1, kernel2, sinc_kernel = generate_realesrgan_kernels()

                # Add batch dim to kernels: (21, 21) -> (1, 21, 21)
                kernel1 = kernel1.unsqueeze(0)
                kernel2 = kernel2.unsqueeze(0)
                sinc_kernel = sinc_kernel.unsqueeze(0)

                # Apply degradation
                lq = apply_realesrgan_degradation(
                    img_batch, kernel1, kernel2, sinc_kernel, jpeger,
                    device=device,
                    stage2_scale=args.stage2_scale,
                )

                # Save degraded image
                rel_path = img_path.relative_to(clean_path)
                dest_file = output_path / rel_path.with_suffix('.png')
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                save_image_from_tensor(lq, str(dest_file))

                # Optionally save the corresponding clean crop
                if clean_save_path is not None:
                    clean_dest = clean_save_path / rel_path.with_suffix('.png')
                    clean_dest.parent.mkdir(parents=True, exist_ok=True)
                    save_image_from_tensor(img_tensor.unsqueeze(0), str(clean_dest))

            processed += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1

    # Save metadata
    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "seed_offset": args.seed_offset,
        "crop_size": args.crop_size,
        "save_clean_dir": args.save_clean_dir,
        "num_processed": processed,
        "num_skipped": skipped,
        "source_dir": str(args.clean_dir),
    }
    if args.mode == "codeformer":
        metadata["params"] = {
            "blur_kernel_size": 41,
            "kernel_list": ["iso", "aniso"],
            "kernel_prob": [0.5, 0.5],
            "blur_sigma": [0.1, 12],
            "downsample_range": [1, 12],
            "noise_range": [0, 15],
            "jpeg_range": [30, 100],
        }
    elif args.mode == "realesrgan":
        metadata["params"] = {
            "stage2_scale": args.stage2_scale,
            "stage1": {
                "kernel_list": ["iso", "aniso", "generalized_iso",
                                "generalized_aniso", "plateau_iso", "plateau_aniso"],
                "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                "blur_sigma": [0.2, 3],
                "sinc_prob": 0.1,
                "resize_prob": [0.2, 0.7, 0.1],
                "resize_range": [0.15, 1.5],
                "gaussian_noise_prob": 0.5,
                "noise_range": [1, 30],
                "poisson_scale_range": [0.05, 3],
                "gray_noise_prob": 0.4,
                "jpeg_range": [30, 95],
            },
            "stage2": {
                "second_blur_prob": 0.8,
                "kernel_list": ["iso", "aniso", "generalized_iso",
                                "generalized_aniso", "plateau_iso", "plateau_aniso"],
                "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
                "blur_sigma": [0.2, 1.5],
                "sinc_prob": 0.1,
                "final_sinc_prob": 0.8,
                "resize_prob": [0.3, 0.4, 0.3],
                "resize_range": [0.3, 1.2],
                "gaussian_noise_prob": 0.5,
                "noise_range": [1, 25],
                "poisson_scale_range": [0.05, 2.5],
                "gray_noise_prob": 0.4,
                "jpeg_range": [30, 95],
            },
        }

    metadata_path = output_path / "degradation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nComplete: {processed} processed, {skipped} skipped")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
