#!/usr/bin/env python3
"""
Test VAE Reconstruction Baseline

Measures the theoretical upper bound for restoration quality by testing
clean image → VAE encode → VAE decode → compare with original.

This establishes the maximum achievable PSNR due to VAE compression.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from marigold import MarigoldRestorationPipeline
from src.util.metric import MetricTracker


def load_validation_images(val_dir, max_images=100):
    """Load validation images"""
    val_dir = Path(val_dir)
    image_files = sorted(list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg")))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {val_dir}")
    
    # Limit to max_images
    image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} validation images")
    return image_files


def test_vae_reconstruction(pipeline, image_path, device):
    """Test VAE encode-decode on a single image"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Convert to tensor [C, H, W] in [-1, 1]
    image_np = np.array(image).astype(np.float32) / 255.0  # [0, 1]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [C, H, W]
    image_tensor = image_tensor * 2.0 - 1.0  # [-1, 1]
    image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
    
    # Match pipeline dtype (FP16 or FP32)
    if pipeline.vae.dtype == torch.float16:
        image_tensor = image_tensor.half()
    
    image_tensor = image_tensor.to(device)
    
    # Encode to latent
    with torch.no_grad():
        latent = pipeline.encode_rgb(image_tensor)
    
    # Decode back to RGB
    with torch.no_grad():
        reconstructed = pipeline.decode_rgb(latent)
    
    # Convert back to [0, 1] range and to FP32 for metrics
    reconstructed = (reconstructed + 1.0) / 2.0
    image_tensor = (image_tensor + 1.0) / 2.0
    
    # Clamp to valid range
    reconstructed = torch.clamp(reconstructed, 0.0, 1.0)
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    
    # Convert to FP32 for metric calculation
    reconstructed = reconstructed.float()
    image_tensor = image_tensor.float()
    
    return image_tensor, reconstructed


def main():
    print("=" * 80)
    print("VAE RECONSTRUCTION BASELINE TEST")
    print("=" * 80)
    print()
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dir = project_root / "data" / "laion" / "val" / "clean"
    max_images = 100  # Test on 1000 images for robust statistics
    
    print(f"Device: {device}")
    print(f"Validation directory: {val_dir}")
    print(f"Max images: {max_images}")
    print()
    
    # Load pipeline
    print("Loading pipeline...")
    pipeline = MarigoldRestorationPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        variant="fp16" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipeline = pipeline.to(device)
    print("✓ Pipeline loaded")
    print()
    
    # Load validation images
    print("Loading validation images...")
    image_files = load_validation_images(val_dir, max_images)
    print()
    
    # Initialize metric tracker
    from src.util.metric import MetricTracker, psnr, ssim, lpips_alex
    
    metric_funcs = [psnr, ssim, lpips_alex]
    metric_tracker = MetricTracker(*[f.__name__ for f in metric_funcs])
    
    # Test VAE reconstruction on all images
    print("Testing VAE reconstruction...")
    print()
    
    # Debug first image
    print("DEBUG: Testing first image...")
    first_image = image_files[0]
    original, reconstructed = test_vae_reconstruction(pipeline, first_image, device)
    print(f"  Original shape: {original.shape}, range: [{original.min():.4f}, {original.max():.4f}]")
    print(f"  Reconstructed shape: {reconstructed.shape}, range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    print(f"  MSE: {F.mse_loss(reconstructed, original).item():.6f}")
    print(f"  Are they identical? {torch.allclose(original, reconstructed, atol=1e-6)}")
    print()
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get original and reconstructed
            original, reconstructed = test_vae_reconstruction(pipeline, image_path, device)
            
            # Compute metrics
            for metric_func in metric_funcs:
                metric_value = metric_func(reconstructed, original, None)
                metric_tracker.update(metric_func.__name__, metric_value)
        
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    # Get results
    results = metric_tracker.result()
    
    print()
    print("=" * 80)
    print("VAE RECONSTRUCTION BASELINE RESULTS")
    print("=" * 80)
    print()
    print(f"Images tested: {len(image_files)}")
    print()
    print("Metrics (Original → VAE Encode → VAE Decode → Compare):")
    print("-" * 80)
    
    for metric_name, metric_value in results.items():
        if "psnr" in metric_name.lower():
            print(f"  {metric_name:20s}: {metric_value:8.2f} dB")
        else:
            print(f"  {metric_name:20s}: {metric_value:8.4f}")
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("This PSNR represents the THEORETICAL UPPER BOUND for restoration quality.")
    print("Since we train in latent space, we cannot exceed this value.")
    print()
    
    psnr = results.get("psnr", 0)
    if psnr > 0:
        print(f"VAE Baseline PSNR: {psnr:.2f} dB")
        print()
        print("Expected restoration performance:")
        print(f"  - Excellent:  {psnr - 2:.1f} - {psnr:.1f} dB  (within 2 dB of baseline)")
        print(f"  - Good:       {psnr - 5:.1f} - {psnr - 2:.1f} dB  (2-5 dB below baseline)")
        print(f"  - Acceptable: {psnr - 8:.1f} - {psnr - 5:.1f} dB  (5-8 dB below baseline)")
        print(f"  - Poor:       < {psnr - 8:.1f} dB  (>8 dB below baseline)")
    
    print()
    print("=" * 80)
    print("✓ Test complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
