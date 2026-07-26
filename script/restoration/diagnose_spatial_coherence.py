# Spatial Coherence Diagnostic Script
# 
# This script compares FINAL RESULTS of different inference step configurations
# to verify the hypothesis: "quality degrades with more steps"
#
# For each step count (1, 2, 5, 10, 20, 50), it runs a COMPLETE inference
# and measures the FINAL output quality.
#
# Metrics computed:
# 1. Gradient magnitude (spatial coherence)
# 2. Frequency spectrum analysis (high-frequency content)
# 3. Local variance (texture preservation)
# 4. Edge strength (structural preservation)
# 5. Perceptual metrics (PSNR, SSIM)
#
# Usage:
#   python script/restoration/diagnose_spatial_coherence.py \
#       --checkpoint checkpoints/013_best_009000/latest \
#       --input external/test_images/DIV2K_valid_512/degraded_multi/0803.png \
#       --clean external/test_images/DIV2K_valid_512/clean/0803.png \
#       --output_dir output/coherence_analysis

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
import json

from marigold import MarigoldRestorationPipeline

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Supported scheduler types
SCHEDULER_CHOICES = ["ddim", "lcm", "heun"]


def load_pipeline(checkpoint_path: str, base_model: str, dtype: torch.dtype, scheduler_type: str = "ddim"):
    """Load pipeline"""
    from diffusers import UNet2DConditionModel, DDIMScheduler, LCMScheduler, HeunDiscreteScheduler
    
    checkpoint_path_obj = Path(checkpoint_path)
    unet_checkpoint_path = checkpoint_path_obj / "unet"
    
    if unet_checkpoint_path.exists():
        logging.info(f"Loading base model from: {base_model}")
        pipe = MarigoldRestorationPipeline.from_pretrained(base_model, torch_dtype=dtype)
        
        logging.info(f"Loading trained U-Net from: {unet_checkpoint_path}")
        pipe.unet = UNet2DConditionModel.from_pretrained(unet_checkpoint_path, torch_dtype=dtype)
    else:
        pipe = MarigoldRestorationPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)
    
    # Fix scheduler config based on scheduler_type
    orig_type = type(pipe.scheduler).__name__
    if scheduler_type == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
        )
    elif scheduler_type == "lcm":
        pipe.scheduler = LCMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
        )
    elif scheduler_type == "heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose from: {SCHEDULER_CHOICES}")
    
    new_type = type(pipe.scheduler).__name__
    logging.info(f"Scheduler: {orig_type} → {new_type}")
    
    # Load restoration config (inference-relevant settings like zero_mean_latents)
    _load_restoration_config(pipe, checkpoint_path_obj)
    
    return pipe


def _load_restoration_config(pipe, checkpoint_path: Path):
    """
    Load restoration config from checkpoint if available.
    
    Args:
        pipe: The pipeline to configure
        checkpoint_path: Path to the checkpoint directory
    """
    import json
    
    config_path = checkpoint_path / "restoration_config.json"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Apply zero_mean_latents setting
        zero_mean = config.get("zero_mean_latents", False)
        pipe.set_zero_mean_latents(zero_mean)
        
        logging.info(f"✓ Loaded restoration config: zero_mean_latents={zero_mean}")
    else:
        # Old checkpoint without restoration_config.json - use defaults
        logging.info("No restoration_config.json found (old checkpoint), using defaults")
        pipe.set_zero_mean_latents(False)


def compute_gradient_magnitude(image: np.ndarray) -> float:
    """Compute mean gradient magnitude as measure of spatial coherence"""
    if image.ndim == 3:
        gray = np.dot(image, [0.299, 0.587, 0.114])
    else:
        gray = image
    
    grad_x = ndimage.sobel(gray, axis=1)
    grad_y = ndimage.sobel(gray, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    return float(np.mean(grad_mag))


def compute_frequency_spectrum(image: np.ndarray) -> dict:
    """Compute frequency spectrum analysis"""
    if image.ndim == 3:
        gray = np.dot(image, [0.299, 0.587, 0.114])
    else:
        gray = image
    
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    max_r = min(center_x, center_y)
    r_bins = np.linspace(0, max_r, 50)
    
    radial_profile = []
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(mask):
            radial_profile.append(np.mean(magnitude_spectrum[mask]))
        else:
            radial_profile.append(0)
    
    low_freq = np.mean(radial_profile[:10])
    mid_freq = np.mean(radial_profile[10:25])
    high_freq = np.mean(radial_profile[25:])
    
    return {
        'low_freq': float(low_freq),
        'mid_freq': float(mid_freq),
        'high_freq': float(high_freq),
        'total_energy': float(np.sum(magnitude_spectrum)),
        'radial_profile': [float(x) for x in radial_profile]
    }


def compute_local_variance(image: np.ndarray, window_size: int = 5) -> float:
    """Compute local variance as measure of texture preservation"""
    if image.ndim == 3:
        gray = np.dot(image, [0.299, 0.587, 0.114])
    else:
        gray = image
    
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    local_mean = ndimage.convolve(gray, kernel, mode='reflect')
    local_mean_sq = ndimage.convolve(gray**2, kernel, mode='reflect')
    local_var = local_mean_sq - local_mean**2
    
    return float(np.mean(local_var))


def compute_edge_strength(image: np.ndarray) -> float:
    """Compute edge strength using Canny edge detector"""
    if image.ndim == 3:
        gray = np.dot(image, [0.299, 0.587, 0.114])
        gray_uint8 = (gray * 255).astype(np.uint8)
    else:
        gray_uint8 = (image * 255).astype(np.uint8)
    
    edges = cv2.Canny(gray_uint8, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    return float(edge_density)


def compute_perceptual_metrics(pred_image: np.ndarray, clean_image: np.ndarray) -> dict:
    """Compute perceptual metrics against clean reference"""
    if pred_image.shape != clean_image.shape:
        pred_resized = cv2.resize(pred_image, (clean_image.shape[1], clean_image.shape[0]))
    else:
        pred_resized = pred_image
    
    if pred_resized.ndim == 3 and clean_image.ndim == 3:
        ssim_val = ssim(pred_resized, clean_image, channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5)
    else:
        if pred_resized.ndim == 3:
            pred_gray = np.dot(pred_resized, [0.299, 0.587, 0.114])
        else:
            pred_gray = pred_resized
            
        if clean_image.ndim == 3:
            clean_gray = np.dot(clean_image, [0.299, 0.587, 0.114])
        else:
            clean_gray = clean_image
            
        ssim_val = ssim(pred_gray, clean_gray, data_range=1.0, gaussian_weights=True, sigma=1.5)
    
    mse = np.mean((pred_resized - clean_image) ** 2)
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 20 * np.log10(1.0 / np.sqrt(mse))
    
    return {
        'ssim': float(ssim_val),
        'psnr': float(psnr_val),
        'mse': float(mse)
    }


def run_single_inference(
    pipe: MarigoldRestorationPipeline,
    rgb_latent: torch.Tensor,
    text_embed: torch.Tensor,
    num_steps: int,
    device: torch.device,
    seed: int = 42,
    feature_rescaling: bool = True,
) -> torch.Tensor:
    """Run a complete inference with specified number of steps
    
    Args:
        pipe: The restoration pipeline (zero_mean setting comes from pipeline config)
        rgb_latent: Encoded degraded image latent
        text_embed: Text embedding
        num_steps: Number of denoising steps
        device: Torch device
        seed: Random seed for reproducibility
        feature_rescaling: If True, rescale latent std at each step to match initial std
    """
    
    # Initialize with SAME random noise for fair comparison
    generator = torch.Generator(device=device).manual_seed(seed)
    target_latent = torch.randn(rgb_latent.shape, device=device, dtype=pipe.dtype, generator=generator)
    
    # Save initial std for feature rescaling
    initial_std = target_latent.std()
    
    # Zero-mean: center conditioning latent (done once) - uses pipeline config
    if pipe.zero_mean_latents:
        cond_mean = rgb_latent.mean(dim=[2, 3], keepdim=True)
        rgb_latent_centered = rgb_latent - cond_mean
    else:
        rgb_latent_centered = rgb_latent
    
    # Set timesteps for this run
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # Complete denoising loop
    for t in timesteps:
        # Zero-mean: center noisy latent before UNet - uses pipeline config
        if pipe.zero_mean_latents:
            current_latent_mean = target_latent.mean(dim=[2, 3], keepdim=True)
            target_latent_input = target_latent - current_latent_mean
        else:
            target_latent_input = target_latent
        
        unet_input = torch.cat([rgb_latent_centered, target_latent_input], dim=1)
        noise_pred = pipe.unet(unet_input, t, encoder_hidden_states=text_embed).sample
        
        step_output = pipe.scheduler.step(noise_pred, t, target_latent_input)
        target_latent = step_output.prev_sample
        
        # Restore Mean - uses pipeline config (always restore if zero_mean enabled)
        if pipe.zero_mean_latents:
            target_latent = target_latent + current_latent_mean
        
        # Feature rescaling: maintain initial std to prevent fading
        if feature_rescaling:
            current_std = target_latent.std()
            if current_std > 0:
                target_latent = target_latent * (initial_std / current_std)
    
    # Restore degraded image mean for photometric consistency (if zero_mean enabled)
    # This ensures output has same exposure/brightness as input
    if pipe.zero_mean_latents:
        target_latent = target_latent + cond_mean
    
    # Decode final result
    final_rgb = pipe.decode_rgb(target_latent)
    final_rgb = torch.clip(final_rgb, -1.0, 1.0)
    
    return final_rgb


def analyze_single_image(image_np: np.ndarray, clean_np: np.ndarray, num_steps: int) -> dict:
    """Analyze a single image and return all metrics"""
    
    mean_val = float(np.mean(image_np))
    std_val = float(np.std(image_np))
    
    gradient_mag = compute_gradient_magnitude(image_np)
    frequency_analysis = compute_frequency_spectrum(image_np)
    local_var = compute_local_variance(image_np)
    edge_strength = compute_edge_strength(image_np)
    perceptual = compute_perceptual_metrics(image_np, clean_np)
    
    return {
        'num_steps': num_steps,
        'statistics': {
            'mean': mean_val,
            'std': std_val
        },
        'gradient_magnitude': gradient_mag,
        'frequency': frequency_analysis,
        'local_variance': local_var,
        'edge_strength': edge_strength,
        'perceptual': perceptual
    }


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor [-1, 1] to numpy [0, 1] in HWC format"""
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.squeeze().cpu().numpy()
    if tensor.ndim == 3:
        tensor = tensor.transpose(1, 2, 0)
    return tensor


def save_tensor_as_image(tensor: torch.Tensor, path: str):
    """Save tensor [-1, 1] as image"""
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.squeeze().cpu().numpy()
    if tensor.ndim == 3:
        tensor = tensor.transpose(1, 2, 0)
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(tensor).save(path)


def compare_step_configurations(
    pipe: MarigoldRestorationPipeline,
    input_image: Image.Image,
    clean_image: Image.Image,
    output_dir: str,
    device: torch.device,
    step_configs: list = None,
    feature_rescaling: bool = True,
):
    """Compare final results of different step configurations"""
    
    if step_configs is None:
        step_configs = [1, 5, 10, 20, 50, 100, 500]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Log pipeline configuration
    logging.info(f"Pipeline configuration:")
    logging.info(f"  zero_mean_latents: {pipe.zero_mean_latents}")
    logging.info(f"  feature_rescaling: {feature_rescaling}")
    
    # Preprocess images
    from torchvision.transforms.functional import pil_to_tensor
    
    rgb = pil_to_tensor(input_image.convert("RGB")).unsqueeze(0)
    rgb_norm = rgb / 255.0 * 2.0 - 1.0
    rgb_norm = rgb_norm.to(device, dtype=pipe.dtype)
    
    clean_rgb = pil_to_tensor(clean_image.convert("RGB")).unsqueeze(0)
    clean_rgb_norm = clean_rgb / 255.0 * 2.0 - 1.0
    clean_rgb_norm = clean_rgb_norm.to(device, dtype=pipe.dtype)
    
    clean_np = (clean_rgb_norm.squeeze().cpu().numpy() + 1.0) / 2.0
    clean_np = clean_np.transpose(1, 2, 0)
    
    # Also analyze degraded input
    degraded_np = (rgb_norm.squeeze().cpu().numpy() + 1.0) / 2.0
    degraded_np = degraded_np.transpose(1, 2, 0)
    
    # Save reference images
    input_image.save(os.path.join(output_dir, "00_input_degraded.png"))
    clean_image.save(os.path.join(output_dir, "00_reference_clean.png"))
    
    # Encode degraded image (done once)
    rgb_latent = pipe.encode_rgb(rgb_norm)
    
    # Text embedding (done once)
    if pipe.empty_text_embed is None:
        pipe.encode_empty_text()
    text_embed = pipe.empty_text_embed.repeat((1, 1, 1)).to(device)
    
    # Analyze clean reference
    clean_metrics = analyze_single_image(clean_np, clean_np, num_steps=0)
    clean_metrics['label'] = 'clean_reference'
    
    # Analyze degraded input
    degraded_metrics = analyze_single_image(degraded_np, clean_np, num_steps=0)
    degraded_metrics['label'] = 'degraded_input'
    
    logging.info("\n" + "="*70)
    logging.info("COMPARING FINAL RESULTS OF DIFFERENT STEP CONFIGURATIONS")
    logging.info("="*70)
    
    logging.info(f"\nClean reference:")
    logging.info(f"  Gradient magnitude: {clean_metrics['gradient_magnitude']:.4f}")
    logging.info(f"  High freq energy:   {clean_metrics['frequency']['high_freq']:.2f}")
    logging.info(f"  Local variance:     {clean_metrics['local_variance']:.6f}")
    logging.info(f"  Edge strength:      {clean_metrics['edge_strength']:.4f}")
    
    logging.info(f"\nDegraded input:")
    logging.info(f"  Gradient magnitude: {degraded_metrics['gradient_magnitude']:.4f}")
    logging.info(f"  High freq energy:   {degraded_metrics['frequency']['high_freq']:.2f}")
    logging.info(f"  Local variance:     {degraded_metrics['local_variance']:.6f}")
    logging.info(f"  Edge strength:      {degraded_metrics['edge_strength']:.4f}")
    logging.info(f"  PSNR vs clean:      {degraded_metrics['perceptual']['psnr']:.2f}")
    logging.info(f"  SSIM vs clean:      {degraded_metrics['perceptual']['ssim']:.4f}")
    
    # Run inference for each step configuration
    all_metrics = [clean_metrics, degraded_metrics]
    
    logging.info("\n" + "-"*70)
    logging.info(f"Running inference with different step counts...")
    logging.info(f"  feature_rescaling={feature_rescaling}")
    logging.info("-"*70)
    
    for num_steps in step_configs:
        logging.info(f"\n=== {num_steps}-step inference ===")
        
        # Run complete inference
        final_rgb = run_single_inference(
            pipe, rgb_latent, text_embed, num_steps, device, seed=42,
            feature_rescaling=feature_rescaling
        )
        
        # Convert to numpy
        result_np = tensor_to_numpy(final_rgb)
        
        # Analyze
        metrics = analyze_single_image(result_np, clean_np, num_steps)
        metrics['label'] = f'{num_steps}_steps'
        all_metrics.append(metrics)
        
        # Save image
        save_tensor_as_image(final_rgb, os.path.join(output_dir, f"result_{num_steps:02d}_steps.png"))
        
        # Log
        logging.info(f"  Gradient magnitude: {metrics['gradient_magnitude']:.4f}")
        logging.info(f"  High freq energy:   {metrics['frequency']['high_freq']:.2f}")
        logging.info(f"  Local variance:     {metrics['local_variance']:.6f}")
        logging.info(f"  Edge strength:      {metrics['edge_strength']:.4f}")
        logging.info(f"  PSNR vs clean:      {metrics['perceptual']['psnr']:.2f} dB")
        logging.info(f"  SSIM vs clean:      {metrics['perceptual']['ssim']:.4f}")
        logging.info(f"  Mean pixel value:   {metrics['statistics']['mean']:.4f}")
    
    # Save all metrics
    metrics_file = os.path.join(output_dir, "step_comparison_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(all_metrics, step_configs, output_dir)
    
    # Print summary table
    print_summary_table(all_metrics, step_configs)
    
    logging.info(f"\nAnalysis complete. Results saved to: {output_dir}")


def create_comparison_plots(all_metrics: list, step_configs: list, output_dir: str):
    """Create comparison plots for different step configurations"""
    
    # Extract metrics for inference results only (skip clean and degraded)
    inference_metrics = [m for m in all_metrics if m.get('num_steps', 0) > 0]
    
    steps = [m['num_steps'] for m in inference_metrics]
    gradient_mags = [m['gradient_magnitude'] for m in inference_metrics]
    high_freqs = [m['frequency']['high_freq'] for m in inference_metrics]
    local_vars = [m['local_variance'] for m in inference_metrics]
    edge_strengths = [m['edge_strength'] for m in inference_metrics]
    ssims = [m['perceptual']['ssim'] for m in inference_metrics]
    psnrs = [m['perceptual']['psnr'] for m in inference_metrics]
    means = [m['statistics']['mean'] for m in inference_metrics]
    
    # Get reference values
    clean_metrics = all_metrics[0]
    degraded_metrics = all_metrics[1]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Spatial Coherence: Comparing Final Results of Different Step Counts', fontsize=14)
    
    # Plot 1: PSNR
    axes[0, 0].plot(steps, psnrs, 'b-o', markersize=6, linewidth=2)
    axes[0, 0].axhline(y=degraded_metrics['perceptual']['psnr'], color='r', linestyle='--', label='Degraded input')
    axes[0, 0].set_title('PSNR vs Clean')
    axes[0, 0].set_xlabel('Number of Steps')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: SSIM
    axes[0, 1].plot(steps, ssims, 'g-o', markersize=6, linewidth=2)
    axes[0, 1].axhline(y=degraded_metrics['perceptual']['ssim'], color='r', linestyle='--', label='Degraded input')
    axes[0, 1].set_title('SSIM vs Clean')
    axes[0, 1].set_xlabel('Number of Steps')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Plot 3: Gradient Magnitude
    axes[0, 2].plot(steps, gradient_mags, 'm-o', markersize=6, linewidth=2)
    axes[0, 2].axhline(y=clean_metrics['gradient_magnitude'], color='g', linestyle='--', label='Clean reference')
    axes[0, 2].axhline(y=degraded_metrics['gradient_magnitude'], color='r', linestyle='--', label='Degraded input')
    axes[0, 2].set_title('Gradient Magnitude\n(Spatial Coherence)')
    axes[0, 2].set_xlabel('Number of Steps')
    axes[0, 2].set_ylabel('Mean Gradient')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xscale('log')
    
    # Plot 4: High Frequency Energy
    axes[0, 3].plot(steps, high_freqs, 'c-o', markersize=6, linewidth=2)
    axes[0, 3].axhline(y=clean_metrics['frequency']['high_freq'], color='g', linestyle='--', label='Clean reference')
    axes[0, 3].axhline(y=degraded_metrics['frequency']['high_freq'], color='r', linestyle='--', label='Degraded input')
    axes[0, 3].set_title('High Frequency Energy\n(Detail Preservation)')
    axes[0, 3].set_xlabel('Number of Steps')
    axes[0, 3].set_ylabel('High Freq Energy')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].set_xscale('log')
    
    # Plot 5: Local Variance
    axes[1, 0].plot(steps, local_vars, 'orange', marker='o', markersize=6, linewidth=2)
    axes[1, 0].axhline(y=clean_metrics['local_variance'], color='g', linestyle='--', label='Clean reference')
    axes[1, 0].axhline(y=degraded_metrics['local_variance'], color='r', linestyle='--', label='Degraded input')
    axes[1, 0].set_title('Local Variance\n(Texture Preservation)')
    axes[1, 0].set_xlabel('Number of Steps')
    axes[1, 0].set_ylabel('Local Variance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 6: Edge Strength
    axes[1, 1].plot(steps, edge_strengths, 'brown', marker='o', markersize=6, linewidth=2)
    axes[1, 1].axhline(y=clean_metrics['edge_strength'], color='g', linestyle='--', label='Clean reference')
    axes[1, 1].axhline(y=degraded_metrics['edge_strength'], color='r', linestyle='--', label='Degraded input')
    axes[1, 1].set_title('Edge Strength\n(Structural Preservation)')
    axes[1, 1].set_xlabel('Number of Steps')
    axes[1, 1].set_ylabel('Edge Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    # Plot 7: Mean Pixel Value
    axes[1, 2].plot(steps, means, 'purple', marker='o', markersize=6, linewidth=2)
    axes[1, 2].axhline(y=clean_metrics['statistics']['mean'], color='g', linestyle='--', label='Clean reference')
    axes[1, 2].axhline(y=degraded_metrics['statistics']['mean'], color='r', linestyle='--', label='Degraded input')
    axes[1, 2].set_title('Mean Pixel Value\n(Global Brightness)')
    axes[1, 2].set_xlabel('Number of Steps')
    axes[1, 2].set_ylabel('Mean [0,1]')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xscale('log')
    
    # Plot 8: Normalized comparison
    # Normalize to clean reference = 1.0
    norm_grad = np.array(gradient_mags) / clean_metrics['gradient_magnitude']
    norm_hf = np.array(high_freqs) / clean_metrics['frequency']['high_freq']
    norm_edge = np.array(edge_strengths) / clean_metrics['edge_strength']
    
    axes[1, 3].plot(steps, norm_grad, 'm-o', markersize=5, label='Gradient')
    axes[1, 3].plot(steps, norm_hf, 'c-s', markersize=5, label='High Freq')
    axes[1, 3].plot(steps, norm_edge, 'brown', marker='^', markersize=5, label='Edges')
    axes[1, 3].axhline(y=1.0, color='g', linestyle='--', label='Clean = 1.0')
    axes[1, 3].set_title('Normalized to Clean Reference')
    axes[1, 3].set_xlabel('Number of Steps')
    axes[1, 3].set_ylabel('Ratio to Clean')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_comparison_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_summary_table(all_metrics: list, step_configs: list):
    """Print a summary table of results"""
    
    logging.info("\n" + "="*90)
    logging.info("SUMMARY TABLE")
    logging.info("="*90)
    logging.info(f"{'Config':<15} {'PSNR':>8} {'SSIM':>8} {'Gradient':>10} {'HighFreq':>10} {'LocalVar':>10} {'Edges':>8}")
    logging.info("-"*90)
    
    for m in all_metrics:
        label = m.get('label', f"{m.get('num_steps', '?')}_steps")
        psnr = m['perceptual']['psnr']
        ssim_val = m['perceptual']['ssim']
        grad = m['gradient_magnitude']
        hf = m['frequency']['high_freq']
        lv = m['local_variance']
        edge = m['edge_strength']
        
        logging.info(f"{label:<15} {psnr:>8.2f} {ssim_val:>8.4f} {grad:>10.4f} {hf:>10.2f} {lv:>10.6f} {edge:>8.4f}")
    
    logging.info("="*90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare spatial coherence across different step counts")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2")
    parser.add_argument("--input", type=str, required=True, help="Input degraded image path")
    parser.add_argument("--clean", type=str, required=True, help="Clean reference image path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--steps", type=str, default="1,2,3,5,10,20,50", 
                        help="Comma-separated list of step counts to test")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--scheduler", type=str, choices=SCHEDULER_CHOICES, default="ddim",
                        help="Scheduler type: ddim, lcm, heun. Default: ddim")
    parser.add_argument("--feature_rescaling", action="store_true", default=False,
                        help="Enable feature rescaling to maintain latent std (default: disabled)")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force CPU usage, disable CUDA even if available")
    
    args = parser.parse_args()
    
    feature_rescaling = args.feature_rescaling
    
    # Parse step configurations
    step_configs = [int(s.strip()) for s in args.steps.split(',')]
    
    # Device selection: force CPU if --cpu flag is set
    if args.cpu:
        device = torch.device("cpu")
        logging.info("Forcing CPU usage (CUDA disabled)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
    
    dtype = torch.float16 if args.fp16 else torch.float32
    
    pipe = load_pipeline(args.checkpoint, args.base_model, dtype, args.scheduler)
    pipe = pipe.to(device)
    
    try:
        # Only enable xformers if using CUDA
        if device.type == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("XFormers memory efficient attention enabled")
        else:
            logging.info("Skipping XFormers (CPU mode)")
    except Exception as e:
        logging.info(f"XFormers not available: {e}")
        pass
    
    input_image = Image.open(args.input)
    clean_image = Image.open(args.clean)
    
    with torch.no_grad():
        compare_step_configurations(
            pipe, input_image, clean_image, args.output_dir, device, step_configs,
            feature_rescaling=feature_rescaling
        )
