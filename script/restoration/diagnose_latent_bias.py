# Diagnostic script to analyze latent space bias in restoration training
# 
# This script investigates if the model has a systematic bias in predictions
# by analyzing:
# 1. Training data latent statistics (degraded vs clean)
# 2. Model prediction bias vs target
# 3. Velocity target statistics
# 4. Visual analysis of latents and decoded images
#
# Usage:
#   python script/restoration/diagnose_latent_bias.py \
#       --checkpoint checkpoints/013_best_009000/latest \
#       --config config/train_marigold_restoration.yaml \
#       --output_dir output/diagnosis \
#       --num_batches 10
#
# DETERMINISM: This script is fully deterministic. Use --seed to control randomness.
# Running with the same seed will produce identical results.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DConditionModel

from marigold import MarigoldRestorationPipeline
from src.dataset import DatasetMode
from src.dataset.restoration_dataset_factory import RestorationDatasetFactory
from src.util.config_util import recursive_load_config

# Supported scheduler types
SCHEDULER_CHOICES = ["ddim", "lcm", "heun"]

# Default seed for reproducibility
DEFAULT_SEED = 42

# Global file logger (set in main)
file_logger = None


def setup_logging(output_dir: Path):
    """Setup logging to both console and file"""
    global file_logger
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logger
    log_file = output_dir / "diagnosis_results.txt"
    file_logger = open(log_file, "w")
    
    # Configure console logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')


def log_message(msg: str):
    """Log message to both console and file"""
    logging.info(msg)
    if file_logger:
        file_logger.write(msg + "\n")
        file_logger.flush()


def set_global_seed(seed: int):
    """Set all random seeds for full determinism"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log_message(f"Global seed set to {seed} (fully deterministic mode)")


def create_dataloader(dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    """Create a deterministic DataLoader with fixed seed generator"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def decode_latent_to_rgb(pipe, latent: torch.Tensor) -> torch.Tensor:
    """
    Decode latent to RGB using VAE decoder with proper scaling.
    
    CRITICAL: Latents in Stable Diffusion are scaled by 0.18215.
    We must divide by this factor before decoding.
    
    Args:
        pipe: MarigoldRestorationPipeline with VAE
        latent: Latent tensor [B, 4, h, w] (already scaled by 0.18215)
    
    Returns:
        RGB tensor [B, 3, H, W] in range [-1, 1]
    """
    # Use pipeline's decode_rgb which handles the scaling factor
    # Verified in marigold_restoration_pipeline.py:
    #   def decode_rgb(self, rgb_latent):
    #       rgb_latent = rgb_latent / self.latent_scale_factor  # 0.18215
    #       z = self.vae.post_quant_conv(rgb_latent)
    #       rgb_image = self.vae.decoder(z)
    #       return rgb_image
    with torch.no_grad():
        rgb = pipe.decode_rgb(latent)
    return rgb


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor image to numpy for visualization.
    
    Args:
        tensor: Image tensor [B, C, H, W] or [C, H, W] in range [-1, 1]
    
    Returns:
        Numpy array [H, W, C] in range [0, 1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Clamp to valid range and convert to [0, 1]
    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    
    # CHW -> HWC
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return img


def visualize_latent_channels(latent: torch.Tensor, title: str = "") -> np.ndarray:
    """
    Visualize 4 latent channels as grayscale images in a row.
    
    Args:
        latent: Latent tensor [B, 4, h, w] or [4, h, w]
        title: Title for the visualization
    
    Returns:
        Numpy array of the visualization
    """
    if latent.dim() == 4:
        latent = latent[0]  # Take first from batch
    
    latent = latent.cpu().numpy()
    
    # Normalize each channel independently for visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, fontsize=12)
    
    for i in range(4):
        channel = latent[i]
        vmin, vmax = channel.min(), channel.max()
        axes[i].imshow(channel, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f"Ch{i}: [{vmin:.3f}, {vmax:.3f}]")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def load_pipeline_and_unet(checkpoint_path: str, base_model: str, device: torch.device, scheduler_type: str = "ddim"):
    """Load pipeline with trained U-Net"""
    from diffusers import DDIMScheduler, LCMScheduler, HeunDiscreteScheduler
    
    checkpoint_path_obj = Path(checkpoint_path)
    unet_checkpoint_path = checkpoint_path_obj / "unet"
    
    log_message(f"Loading base model from: {base_model}")
    pipe = MarigoldRestorationPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
    
    if unet_checkpoint_path.exists():
        log_message(f"Loading trained U-Net from: {unet_checkpoint_path}")
        pipe.unet = UNet2DConditionModel.from_pretrained(unet_checkpoint_path, torch_dtype=torch.float32)
    else:
        log_message(f"WARNING: No trained U-Net found at {unet_checkpoint_path}, using base model")
    
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
    log_message(f"Scheduler: {orig_type} → {new_type}")
    
    _load_restoration_config(pipe, checkpoint_path_obj)
    
    pipe = pipe.to(device)
    pipe.unet.eval()
    
    return pipe


def _load_restoration_config(pipe, checkpoint_path: Path):
    """Load restoration config from checkpoint if available."""
    import json
    
    config_path = checkpoint_path / "restoration_config.json"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        
        normalize_latents = config.get("normalize_latents", False)
        pipe.set_normalize_latents(normalize_latents)
        
        log_message(f"✓ Loaded restoration config: normalize_latents={normalize_latents}")
    else:
        log_message("No restoration_config.json found (old checkpoint), using defaults")
        pipe.set_normalize_latents(False)


def analyze_latent_statistics(pipe, dataloader, device, num_batches: int = 10):
    """Analyze latent space statistics for degraded vs clean images"""
    
    log_message("\n" + "="*70)
    log_message("PART 1: LATENT SPACE STATISTICS (degraded vs clean)")
    log_message("="*70)
    
    degraded_means = []
    degraded_stds = []
    clean_means = []
    clean_stds = []
    diff_means = []
    
    for i, batch in enumerate(tqdm(dataloader, desc="Analyzing latents", total=num_batches)):
        if i >= num_batches:
            break
            
        degraded_rgb = batch['degraded_rgb_norm'].to(device)
        clean_rgb = batch['clean_rgb_norm'].to(device)
        
        with torch.no_grad():
            degraded_latent = pipe.encode_rgb(degraded_rgb)
            clean_latent = pipe.encode_rgb(clean_rgb)
        
        degraded_means.append(degraded_latent.mean().item())
        degraded_stds.append(degraded_latent.std().item())
        clean_means.append(clean_latent.mean().item())
        clean_stds.append(clean_latent.std().item())
        diff_means.append((clean_latent - degraded_latent).mean().item())

    log_message(f"\nAnalyzed {len(degraded_means)} batches")
    log_message(f"\nDegraded latent statistics:")
    log_message(f"  Mean of means: {np.mean(degraded_means):.6f} ± {np.std(degraded_means):.6f}")
    log_message(f"  Mean of stds:  {np.mean(degraded_stds):.6f} ± {np.std(degraded_stds):.6f}")
    
    log_message(f"\nClean latent statistics:")
    log_message(f"  Mean of means: {np.mean(clean_means):.6f} ± {np.std(clean_means):.6f}")
    log_message(f"  Mean of stds:  {np.mean(clean_stds):.6f} ± {np.std(clean_stds):.6f}")
    
    log_message(f"\nDifference (clean - degraded):")
    log_message(f"  Mean of means: {np.mean(diff_means):.6f} ± {np.std(diff_means):.6f}")
    
    return {
        'degraded_mean': np.mean(degraded_means),
        'clean_mean': np.mean(clean_means),
        'diff_mean': np.mean(diff_means),
    }


def analyze_model_predictions(pipe, dataloader, device, num_batches: int = 10):
    """Analyze model prediction bias vs target"""
    
    log_message("\n" + "="*70)
    log_message("PART 2: MODEL PREDICTION ANALYSIS")
    log_message("="*70)
    
    training_scheduler = DDPMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    
    if pipe.empty_text_embed is None:
        pipe.encode_empty_text()
    
    timestep_ranges = [(0, 250), (250, 500), (500, 750), (750, 1000)]
    stats_by_range = {r: {'target_means': [], 'pred_means': [], 'bias': []} for r in timestep_ranges}
    
    all_target_means = []
    all_pred_means = []
    all_biases = []
    
    generator = torch.Generator(device=device)
    generator.manual_seed(DEFAULT_SEED + 2000)
    
    for i, batch in enumerate(tqdm(dataloader, desc="Analyzing predictions", total=num_batches)):
        if i >= num_batches:
            break
            
        degraded_rgb = batch['degraded_rgb_norm'].to(device)
        clean_rgb = batch['clean_rgb_norm'].to(device)
        batch_size = degraded_rgb.shape[0]
        
        with torch.no_grad():
            degraded_latent = pipe.encode_rgb(degraded_rgb)
            clean_latent = pipe.encode_rgb(clean_rgb)
        
        timesteps = torch.randint(0, 1000, (batch_size,), device=device, generator=generator).long()
        noise = torch.randn(clean_latent.shape, device=device, generator=generator)
        noisy_latents = training_scheduler.add_noise(clean_latent, noise, timesteps)
        target = training_scheduler.get_velocity(clean_latent, noise, timesteps)
        
        text_embed = pipe.empty_text_embed.repeat((batch_size, 1, 1)).to(device)
        cat_latents = torch.cat([degraded_latent, noisy_latents], dim=1)
        
        with torch.no_grad():
            model_pred = pipe.unet(cat_latents, timesteps, encoder_hidden_states=text_embed).sample

        for b in range(batch_size):
            t = timesteps[b].item()
            target_mean = target[b].mean().item()
            pred_mean = model_pred[b].mean().item()
            bias = pred_mean - target_mean
            
            all_target_means.append(target_mean)
            all_pred_means.append(pred_mean)
            all_biases.append(bias)
            
            for (t_min, t_max) in timestep_ranges:
                if t_min <= t < t_max:
                    stats_by_range[(t_min, t_max)]['target_means'].append(target_mean)
                    stats_by_range[(t_min, t_max)]['pred_means'].append(pred_mean)
                    stats_by_range[(t_min, t_max)]['bias'].append(bias)
                    break
    
    log_message(f"\nOverall statistics ({len(all_biases)} samples):")
    log_message(f"  Target mean:     {np.mean(all_target_means):.6f} ± {np.std(all_target_means):.6f}")
    log_message(f"  Prediction mean: {np.mean(all_pred_means):.6f} ± {np.std(all_pred_means):.6f}")
    log_message(f"  Bias (pred-target): {np.mean(all_biases):.6f} ± {np.std(all_biases):.6f}")
    
    log_message(f"\nStatistics by timestep range:")
    for (t_min, t_max), stats in stats_by_range.items():
        if len(stats['bias']) > 0:
            log_message(f"\n  Timestep [{t_min}, {t_max})  (n={len(stats['bias'])})")
            log_message(f"    Target mean:     {np.mean(stats['target_means']):.6f}")
            log_message(f"    Prediction mean: {np.mean(stats['pred_means']):.6f}")
            log_message(f"    Bias:            {np.mean(stats['bias']):.6f}")
    
    return {
        'overall_bias': np.mean(all_biases),
        'bias_std': np.std(all_biases),
        'stats_by_range': stats_by_range,
    }


def analyze_velocity_target(pipe, dataloader, device, num_batches: int = 10):
    """Analyze velocity target statistics"""
    
    log_message("\n" + "="*70)
    log_message("PART 3: VELOCITY TARGET ANALYSIS")
    log_message("="*70)
    
    training_scheduler = DDPMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    
    test_timesteps = [999, 750, 500, 250, 100, 50, 10]
    
    for t_val in test_timesteps:
        velocity_means = []
        
        generator = torch.Generator(device=device)
        generator.manual_seed(DEFAULT_SEED + 3000 + t_val)
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            clean_rgb = batch['clean_rgb_norm'].to(device)
            batch_size = clean_rgb.shape[0]
            
            with torch.no_grad():
                clean_latent = pipe.encode_rgb(clean_rgb)
            
            timesteps = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            noise = torch.randn(clean_latent.shape, device=device, generator=generator)
            
            velocity = training_scheduler.get_velocity(clean_latent, noise, timesteps)
            velocity_means.append(velocity.mean().item())
        
        log_message(f"  t={t_val:4d}: velocity mean = {np.mean(velocity_means):.6f} ± {np.std(velocity_means):.6f}")


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute PSNR between two images.
    
    Args:
        img1, img2: Images as numpy arrays [H, W, C] in range [0, 1]
    
    Returns:
        PSNR value in dB
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def save_image_grid_with_dual_psnr(
    images: list,
    titles: list,
    save_path: Path,
    figsize_per_image: tuple = (4, 4),
    ref1_idx: int = None,
    ref1_name: str = "Clean",
    ref2_idx: int = None,
    ref2_name: str = "Clean VAE",
    skip_psnr_indices: list = None,
):
    """
    Save a grid of images with titles and dual PSNR references.
    
    Args:
        images: List of numpy arrays [H, W, C] in range [0, 1]
        titles: List of titles for each image
        save_path: Path to save the figure
        figsize_per_image: Size per image in the grid
        ref1_idx: Index of first reference image for PSNR (e.g., Clean original)
        ref1_name: Name for first reference in PSNR label
        ref2_idx: Index of second reference image for PSNR (e.g., Clean VAE roundtrip)
        ref2_name: Name for second reference in PSNR label
        skip_psnr_indices: List of indices to skip PSNR calculation (reference images)
    """
    if skip_psnr_indices is None:
        skip_psnr_indices = []
    
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(figsize_per_image[0] * n_images, figsize_per_image[1]))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(np.clip(img, 0, 1))
        
        # Add PSNR to title if this is not a reference image
        if i not in skip_psnr_indices:
            psnr_lines = []
            if ref1_idx is not None:
                psnr1 = compute_psnr(images[ref1_idx], img)
                psnr_lines.append(f"vs {ref1_name}: {psnr1:.2f} dB")
            if ref2_idx is not None:
                psnr2 = compute_psnr(images[ref2_idx], img)
                psnr_lines.append(f"vs {ref2_name}: {psnr2:.2f} dB")
            if psnr_lines:
                title = f"{title}\n" + "\n".join(psnr_lines)
        
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_image_grid(
    images: list,
    titles: list,
    save_path: Path,
    figsize_per_image: tuple = (4, 4),
    reference_idx: int = None,
):
    """
    Save a grid of images with titles and optional PSNR vs single reference.
    
    Args:
        images: List of numpy arrays [H, W, C] in range [0, 1]
        titles: List of titles for each image
        save_path: Path to save the figure
        figsize_per_image: Size per image in the grid
        reference_idx: Index of reference image for PSNR calculation (None to skip)
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(figsize_per_image[0] * n_images, figsize_per_image[1]))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(np.clip(img, 0, 1))
        
        # Add PSNR to title if reference is provided and this is not the reference
        if reference_idx is not None and i != reference_idx:
            psnr = compute_psnr(images[reference_idx], img)
            title = f"{title}\nPSNR: {psnr:.2f} dB"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_latent_channels_grid(
    latents_dict: dict,
    save_path: Path,
):
    """
    Save a grid showing 4 latent channels for multiple latent tensors.
    
    Args:
        latents_dict: Dict of {name: latent_tensor} where latent is [B, 4, h, w] or [4, h, w]
        save_path: Path to save the figure
    """
    n_rows = len(latents_dict)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    
    if n_rows == 1:
        axes = [axes]
    
    for row_idx, (name, latent) in enumerate(latents_dict.items()):
        if latent.dim() == 4:
            latent = latent[0]
        latent_np = latent.cpu().numpy()
        
        for ch in range(4):
            channel = latent_np[ch]
            vmin, vmax = channel.min(), channel.max()
            im = axes[row_idx][ch].imshow(channel, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row_idx][ch].set_title(f"{name} Ch{ch}\n[{vmin:.3f}, {vmax:.3f}]", fontsize=9)
            axes[row_idx][ch].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_and_visualize_batch(
    pipe,
    batch,
    device,
    output_dir: Path,
    batch_idx: int,
    num_inference_steps: int = 10,
    seed: int = DEFAULT_SEED,
):
    """
    Analyze a single batch and save visualizations.
    
    Saves:
    1. image_grid.png: Degraded / Clean / VAE-Decoded-Degraded / Model Output
    2. latent_channels.png: 4-channel visualization for Clean/Degraded/Difference/Output
    3. denoising_progression.png: Latent evolution at key timesteps
    
    Args:
        pipe: MarigoldRestorationPipeline
        batch: Data batch from dataloader
        device: torch device
        output_dir: Directory to save outputs
        batch_idx: Batch index for naming
        num_inference_steps: Number of denoising steps
        seed: Random seed for reproducibility
    """
    from diffusers import DDIMScheduler
    
    batch_dir = output_dir / f"batch_{batch_idx:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    degraded_rgb = batch['degraded_rgb_norm'].to(device)  # [-1, 1]
    clean_rgb = batch['clean_rgb_norm'].to(device)  # [-1, 1]
    
    # Encode to latent space
    with torch.no_grad():
        degraded_latent = pipe.encode_rgb(degraded_rgb)
        clean_latent = pipe.encode_rgb(clean_rgb)
    
    # Decode latents back to RGB (VAE roundtrip test)
    with torch.no_grad():
        vae_decoded_degraded = decode_latent_to_rgb(pipe, degraded_latent)
        vae_decoded_clean = decode_latent_to_rgb(pipe, clean_latent)
    
    # Compute latent difference
    latent_diff = torch.abs(clean_latent - degraded_latent)

    # Run inference and collect intermediate latents
    inference_scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
    )
    
    if pipe.empty_text_embed is None:
        pipe.encode_empty_text()
    
    # Initialize with noise
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    target_latent = torch.randn(degraded_latent.shape, device=device, generator=generator)
    
    text_embed = pipe.empty_text_embed.repeat((degraded_latent.shape[0], 1, 1)).to(device)
    
    inference_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = inference_scheduler.timesteps
    
    # Store intermediate latents at ALL steps (not just key steps)
    all_step_latents = {}
    all_step_latents['initial_noise'] = target_latent.clone()
    
    # Key timesteps to capture for detailed logging (first, middle, last)
    key_step_indices = [0, len(timesteps) // 2, len(timesteps) - 1]
    
    log_message(f"\n  Batch {batch_idx}: Running {num_inference_steps} inference steps...")
    log_message(f"    Timesteps: {timesteps.tolist()}")

    for i, t in enumerate(timesteps):
        # Apply latent normalization if enabled (matching pipeline behavior)
        if pipe.normalize_latents:
            degraded_latent_mean = degraded_latent.mean(dim=(2, 3), keepdim=True)
            degraded_latent_std = degraded_latent.std(dim=(2, 3), keepdim=True)
            degraded_latent_input = (degraded_latent - degraded_latent_mean) / (degraded_latent_std + 1e-8)
            
            target_latent_mean = target_latent.mean(dim=(2, 3), keepdim=True)
            target_latent_std = target_latent.std(dim=(2, 3), keepdim=True)
            target_latent_input = (target_latent - target_latent_mean) / (target_latent_std + 1e-8)
        else:
            degraded_latent_input = degraded_latent
            target_latent_input = target_latent
        
        unet_input = torch.cat([degraded_latent_input, target_latent_input], dim=1)
        
        with torch.no_grad():
            noise_pred = pipe.unet(unet_input, t, encoder_hidden_states=text_embed).sample
        
        step_output = inference_scheduler.step(noise_pred, t, target_latent_input, generator=generator)
        target_latent = step_output.prev_sample
        
        # Restore normalization if applied
        if pipe.normalize_latents:
            target_latent = target_latent * target_latent_std + target_latent_mean
        
        # Store latent at EVERY step
        step_name = f"step{i+1}_t{int(t.item())}"
        all_step_latents[step_name] = target_latent.clone()
        
        # Log at key steps
        if i in key_step_indices:
            log_message(f"    Step {i+1} (t={t.item()}): latent mean={target_latent.mean():.4f}, std={target_latent.std():.4f}")
    
    # Final output latent
    final_latent = target_latent
    
    # Restore degraded normalization for photometric consistency (if normalization enabled)
    if pipe.normalize_latents:
        degraded_latent_mean = degraded_latent.mean(dim=(2, 3), keepdim=True)
        degraded_latent_std = degraded_latent.std(dim=(2, 3), keepdim=True)
        final_latent = (final_latent - final_latent.mean(dim=(2, 3), keepdim=True)) / (final_latent.std(dim=(2, 3), keepdim=True) + 1e-8)
        final_latent = final_latent * degraded_latent_std + degraded_latent_mean

    # Decode final latent to RGB
    with torch.no_grad():
        model_output_rgb = decode_latent_to_rgb(pipe, final_latent)
    
    # === SAVE VISUALIZATIONS ===
    
    # Convert images to numpy for PSNR calculation
    clean_np = tensor_to_numpy_image(clean_rgb)
    clean_vae_np = tensor_to_numpy_image(vae_decoded_clean)
    degraded_np = tensor_to_numpy_image(degraded_rgb)
    degraded_vae_np = tensor_to_numpy_image(vae_decoded_degraded)
    output_np = tensor_to_numpy_image(model_output_rgb)
    
    # 1. Image grid: Clean / Clean VAE / Degraded / Degraded VAE / Model Output
    # With dual PSNR: vs Clean original AND vs Clean VAE roundtrip
    images = [
        clean_np,
        clean_vae_np,
        degraded_np,
        degraded_vae_np,
        output_np,
    ]
    titles = [
        "Target (Clean)",
        "Clean VAE Roundtrip",
        "Input (Degraded)",
        "Degraded VAE Roundtrip",
        f"Model Output ({num_inference_steps} steps)",
    ]
    # ref1_idx=0 (Clean), ref2_idx=1 (Clean VAE)
    # skip_psnr_indices=[0] means only Clean (idx 0) has no PSNR
    # Clean VAE (idx 1) will show PSNR vs Clean (to see VAE reconstruction loss)
    save_image_grid_with_dual_psnr(
        images, titles, 
        batch_dir / f"image_grid_{num_inference_steps}steps.png",
        ref1_idx=0, ref1_name="Clean",
        ref2_idx=1, ref2_name="Clean VAE",
        skip_psnr_indices=[0],
    )
    
    # 2. Latent channels grid: Clean / Degraded / Difference / Final Output
    latents_dict = {
        "Clean (target latent)": clean_latent,
        "Degraded (input latent)": degraded_latent,
        "|Clean-Degraded| (what model must learn)": latent_diff,
        f"Final Output ({num_inference_steps} steps)": final_latent,
    }
    save_latent_channels_grid(latents_dict, batch_dir / f"latent_channels_{num_inference_steps}steps.png")

    # 3. Denoising progression: ALL steps latent channels
    save_latent_channels_grid(all_step_latents, batch_dir / f"latent_progression_{num_inference_steps}steps.png")
    
    # 4. Decoded progression images (all steps)
    progression_images = []
    progression_titles = []
    for name, latent in all_step_latents.items():
        with torch.no_grad():
            decoded = decode_latent_to_rgb(pipe, latent)
        progression_images.append(tensor_to_numpy_image(decoded))
        progression_titles.append(name)
    save_image_grid(progression_images, progression_titles, batch_dir / f"denoising_decoded_{num_inference_steps}steps.png")
    
    log_message(f"    Saved visualizations to {batch_dir}")
    
    return {
        'degraded_latent': degraded_latent,
        'clean_latent': clean_latent,
        'final_latent': final_latent,
        'all_step_latents': all_step_latents,
    }


def analyze_single_step_inference(pipe, dataloader, device, seed: int, output_dir: Path):
    """Analyze what happens in a single inference step with visualizations"""
    
    log_message("\n" + "="*70)
    log_message("PART 4: SINGLE STEP INFERENCE ANALYSIS WITH VISUALIZATIONS")
    log_message("="*70)
    
    # Create dedicated subfolder for Part 4
    part4_dir = output_dir / "part4_inference"
    part4_dir.mkdir(parents=True, exist_ok=True)
    
    # Get first batch
    batch = next(iter(dataloader))
    
    degraded_rgb = batch['degraded_rgb_norm'].to(device)
    clean_rgb = batch['clean_rgb_norm'].to(device)
    
    # Log input latent statistics (as in original script)
    with torch.no_grad():
        degraded_latent = pipe.encode_rgb(degraded_rgb)
        clean_latent = pipe.encode_rgb(clean_rgb)
    
    log_message(f"\nInput latent statistics:")
    log_message(f"  Degraded: mean={degraded_latent.mean():.4f}, std={degraded_latent.std():.4f}")
    log_message(f"  Clean:    mean={clean_latent.mean():.4f}, std={clean_latent.std():.4f}")
    
    # Initialize noise and log its stats
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    initial_noise = torch.randn(degraded_latent.shape, device=device, generator=generator)
    log_message(f"  Initial noise: mean={initial_noise.mean():.4f}, std={initial_noise.std():.4f}")
    
    # Test with different step counts in order: 1, 5, 10
    for num_steps in [1, 5, 10]:
        log_message(f"\n  Testing with {num_steps} steps:")
        analyze_and_visualize_batch(
            pipe=pipe,
            batch=batch,
            device=device,
            output_dir=part4_dir,
            batch_idx=0,
            num_inference_steps=num_steps,
            seed=seed,
        )


def main():
    parser = argparse.ArgumentParser(description="Diagnose latent bias in restoration model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="config/train_marigold_restoration.yaml")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2")
    parser.add_argument("--base_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to analyze")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--scheduler", type=str, choices=SCHEDULER_CHOICES, default="ddim",
                        help="Scheduler type: ddim, lcm, heun. Default: ddim")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, 
                        help=f"Random seed for reproducibility. Default: {DEFAULT_SEED}")
    parser.add_argument("--num_vis_batches", type=int, default=3,
                        help="Number of batches to visualize in detail. Default: 3")
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    
    # Set global seed FIRST for full determinism
    set_global_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Device: {device}")
    
    # Load config
    cfg = recursive_load_config(args.config)

    # Resolve base data dir
    base_data_dir = args.base_data_dir or os.environ.get("BASE_DATA_DIR", "./data")
    base_data_dir = os.path.abspath(base_data_dir)
    log_message(f"Base data dir: {base_data_dir}")
    log_message(f"Output dir: {output_dir}")
    
    # Load pipeline
    pipe = load_pipeline_and_unet(args.checkpoint, args.base_model, device, args.scheduler)
    
    # Create training dataset
    log_message("Loading training dataset...")
    train_dataset = RestorationDatasetFactory.create_dataset(
        cfg.dataset.train,
        mode=DatasetMode.TRAIN,
        base_data_dir=base_data_dir,
    )
    
    if hasattr(train_dataset, 'set_epoch'):
        train_dataset.set_epoch(0)
    
    log_message(f"Dataset size: {len(train_dataset)}")
    
    log_message("\n" + "="*70)
    log_message(f"DIAGNOSTIC RUN (seed={args.seed})")
    log_message("="*70)
    
    # Part 1: Latent statistics
    loader1 = create_dataloader(train_dataset, args.batch_size, args.seed)
    analyze_latent_statistics(pipe, loader1, device, args.num_batches)
    
    # Part 2: Model predictions
    loader2 = create_dataloader(train_dataset, args.batch_size, args.seed + 1000)
    analyze_model_predictions(pipe, loader2, device, args.num_batches)
    
    # Part 3: Velocity target
    loader3 = create_dataloader(train_dataset, args.batch_size, args.seed)
    analyze_velocity_target(pipe, loader3, device, args.num_batches)

    # Part 4: Single step inference with visualizations
    loader4 = create_dataloader(train_dataset, args.batch_size, args.seed, shuffle=False)
    analyze_single_step_inference(pipe, loader4, device, args.seed, output_dir)
    
    # Part 5: Visualize multiple batches
    log_message("\n" + "="*70)
    log_message(f"PART 5: DETAILED VISUALIZATION ({args.num_vis_batches} batches)")
    log_message("="*70)
    
    # Create dedicated subfolder for Part 5
    part5_dir = output_dir / "part5_batches"
    part5_dir.mkdir(parents=True, exist_ok=True)
    
    loader5 = create_dataloader(train_dataset, args.batch_size, args.seed, shuffle=False)
    for batch_idx, batch in enumerate(loader5):
        if batch_idx >= args.num_vis_batches:
            break
        analyze_and_visualize_batch(
            pipe=pipe,
            batch=batch,
            device=device,
            output_dir=part5_dir,
            batch_idx=batch_idx,
            num_inference_steps=10,
            seed=args.seed + batch_idx,
        )
    
    log_message("\n" + "="*70)
    log_message("DIAGNOSIS COMPLETE")
    log_message(f"Results saved to: {output_dir}")
    log_message("="*70)
    
    # Close file logger
    if file_logger:
        file_logger.close()


if __name__ == "__main__":
    main()
