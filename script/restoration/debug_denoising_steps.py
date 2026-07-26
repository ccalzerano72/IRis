# Debug script to visualize what happens at each denoising step
# This helps diagnose why quality degrades with more steps

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from marigold import MarigoldRestorationPipeline

logging.basicConfig(level=logging.INFO)

# Supported scheduler types
SCHEDULER_CHOICES = ["ddim", "lcm", "heun"]


def load_pipeline(checkpoint_path: str, base_model: str, dtype: torch.dtype, scheduler_type: str = "ddim"):
    """Load pipeline (simplified from run.py)"""
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
    
    # Load restoration config (inference-relevant settings like normalize_latents)
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
        
        # Apply normalize_latents setting
        normalize_latents = config.get("normalize_latents", False)
        pipe.set_normalize_latents(normalize_latents)
        
        logging.info(f"✓ Loaded restoration config: normalize_latents={normalize_latents}")
    else:
        # Old checkpoint without restoration_config.json - use defaults
        logging.info("No restoration_config.json found (old checkpoint), using defaults")
        pipe.set_normalize_latents(False)


def debug_denoising(
    pipe: MarigoldRestorationPipeline,
    input_image: Image.Image,
    num_steps: int,
    output_dir: str,
    device: torch.device,
):
    """Run denoising and save intermediate results at each step"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess image
    from torchvision.transforms.functional import pil_to_tensor
    rgb = pil_to_tensor(input_image.convert("RGB")).unsqueeze(0)  # [1, 3, H, W]
    rgb_norm = rgb / 255.0 * 2.0 - 1.0  # [-1, 1]
    rgb_norm = rgb_norm.to(device, dtype=pipe.dtype)
    
    # Save input
    input_image.save(os.path.join(output_dir, "00_input.png"))
    
    # Encode degraded image
    rgb_latent = pipe.encode_rgb(rgb_norm)  # [1, 4, h, w]
    logging.info(f"Encoded latent shape: {rgb_latent.shape}")
    logging.info(f"Encoded latent stats: min={rgb_latent.min():.3f}, max={rgb_latent.max():.3f}, mean={rgb_latent.mean():.3f}")
    
    # Normalize conditioning latent (done once)
    if pipe.normalize_latents:
        cond_mean = rgb_latent.mean(dim=[2, 3], keepdim=True)
        cond_std = rgb_latent.std(dim=[2, 3], keepdim=True)
        rgb_latent_normalized = (rgb_latent - cond_mean) / (cond_std + 1e-8)
        logging.info(f"Normalization enabled: normalizing conditioning latent (mean per channel: {cond_mean.squeeze().tolist()}, std per channel: {cond_std.squeeze().tolist()})")
    else:
        rgb_latent_normalized = rgb_latent
        cond_mean = None
        cond_std = None
    
    # Initialize with random noise
    generator = torch.Generator(device=device).manual_seed(42)
    target_latent = torch.randn(rgb_latent.shape, device=device, dtype=pipe.dtype, generator=generator)
    logging.info(f"Initial noise stats: min={target_latent.min():.3f}, max={target_latent.max():.3f}, mean={target_latent.mean():.3f}")
    
    # Text embedding
    if pipe.empty_text_embed is None:
        pipe.encode_empty_text()
    text_embed = pipe.empty_text_embed.repeat((1, 1, 1)).to(device)
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    logging.info(f"Timesteps: {timesteps.tolist()}")
    
    # Save initial (pure noise) decoded
    initial_decoded = pipe.decode_rgb(target_latent)
    initial_decoded = torch.clip(initial_decoded, -1.0, 1.0)
    save_tensor_as_image(initial_decoded, os.path.join(output_dir, "01_initial_noise_decoded.png"))
    
    # Denoising loop with intermediate saves
    for i, t in enumerate(timesteps):
        logging.info(f"\n=== Step {i+1}/{num_steps}, timestep={t.item()} ===")
        
        # Normalize noisy latent before UNet
        if pipe.normalize_latents:
            current_latent_mean = target_latent.mean(dim=[2, 3], keepdim=True)
            current_latent_std = target_latent.std(dim=[2, 3], keepdim=True)
            target_latent_input = (target_latent - current_latent_mean) / (current_latent_std + 1e-8)
        else:
            target_latent_input = target_latent
        
        # Apply timestep-aware conditioning scaling (must match training)
        if pipe.cond_timestep_scaling:
            cond_scale = pipe._get_conditioning_scale(t)
            rgb_latent_scaled = rgb_latent_normalized * cond_scale
            logging.info(f"  cond_scale: {cond_scale.item():.4f}")
        else:
            rgb_latent_scaled = rgb_latent_normalized
        
        # Concat conditioning and noisy latent
        unet_input = torch.cat([rgb_latent_scaled, target_latent_input], dim=1)
        
        # Predict
        noise_pred = pipe.unet(unet_input, t, encoder_hidden_states=text_embed).sample
        
        logging.info(f"  noise_pred stats: min={noise_pred.min():.3f}, max={noise_pred.max():.3f}, mean={noise_pred.mean():.3f}")
        
        # Scheduler step
        step_output = pipe.scheduler.step(noise_pred, t, target_latent_input)
        target_latent = step_output.prev_sample
        
        # Restore normalization after scheduler step
        if pipe.normalize_latents:
            target_latent = target_latent * current_latent_std + current_latent_mean
        
        logging.info(f"  target_latent stats: min={target_latent.min():.3f}, max={target_latent.max():.3f}, mean={target_latent.mean():.3f}")
        
        # Decode and save intermediate result
        intermediate_rgb = pipe.decode_rgb(target_latent)
        intermediate_rgb = torch.clip(intermediate_rgb, -1.0, 1.0)
        
        save_tensor_as_image(
            intermediate_rgb, 
            os.path.join(output_dir, f"step_{i+1:02d}_t{t.item():04d}.png")
        )
        
        # Also save the predicted x0 if available
        if hasattr(step_output, 'pred_original_sample') and step_output.pred_original_sample is not None:
            pred_x0 = step_output.pred_original_sample
            pred_x0_rgb = pipe.decode_rgb(pred_x0)
            pred_x0_rgb = torch.clip(pred_x0_rgb, -1.0, 1.0)
            save_tensor_as_image(
                pred_x0_rgb,
                os.path.join(output_dir, f"step_{i+1:02d}_t{t.item():04d}_pred_x0.png")
            )
            logging.info(f"  pred_x0 stats: min={pred_x0.min():.3f}, max={pred_x0.max():.3f}")
    
    # Restore conditioning normalization for correct brightness and contrast
    if pipe.normalize_latents and cond_mean is not None and cond_std is not None:
        target_latent = (target_latent - target_latent.mean(dim=(2, 3), keepdim=True)) / (target_latent.std(dim=(2, 3), keepdim=True) + 1e-8)
        target_latent = target_latent * cond_std + cond_mean
        logging.info(f"Normalization: restored conditioning mean and std to final latent")
    
    # Final result
    final_rgb = pipe.decode_rgb(target_latent)
    final_rgb = torch.clip(final_rgb, -1.0, 1.0)
    save_tensor_as_image(final_rgb, os.path.join(output_dir, "99_final.png"))
    
    logging.info(f"\nResults saved to: {output_dir}")


def save_tensor_as_image(tensor: torch.Tensor, path: str):
    """Save tensor [-1, 1] as image"""
    tensor = (tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    tensor = tensor.squeeze().cpu().numpy()
    if tensor.ndim == 3:
        tensor = tensor.transpose(1, 2, 0)  # CHW -> HWC
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(tensor).save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug denoising steps")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--scheduler", type=str, choices=SCHEDULER_CHOICES, default="ddim",
                        help="Scheduler type: ddim, lcm, heun. Default: ddim")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.float32
    
    pipe = load_pipeline(args.checkpoint, args.base_model, dtype, args.scheduler)
    pipe = pipe.to(device)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    input_image = Image.open(args.input)
    
    with torch.no_grad():
        debug_denoising(pipe, input_image, args.steps, args.output_dir, device)
