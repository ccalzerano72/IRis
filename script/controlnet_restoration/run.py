# --------------------------------------------------------------------------
# Thesis Implementation: ControlNet-Based Blind Image Restoration
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# Inference script for ControlNet restoration approach.
# Based on script/restoration/run.py but adapted for ControlNet architecture:
# - Loads SD2 components individually + trained ControlNet from checkpoint
# - No ARNIQA conditioner, no normalize_latents
# - Supports CFG via guidance_scale parameter
# --------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import numpy as np
import torch
from glob import glob
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from marigold import MarigoldControlNetRestorationPipeline
from marigold.marigold_controlnet_restoration_pipeline import MarigoldRestorationOutput

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
SCHEDULER_CHOICES = ["ddim", "lcm"]


def load_controlnet_pipeline(
    checkpoint_path: str,
    base_model: str,
    dtype: torch.dtype,
    scheduler_type: str = "ddim",
    pipeline_kwargs: dict = None,
    load_unet_from_checkpoint: bool = False,
):
    """
    Load ControlNet restoration pipeline from a trained checkpoint.

    The checkpoint must contain a controlnet/ subdirectory with the trained
    ControlNet weights (saved via save_pretrained during training).

    Args:
        checkpoint_path: Path to checkpoint directory containing controlnet/ subdirectory
        base_model: HuggingFace model ID for the base SD2 model
        dtype: Torch dtype for frozen components (float16 recommended)
        scheduler_type: Scheduler type ("ddim" or "lcm")
        pipeline_kwargs: Additional kwargs for the pipeline constructor
        load_unet_from_checkpoint: If True, load trained UNet from checkpoint/unet/
            instead of from the base SD2 model. Used for re_004 (4ch trainable UNet).

    Returns:
        MarigoldControlNetRestorationPipeline
    """
    checkpoint_path_obj = Path(checkpoint_path)
    controlnet_path = checkpoint_path_obj / "controlnet"

    if not controlnet_path.exists():
        raise ValueError(
            f"Checkpoint path '{checkpoint_path}' does not contain a controlnet/ subdirectory. "
            f"Expected trained ControlNet weights at: {controlnet_path}"
        )

    # Load UNet: from checkpoint (trainable 4ch) or from SD2 base (frozen)
    if load_unet_from_checkpoint:
        unet_path = checkpoint_path_obj / "unet"
        if not unet_path.exists():
            raise ValueError(
                f"--load_unet_from_checkpoint set but no unet/ found at: {unet_path}"
            )
        logging.info(f"Loading trained UNet from checkpoint: {unet_path}")
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=dtype)
        logging.info(
            f"UNet loaded from checkpoint: in_channels={unet.config.in_channels}, "
            f"{sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M params"
        )
    else:
        logging.info(f"Loading SD2 base UNet from: {base_model}")
        unet = UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet", torch_dtype=dtype
        )

    # Load remaining SD2 base components (frozen, float16 to save VRAM)
    logging.info(f"Loading SD2 base components from: {base_model}")
    vae = AutoencoderKL.from_pretrained(
        base_model, subfolder="vae", torch_dtype=dtype
    )
    scheduler = DDIMScheduler.from_pretrained(
        base_model, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model, subfolder="tokenizer"
    )
    logging.info("SD2 base components loaded")

    # Load trained ControlNet from checkpoint
    logging.info(f"Loading trained ControlNet from: {controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=dtype
    )
    logging.info(
        f"ControlNet loaded: {sum(p.numel() for p in controlnet.parameters()) / 1e6:.1f}M params"
    )

    # Construct pipeline
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    pipe = MarigoldControlNetRestorationPipeline(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        **pipeline_kwargs,
    )

    # Fix scheduler config to match training setup
    # Training uses trailing timestep_spacing and rescale_betas_zero_snr=True
    _fix_scheduler_config(pipe, scheduler_type)

    return pipe


def _fix_scheduler_config(pipe, scheduler_type: str = "ddim"):
    """
    Fix scheduler configuration to match training setup.

    CRITICAL: Training uses timestep_spacing="trailing" and rescale_betas_zero_snr=True,
    but SD2 defaults to "leading" and rescale_betas_zero_snr=False.

    Pattern copied from script/restoration/run.py _fix_scheduler_config().
    """
    orig_spacing = getattr(pipe.scheduler.config, 'timestep_spacing', 'unknown')
    orig_rescale = getattr(pipe.scheduler.config, 'rescale_betas_zero_snr', False)

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
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose from: {SCHEDULER_CHOICES}")

    new_type = type(pipe.scheduler).__name__
    logging.info(
        f"Scheduler fixed: timestep_spacing {orig_spacing} -> trailing, "
        f"rescale_betas_zero_snr {orig_rescale} -> True, type: {new_type}"
    )


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : ControlNet Image Restoration : Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory containing controlnet/ subdirectory.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-2",
        help="Base SD2 model for frozen UNet/VAE/text_encoder. Default: stabilityai/stable-diffusion-2",
    )
    parser.add_argument(
        "--load_unet_from_checkpoint",
        action="store_true",
        help="Load trained UNet from checkpoint/unet/ instead of SD2 base. "
             "Used for 4ch trainable UNet + ControlNet (re_004).",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input degraded image folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Denoising steps. None uses default from pipeline config.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Processing resolution. 0 = native resolution. None = default from config.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to ensemble. Default: 1.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="CFG guidance scale. >1.0 increases conditioning. Default: 1.0 (no CFG).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=SCHEDULER_CHOICES,
        default="ddim",
        help="Scheduler type. Default: ddim.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (float16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. None = random.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. 0 = automatic.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method for resizing. Default: bilinear.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Output at processing resolution instead of input resolution.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Max images to process. 0 = all.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    seed = args.seed
    batch_size = args.batch_size

    # -------------------- Preparation --------------------
    output_dir_restored = os.path.join(output_dir, "restored")
    output_dir_npy = os.path.join(output_dir, "restored_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_restored, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA not available. Running on CPU will be slow.")
    logging.info(f"Device: {device}")

    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)

    if n_images == 0:
        logging.error(f"No images found in '{input_rgb_dir}'")
        exit(1)
    logging.info(f"Found {n_images} images")

    # Skip already-processed images
    rgb_filename_list_filtered = []
    skipped_count = 0
    for rgb_path in rgb_filename_list:
        rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
        restored_save_path = os.path.join(output_dir_restored, f"{rgb_name_base}_restored.png")
        if os.path.exists(restored_save_path):
            skipped_count += 1
        else:
            rgb_filename_list_filtered.append(rgb_path)

    if skipped_count > 0:
        logging.info(f"Skipping {skipped_count} already-processed images")

    if len(rgb_filename_list_filtered) == 0:
        logging.info("All images already processed.")
        exit(0)

    rgb_filename_list = rgb_filename_list_filtered

    # Apply max_images limit
    max_images = args.max_images
    if max_images > 0 and len(rgb_filename_list) > max_images:
        logging.info(f"Limiting to {max_images} images")
        rgb_filename_list = rgb_filename_list[:max_images]

    logging.info(f"Processing {len(rgb_filename_list)} images")

    # -------------------- Model --------------------
    dtype = torch.float16 if args.half_precision else torch.float32

    pipe = load_controlnet_pipeline(
        checkpoint_path=checkpoint_path,
        base_model=args.base_model,
        dtype=dtype,
        scheduler_type=args.scheduler,
        load_unet_from_checkpoint=args.load_unet_from_checkpoint,
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass

    pipe = pipe.to(device)
    logging.info(
        f"Pipeline loaded. denoise_steps={denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size={ensemble_size}, "
        f"processing_res={processing_res or pipe.default_processing_resolution}, "
        f"guidance_scale={args.guidance_scale}, seed={seed}"
    )

    # -------------------- Inference --------------------
    with torch.no_grad():
        for rgb_path in tqdm(rgb_filename_list, desc="ControlNet Restoration", leave=True):
            input_image = Image.open(rgb_path)

            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Verified: MarigoldControlNetRestorationPipeline.__call__ at line 294
            # Returns MarigoldRestorationOutput with restored_np, restored_img
            pipe_out: MarigoldRestorationOutput = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                show_progress_bar=True,
                resample_method=args.resample_method,
                generator=generator,
                guidance_scale=args.guidance_scale,
            )

            restored_pred: np.ndarray = pipe_out.restored_np
            restored_img: Image.Image = pipe_out.restored_img

            # Save outputs
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_restored"

            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            np.save(npy_save_path, restored_pred)

            restored_save_path = os.path.join(output_dir_restored, f"{pred_name_base}.png")
            restored_img.save(restored_save_path)

    logging.info(f"Done. Results saved to: {output_dir}")
