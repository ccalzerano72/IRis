# Marigold Restoration Inference Script
# Thesis Project: Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import numpy as np
import os
import torch
from pathlib import Path
from PIL import Image
from glob import glob
from tqdm.auto import tqdm

from marigold import MarigoldRestorationPipeline, MarigoldRestorationOutput

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

# Supported scheduler types
SCHEDULER_CHOICES = ["ddim", "lcm", "heun"]


def load_pipeline(checkpoint_path: str, base_model: str, dtype: torch.dtype, variant: str, scheduler_type: str = "ddim"):
    """
    Load restoration pipeline, supporting both HuggingFace hub models and local trained checkpoints.
    
    Pattern copied from webserver_restoration/utils/model_loader.py (verified working)
    
    Args:
        checkpoint_path: Path to checkpoint (HuggingFace hub name or local path with unet/ subdirectory)
        base_model: Base model to use when loading local checkpoints
        dtype: Torch dtype (float16 or float32)
        variant: Variant string for HuggingFace loading
        scheduler_type: Scheduler type ("ddim", "lcm", "heun"). Default: "ddim"
    
    Returns:
        MarigoldRestorationPipeline: Loaded pipeline
    """
    checkpoint_path_obj = Path(checkpoint_path)
    
    # Check if this is a local checkpoint with unet/ subdirectory
    unet_checkpoint_path = checkpoint_path_obj / "unet"
    is_local_unet_checkpoint = unet_checkpoint_path.exists()
    
    # Check if this is a full diffusers model (has model_index.json)
    model_index_path = checkpoint_path_obj / "model_index.json"
    is_full_diffusers_model = model_index_path.exists()
    
    if is_local_unet_checkpoint:
        # Load base model first, then replace U-Net with trained weights
        # Pattern from webserver_restoration/utils/model_loader.py lines 55-72
        logging.info(f"Loading base model from: {base_model}")
        pipe = MarigoldRestorationPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype
        )
        logging.info(f"✓ Base model loaded from: {base_model}")
        
        # Load trained U-Net weights
        logging.info(f"Loading trained U-Net from: {unet_checkpoint_path}")
        from diffusers import UNet2DConditionModel
        pipe.unet = UNet2DConditionModel.from_pretrained(
            unet_checkpoint_path,
            torch_dtype=dtype
        )
        logging.info("✓ Loaded trained U-Net weights from checkpoint")
        
    elif is_full_diffusers_model or "/" in checkpoint_path:
        # Load directly from HuggingFace hub or full diffusers model
        logging.info(f"Loading full model from: {checkpoint_path}")
        pipe = MarigoldRestorationPipeline.from_pretrained(
            checkpoint_path, variant=variant, torch_dtype=dtype
        )
        logging.info(f"✓ Loaded model from: {checkpoint_path}")
    else:
        raise ValueError(
            f"Checkpoint path '{checkpoint_path}' is not valid. "
            f"Expected either a HuggingFace hub name, a path with model_index.json, "
            f"or a path with unet/ subdirectory containing trained weights."
        )
    
    # Fix scheduler configuration for ALL loading paths
    # This is critical: training uses trailing/rescale_betas_zero_snr=True,
    # but SD2 defaults to leading/rescale_betas_zero_snr=False
    _fix_scheduler_config(pipe, scheduler_type)
    
    # Load restoration config (inference-relevant settings like normalize_latents)
    # This ensures the pipeline is configured to match training settings
    _load_restoration_config(pipe, checkpoint_path_obj)
    
    # Load ARNIQA conditioner if checkpoint was trained with ARNIQA
    # Supports Stage 1, Stage 2, and classification mode
    _load_arniqa_conditioner(pipe, checkpoint_path_obj)
    
    return pipe


def _load_restoration_config(pipe, checkpoint_path: Path):
    """
    Load restoration config from checkpoint if available.
    
    This loads inference-relevant settings (like normalize_latents) that were
    saved during training. If the config file doesn't exist (old checkpoint),
    defaults are used for backward compatibility.
    
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
        
        logging.info(f"✓ Loaded restoration config from: {config_path}")
        logging.info(f"  normalize_latents: {normalize_latents}")
    else:
        # Old checkpoint without restoration_config.json - use defaults
        logging.info("No restoration_config.json found (old checkpoint), using defaults")
        pipe.set_normalize_latents(False)


def _load_arniqa_conditioner(pipe, checkpoint_path: Path):
    """
    Load ARNIQA conditioner from checkpoint if available.
    
    If the checkpoint contains arniqa_adapter.pt, this function instantiates
    an ArniqaConditioner with the saved config, loads the trained weights,
    and attaches it to the pipeline. Supports all variants:
    - Stage 1: Global-only conditioning (1 or more tokens)
    - Stage 2: Global + Spatial conditioning
    - Classification mode: Same as Stage 1 with different conceptual usage
    
    If arniqa_adapter.pt doesn't exist, the checkpoint was trained without
    ARNIQA and no conditioner is loaded (backward compatible).
    
    NOTE: The conditioner is loaded on CPU. It must be moved to the target
    device after pipe.to(device), since it's a plain attribute not a
    registered Diffusers component.
    
    Args:
        pipe: The pipeline to configure
        checkpoint_path: Path to the checkpoint directory
    """
    adapter_path = checkpoint_path / "arniqa_adapter.pt"
    
    if not adapter_path.exists():
        logging.info("No arniqa_adapter.pt found — checkpoint trained without ARNIQA")
        return
    
    logging.info(f"Loading ARNIQA adapter from: {adapter_path}")
    
    from src.ARNIQA import ArniqaConditioner
    
    arniqa_ckpt = torch.load(adapter_path, map_location="cpu")
    config = arniqa_ckpt["config"]
    
    # Instantiate conditioner with saved config
    conditioner = ArniqaConditioner(
        output_dim=config["output_dim"],
        conditioning_dropout=0.0,  # No dropout during inference
        stage=config["stage"],
        spatial_size=config.get("spatial_size") or 24,
        num_tokens=config.get("num_tokens", 1),
    )
    
    # Load trained adapter weights (with backward compatibility for old checkpoints)
    state_dict = arniqa_ckpt["state_dict"]
    
    # Remap old key names: in older checkpoints, LayerNorm was inside the
    # Sequential as mlp.4, but current code has it as a separate layer_norm
    key_remap = {
        "global_adapter.mlp.4.weight": "global_adapter.layer_norm.weight",
        "global_adapter.mlp.4.bias": "global_adapter.layer_norm.bias",
    }
    remapped = False
    for old_key, new_key in key_remap.items():
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
            remapped = True
    if remapped:
        logging.info("  Remapped old checkpoint keys (mlp.4 → layer_norm)")
    
    conditioner.load_state_dict(state_dict)
    conditioner.eval()
    
    # Attach to pipeline (still on CPU, will be moved with pipe later)
    pipe.set_arniqa_conditioner(conditioner)
    
    stage = config["stage"]
    version = arniqa_ckpt.get("version", "1.0")
    num_tokens = config.get("num_tokens", 1)
    spatial_size = config.get("spatial_size")
    
    logging.info(f"✓ ARNIQA conditioner loaded (version={version}, stage={stage})")
    if stage >= 2 and spatial_size:
        total_tokens = 1 + spatial_size * spatial_size
        logging.info(f"  Stage 2: 1 global + {spatial_size}×{spatial_size}={spatial_size**2} spatial = {total_tokens} tokens")
    else:
        logging.info(f"  Stage 1: {num_tokens} global token(s)")


def _fix_scheduler_config(pipe, scheduler_type: str = "ddim"):
    """
    Fix scheduler configuration to match training setup.
    
    CRITICAL: Training uses timestep_spacing="trailing" and rescale_betas_zero_snr=True,
    but SD2 defaults to timestep_spacing="leading" and rescale_betas_zero_snr=False.
    
    This mismatch causes inference quality to degrade with more denoising steps because:
    - "leading" timesteps don't reach the maximum noise level seen during training
    - rescale_betas_zero_snr=False causes incorrect SNR scaling for v_prediction
    
    The scheduler must be RECREATED (not just config modified) to apply these changes.
    
    Args:
        pipe: The pipeline to fix
        scheduler_type: "ddim", "lcm", or "heun"
    """
    from diffusers import DDIMScheduler, LCMScheduler, HeunDiscreteScheduler
    
    # Log original config for debugging
    orig_spacing = getattr(pipe.scheduler.config, 'timestep_spacing', 'unknown')
    orig_rescale = getattr(pipe.scheduler.config, 'rescale_betas_zero_snr', False)
    orig_type = type(pipe.scheduler).__name__
    
    # Recreate scheduler with correct config based on scheduler_type
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
    logging.info(
        f"✓ Scheduler: {orig_type} → {new_type}, "
        f"timestep_spacing: {orig_spacing} → trailing, "
        f"rescale_betas_zero_snr: {orig_rescale} → True"
    )


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Image Restoration : Multi-image Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-restoration-v1-0",
        help="Checkpoint path (HuggingFace hub name or local path with unet/ subdirectory).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-2",
        help="Base model to use when loading local checkpoints with unet/ subdirectory. "
        "Default: `stabilityai/stable-diffusion-2`",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input degraded image folder.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. If set to "
        "`None`, default value will be read from checkpoint.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Resolution to which the input is resized before performing restoration. `0` uses the original input "
        "resolution; `None` resolves the best default from the model checkpoint. Default: `None`",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled. Default: `1`.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Setting this flag will output the result at the effective value of `processing_res`, otherwise the "
        "output will be resized to the input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or "
        "`nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for randomized inference. Default: `None`",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Use Apple Silicon for faster inference (subject to availability).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (slow but works when GPU is unavailable or full).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=SCHEDULER_CHOICES,
        default="ddim",
        help="Scheduler type for inference. Choices: ddim, lcm, heun. Default: ddim",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-Free Guidance scale. Values > 1.0 increase conditioning influence. "
        "Requires model trained with CFG (conditioning_dropout_prob > 0). Default: 1.0 (no CFG)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="Number of CPU threads for inference. 0 = use all available cores. "
        "Only relevant when using --cpu. Default: 0",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Maximum number of images to process. 0 = all images. Default: 0",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    base_model = args.base_model
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, "
            "due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    force_cpu = args.cpu
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Output directories
    output_dir_restored = os.path.join(output_dir, "restored")
    output_dir_npy = os.path.join(output_dir, "restored_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_restored, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if force_cpu:
        device = torch.device("cpu")
        logging.info("Forcing CPU inference (--cpu flag set).")
        
        # CPU optimizations
        # 1. Set number of threads (0 = all available cores)
        num_threads = args.num_threads
        if num_threads <= 0:
            num_threads = os.cpu_count() or 1
        torch.set_num_threads(num_threads)
        logging.info(f"CPU threads: {num_threads}")
        
        # 2. Force float32 on CPU (fp16 is slower on CPU due to emulation)
        if half_precision:
            logging.warning("Overriding --half_precision: float32 is faster on CPU (fp16 requires emulation)")
            half_precision = False
        
        # 3. Set matmul precision for better CPU performance
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')
            logging.info("Set float32 matmul precision to 'medium' for faster CPU inference")
        
    elif apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)
    
    # Filter out already-processed images
    rgb_filename_list_filtered = []
    skipped_count = 0
    for rgb_path in rgb_filename_list:
        rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
        pred_name_base = rgb_name_base + "_restored"
        restored_save_path = os.path.join(output_dir_restored, f"{pred_name_base}.png")
        
        if os.path.exists(restored_save_path):
            skipped_count += 1
        else:
            rgb_filename_list_filtered.append(rgb_path)
    
    if skipped_count > 0:
        logging.info(f"Skipping {skipped_count} already-processed images")
    
    if len(rgb_filename_list_filtered) == 0:
        logging.info("All images already processed. Nothing to do.")
        exit(0)
    
    logging.info(f"Processing {len(rgb_filename_list_filtered)} images")
    rgb_filename_list = rgb_filename_list_filtered

    # Apply max_images limit if set
    max_images = args.max_images
    if max_images > 0 and len(rgb_filename_list) > max_images:
        logging.info(f"Limiting to {max_images} images (out of {len(rgb_filename_list)})")
        rgb_filename_list = rgb_filename_list[:max_images]

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    # Load pipeline (supports both HuggingFace hub and local checkpoints with unet/)
    pipe = load_pipeline(checkpoint_path, base_model, dtype, variant, args.scheduler)

    # Enable xformers only on GPU (not available on CPU)
    if not force_cpu and device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass  # run without xformers

    pipe = pipe.to(device)
    
    # Move ARNIQA conditioner to device (not a registered Diffusers component,
    # so pipe.to(device) doesn't move it automatically)
    if pipe.arniqa_conditioner is not None:
        pipe.arniqa_conditioner = pipe.arniqa_conditioner.to(device=device, dtype=dtype)
        logging.info(f"✓ ARNIQA conditioner moved to {device} ({dtype})")
    
    logging.info(f"Loaded restoration pipeline")

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}."
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path in tqdm(rgb_filename_list, desc="Restoration Inference", leave=True):
            # Read input image
            input_image = Image.open(rgb_path)

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Perform inference
            pipe_out: MarigoldRestorationOutput = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
                guidance_scale=args.guidance_scale,
            )

            restored_pred: np.ndarray = pipe_out.restored_np
            restored_img: Image.Image = pipe_out.restored_img

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_restored"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, restored_pred)

            # Save restored image
            restored_save_path = os.path.join(output_dir_restored, f"{pred_name_base}.png")
            if os.path.exists(restored_save_path):
                logging.warning(f"Existing file: '{restored_save_path}' will be overwritten")
            restored_img.save(restored_save_path)