# --------------------------------------------------------------------------
# Thesis Implementation: Hybrid ControlNet + Degraded Latent Concatenation
# Blind Image Restoration
#
# Inference script for Hybrid ControlNet restoration approach.
# Based on script/controlnet_restoration/run.py with key differences:
# - Uses MarigoldHybridControlNetRestorationPipeline (8ch UNet + ControlNet)
# - Requires --base_checkpoint (8ch UNet weights from base restoration training)
# - Requires --controlnet_checkpoint (trained ControlNet from hybrid training)
# - Replaces UNet conv_in to 8ch and loads base checkpoint UNet weights
# --------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import logging
import numpy as np
import torch
from glob import glob
from pathlib import Path
from PIL import Image
from torch.nn import Conv2d, Parameter
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from marigold import (
    MarigoldHybridControlNetRestorationPipeline,
    MarigoldHybridControlNetArniqa003Pipeline,
    MarigoldHybridControlNetArniqa003PipelinePatched,
)
from marigold.marigold_hybrid_controlnet_restoration_pipeline import MarigoldRestorationOutput

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
SCHEDULER_CHOICES = ["ddim", "lcm"]


def _replace_unet_conv_in(unet):
    """Replace UNet first layer to accept 8 in_channels.

    Exact copy of MarigoldHybridControlNetRestorationTrainer._replace_unet_conv_in
    (src/trainer/marigold_hybrid_controlnet_restoration_trainer.py, line 378).
    Channel layout: [0:4] = degraded (condition), [4:8] = noisy (target to denoise).
    """
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]

    _weight = _weight.repeat((1, 2, 1, 1))  # [320, 8, 3, 3]
    # half the activation magnitude
    _weight *= 0.5

    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    # replace config
    unet.config["in_channels"] = 8
    logging.info("UNet conv_in replaced: 4ch -> 8ch")


def load_hybrid_pipeline(
    base_checkpoint_path: str,
    controlnet_checkpoint_path: str,
    sd2_model: str,
    dtype: torch.dtype,
    scheduler_type: str = "ddim",
    pipeline_kwargs: dict = None,
):
    """
    Load Hybrid ControlNet restoration pipeline from trained checkpoints.

    Args:
        base_checkpoint_path: Path to base restoration checkpoint containing
            unet/diffusion_pytorch_model.safetensors (8ch UNet weights)
        controlnet_checkpoint_path: Path to hybrid training checkpoint containing
            controlnet/ subdirectory (trained ControlNet weights)
        sd2_model: HuggingFace model ID for the base SD2 model
        dtype: Torch dtype for frozen components (float16 recommended)
        scheduler_type: Scheduler type ("ddim" or "lcm")
        pipeline_kwargs: Additional kwargs for the pipeline constructor

    Returns:
        MarigoldHybridControlNetRestorationPipeline
    """
    # Validate base checkpoint
    base_path_obj = Path(base_checkpoint_path)
    unet_weights_path = base_path_obj / "unet" / "diffusion_pytorch_model.safetensors"
    if not unet_weights_path.exists():
        raise FileNotFoundError(
            f"Base checkpoint missing UNet weights: {unet_weights_path}. "
            f"Expected structure: {base_checkpoint_path}/unet/diffusion_pytorch_model.safetensors"
        )

    # Validate controlnet checkpoint
    controlnet_path_obj = Path(controlnet_checkpoint_path)
    controlnet_dir = controlnet_path_obj / "controlnet"
    if not controlnet_dir.exists():
        raise ValueError(
            f"ControlNet checkpoint '{controlnet_checkpoint_path}' does not contain "
            f"a controlnet/ subdirectory. Expected trained ControlNet weights at: {controlnet_dir}"
        )

    # Load SD2 base components (frozen, dtype to save VRAM)
    logging.info(f"Loading SD2 base model from: {sd2_model}")
    unet = UNet2DConditionModel.from_pretrained(
        sd2_model, subfolder="unet", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(
        sd2_model, subfolder="vae", torch_dtype=dtype
    )
    scheduler = DDIMScheduler.from_pretrained(
        sd2_model, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        sd2_model, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        sd2_model, subfolder="tokenizer"
    )
    logging.info("SD2 base components loaded")

    # Replace UNet conv_in to 8 channels
    # Pattern from trainer._replace_unet_conv_in (line 378)
    _replace_unet_conv_in(unet)

    # Load 8ch UNet weights from base checkpoint
    # Pattern from trainer._load_base_checkpoint_unet (line 328)
    from safetensors.torch import load_file
    unet_state_dict = load_file(str(unet_weights_path))
    unet.load_state_dict(unet_state_dict)
    logging.info(
        f"8ch UNet weights loaded from: {unet_weights_path} "
        f"(conv_in: {unet.conv_in.weight.shape[1]}ch)"
    )

    # Load trained ControlNet from hybrid checkpoint
    logging.info(f"Loading trained ControlNet from: {controlnet_dir}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_dir, torch_dtype=dtype
    )
    logging.info(
        f"ControlNet loaded: {sum(p.numel() for p in controlnet.parameters()) / 1e6:.1f}M params"
    )

    # Construct pipeline
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    pipe = MarigoldHybridControlNetRestorationPipeline(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        **pipeline_kwargs,
    )

    # Fix scheduler config to match training setup
    _fix_scheduler_config(pipe, scheduler_type)

    return pipe


def load_hybrid_003_pipeline(
    checkpoint_path: str,
    sd2_model: str,
    dtype: torch.dtype,
    scheduler_type: str = "ddim",
    pipeline_kwargs: dict = None,
):
    """
    Load Hybrid-003 ControlNet restoration pipeline from a single checkpoint.

    Hybrid-003 checkpoints contain both unet/ (already 8ch, trainable) and controlnet/
    in the same directory. No need for _replace_unet_conv_in or separate base checkpoint.

    Components are always loaded in fp32 so that autocast (inside single_infer) can
    keep softmax and layernorm in fp32, matching training validation precision.
    The dtype parameter is accepted for API compatibility but ignored for loading.

    Verified from MarigoldHybridControlNetArniqa003Trainer.save_checkpoint (line 1110):
    - Saves unet/ via self.model.unet.save_pretrained(unet_path)
    - Saves controlnet/ via self.model.controlnet.save_pretrained(controlnet_path)
    - Saves hybrid_003_config.json with {"architecture": "hybrid-003", ...}

    Args:
        checkpoint_path: Path to hybrid-003 checkpoint containing unet/ and controlnet/
        sd2_model: HuggingFace model ID for the base SD2 model
        dtype: Torch dtype (accepted for API compatibility; components load in fp32)
        scheduler_type: Scheduler type ("ddim" or "lcm")
        pipeline_kwargs: Additional kwargs for the pipeline constructor

    Returns:
        MarigoldHybridControlNetArniqa003Pipeline
    """
    checkpoint_path_obj = Path(checkpoint_path)
    unet_dir = checkpoint_path_obj / "unet"
    controlnet_dir = checkpoint_path_obj / "controlnet"

    if not unet_dir.exists():
        raise FileNotFoundError(
            f"Hybrid-003 checkpoint missing unet/ at: {unet_dir}"
        )
    if not controlnet_dir.exists():
        raise FileNotFoundError(
            f"Hybrid-003 checkpoint missing controlnet/ at: {controlnet_dir}"
        )

    # Load components using the requested dtype (fp16 saves VRAM).
    # Autocast inside single_infer handles precision-sensitive ops (softmax, layernorm).
    load_dtype = dtype

    # Load trained 8ch UNet directly from checkpoint
    # Verified: trainer saves via self.model.unet.save_pretrained() (line 1143)
    logging.info(f"Loading trained 8ch UNet from: {unet_dir}")
    unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=load_dtype)
    logging.info(
        f"UNet loaded: in_channels={unet.config.in_channels}, "
        f"conv_in: {unet.conv_in.weight.shape[1]}ch, "
        f"{sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M params"
    )

    # Load trained ControlNet from checkpoint
    # Verified: trainer saves via self.model.controlnet.save_pretrained() (line 1148)
    logging.info(f"Loading trained ControlNet from: {controlnet_dir}")
    controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=load_dtype)
    logging.info(
        f"ControlNet loaded: {sum(p.numel() for p in controlnet.parameters()) / 1e6:.1f}M params"
    )

    # Load frozen SD2 components (VAE, text_encoder, tokenizer, scheduler)
    logging.info(f"Loading SD2 frozen components from: {sd2_model}")
    vae = AutoencoderKL.from_pretrained(sd2_model, subfolder="vae", torch_dtype=load_dtype)
    scheduler = DDIMScheduler.from_pretrained(sd2_model, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        sd2_model, subfolder="text_encoder", torch_dtype=load_dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(sd2_model, subfolder="tokenizer")
    logging.info("SD2 frozen components loaded")

    # Construct pipeline using MarigoldHybridControlNetArniqa003PipelinePatched
    # Uses patched version with patch-based inference for large images (768x768 patches)
    # Verified: __init__ takes (unet, controlnet, vae, scheduler, text_encoder, tokenizer, patch_size, ...)
    # (marigold_hybrid_controlnet_arniqa_003_pipeline_patched.py)
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    pipe = MarigoldHybridControlNetArniqa003PipelinePatched(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        patch_size=768,
        overlap_ratio=0.25,
        blend_mode="gaussian",
        **pipeline_kwargs,
    )

    # Fix scheduler config to match training setup
    _fix_scheduler_config(pipe, scheduler_type)

    # Load ARNIQA adapter if present in checkpoint
    # Verified: trainer saves arniqa_adapter.pt with keys: stage, version, config, state_dict
    # (src/trainer/marigold_hybrid_controlnet_arniqa_003_trainer.py lines 1162-1178)
    arniqa_adapter_path = checkpoint_path_obj / "arniqa_adapter.pt"
    if arniqa_adapter_path.exists():
        from src.ARNIQA.model import ArniqaConditioner

        arniqa_checkpoint = torch.load(str(arniqa_adapter_path), map_location="cpu")
        arniqa_config = arniqa_checkpoint.get("config", {})

        # Verified: ArniqaConditioner.__init__ signature (src/ARNIQA/model.py line 625)
        conditioner = ArniqaConditioner(
            output_dim=arniqa_config.get("output_dim", 1024),
            conditioning_dropout=0.0,  # No dropout at inference
            stage=arniqa_config.get("stage", 1),
            spatial_size=arniqa_config.get("spatial_size", 24),
            num_tokens=arniqa_config.get("num_tokens", 1),
        )
        # Verified: trainer loads via conditioner.load_state_dict(checkpoint["state_dict"])
        # (trainer line 1267)
        conditioner.load_state_dict(arniqa_checkpoint["state_dict"])
        conditioner.eval()
        # Load in fp32 to match other components; autocast handles fp16 casting
        conditioner = conditioner.to(load_dtype)

        # Verified: pipeline.set_arniqa_conditioner() sets self.arniqa_conditioner
        # (marigold_hybrid_controlnet_arniqa_003_pipeline.py line 131)
        pipe.set_arniqa_conditioner(conditioner)
        logging.info(
            f"ARNIQA adapter loaded from {arniqa_adapter_path} "
            f"(stage={arniqa_config.get('stage')}, "
            f"spatial_size={arniqa_config.get('spatial_size')})"
        )
    else:
        logging.info("No arniqa_adapter.pt found, using empty text embedding")

    return pipe


def _fix_scheduler_config(pipe, scheduler_type: str = "ddim"):
    """Fix scheduler configuration to match training setup.

    CRITICAL: Training uses timestep_spacing="trailing" and rescale_betas_zero_snr=True,
    but SD2 defaults to "leading" and rescale_betas_zero_snr=False.

    Pattern copied from script/controlnet_restoration/run.py _fix_scheduler_config().
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
        description="Marigold : Hybrid ControlNet Image Restoration : Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to hybrid-003 checkpoint containing both unet/ and controlnet/. "
             "Auto-detects architecture from hybrid_003_config.json. "
             "Mutually exclusive with --base_checkpoint/--controlnet_checkpoint.",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=None,
        help="Path to base restoration checkpoint containing unet/ with 8ch UNet weights. "
             "(hybrid-002 mode, use with --controlnet_checkpoint)",
    )
    parser.add_argument(
        "--controlnet_checkpoint",
        type=str,
        default=None,
        help="Path to hybrid training checkpoint containing controlnet/ subdirectory. "
             "(hybrid-002 mode, use with --base_checkpoint)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-2",
        help="Base SD2 model for VAE/text_encoder/scheduler. Default: stabilityai/stable-diffusion-2",
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
    parser.add_argument(
        "--recycle_start_step",
        type=int,
        default=None,
        help="ControlNet Feature Recycling: step index (0-based) at which to start "
             "re-conditioning the ControlNet on the partially-restored image instead "
             "of the original degraded input. None or 0 disables recycling (default). "
             "Recommended: start at step 1 or 2 (after the first denoising step "
             "produces a usable x0 estimate).",
    )
    parser.add_argument(
        "--recycle_interval",
        type=int,
        default=1,
        help="ControlNet Feature Recycling: recycle every N steps after "
             "recycle_start_step. 1 = every step (default), 2 = every other step.",
    )

    args = parser.parse_args()

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

    # Auto-detect checkpoint architecture
    if args.checkpoint is not None:
        # Single-path mode: hybrid-003 checkpoint with both unet/ and controlnet/
        if args.base_checkpoint or args.controlnet_checkpoint:
            parser.error(
                "--checkpoint cannot be used with --base_checkpoint/--controlnet_checkpoint"
            )
        ckpt_path = Path(args.checkpoint)
        hybrid_003_config = ckpt_path / "hybrid_003_config.json"
        if hybrid_003_config.exists():
            logging.info(f"Detected hybrid-003 architecture from {hybrid_003_config}")
            with open(hybrid_003_config, "r") as f:
                h003_cfg = json.load(f)
            logging.info(f"Hybrid-003 config: {h003_cfg}")
        else:
            logging.info(
                f"No hybrid_003_config.json found at {ckpt_path}, "
                f"loading as hybrid-003 anyway (unet/ + controlnet/ present)"
            )

        pipe = load_hybrid_003_pipeline(
            checkpoint_path=args.checkpoint,
            sd2_model=args.base_model,
            dtype=dtype,
            scheduler_type=args.scheduler,
        )
    elif args.base_checkpoint and args.controlnet_checkpoint:
        # Two-path mode: hybrid-002 with separate base + controlnet checkpoints
        logging.info("Using hybrid-002 mode: separate base + controlnet checkpoints")
        pipe = load_hybrid_pipeline(
            base_checkpoint_path=args.base_checkpoint,
            controlnet_checkpoint_path=args.controlnet_checkpoint,
            sd2_model=args.base_model,
            dtype=dtype,
            scheduler_type=args.scheduler,
        )
    else:
        parser.error(
            "Either --checkpoint (hybrid-003) or both "
            "--base_checkpoint and --controlnet_checkpoint (hybrid-002) are required"
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
    if args.recycle_start_step and args.recycle_start_step > 0:
        logging.info(
            f"ControlNet Feature Recycling ENABLED: "
            f"start_step={args.recycle_start_step}, interval={args.recycle_interval}"
        )
    else:
        logging.info("ControlNet Feature Recycling disabled")

    # -------------------- Inference --------------------
    with torch.no_grad():
        for rgb_path in tqdm(rgb_filename_list, desc="Hybrid ControlNet Restoration", leave=True):
            input_image = Image.open(rgb_path)

            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Verified: MarigoldHybridControlNetRestorationPipeline.__call__ at line 317
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
                recycle_start_step=args.recycle_start_step,
                recycle_interval=args.recycle_interval,
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
