# IRis backend: hybrid-003 (8ch UNet + ControlNet) restoration pipeline.
# Loading logic mirrors script/hybrid_controlnet_restoration/run.py::load_hybrid_003_pipeline
# (ARNIQA branch omitted: checkpoint 002_re_015000 has arniqa_enabled=false).

import logging
import os

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    UNet2DConditionModel,
)
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

from marigold import MarigoldHybridControlNetArniqa003PipelinePatched

MODEL_REPO = "ccalzerano72/IRis-hybrid-003"
SD2_MODEL = "sd2-community/stable-diffusion-2-1"


def load_iris(dtype: torch.dtype = torch.float16):
    logging.info("Downloading IRis checkpoint...")
    ckpt_dir = snapshot_download(repo_id=MODEL_REPO, repo_type="model")

    logging.info("Loading SD2 frozen components...")
    vae = AutoencoderKL.from_pretrained(SD2_MODEL, subfolder="vae", torch_dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(
        SD2_MODEL, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(SD2_MODEL, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(SD2_MODEL, subfolder="scheduler")
    # Match training: timestep_spacing="trailing", rescale_betas_zero_snr=True
    scheduler = DDIMScheduler.from_config(
        scheduler.config,
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
    )

    logging.info("Loading hybrid-003 UNet (8ch) and ControlNet...")
    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(ckpt_dir, "unet"), torch_dtype=dtype
    )
    controlnet = ControlNetModel.from_pretrained(
        os.path.join(ckpt_dir, "controlnet"), torch_dtype=dtype
    )

    pipe = MarigoldHybridControlNetArniqa003PipelinePatched(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        default_denoising_steps=5,
        default_processing_resolution=768,
        patch_size=768,
        overlap_ratio=0.25,
        blend_mode="gaussian",
    )
    logging.info("IRis pipeline built.")
    return pipe


def run_iris(pipe, image, steps: int = 5, seed: int = -1):
    """Run IRis restoration. `image` is a PIL RGB image. Returns PIL RGB."""
    from PIL import Image

    img = image.convert("RGB")
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        input_image=img,
        denoising_steps=int(steps),
        ensemble_size=1,
        processing_res=768,
        match_input_res=True,
        show_progress_bar=False,
        generator=generator,
        guidance_scale=1.0,
    )
    result: Image.Image = out.restored_img
    return result
