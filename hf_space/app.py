# IRis: Blind Image Restoration via Dual-Conditioned Latent Diffusion
# HuggingFace Spaces (ZeroGPU) demo app.

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # must be imported before torch

import logging
import tempfile

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    UNet2DConditionModel,
)
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from marigold import MarigoldHybridControlNetArniqa003PipelinePatched

logging.basicConfig(level=logging.INFO)

MODEL_REPO = "ccalzerano72/IRis-hybrid-003"
SD2_MODEL = "sd2-community/stable-diffusion-2-1"
DTYPE = torch.float16

# ---------------------------------------------------------------------------
# Load model at module scope so that ZeroGPU can "pack" the weights once at
# startup and stream them to a GPU worker on demand.
# ---------------------------------------------------------------------------
print("Downloading IRis checkpoint...")
CHECKPOINT_DIR = snapshot_download(repo_id=MODEL_REPO, repo_type="model")
print(f"Checkpoint cached at: {CHECKPOINT_DIR}")

print("Loading SD2 frozen components (VAE, text encoder, tokenizer, scheduler)...")
vae = AutoencoderKL.from_pretrained(SD2_MODEL, subfolder="vae", torch_dtype=DTYPE)
text_encoder = CLIPTextModel.from_pretrained(
    SD2_MODEL, subfolder="text_encoder", torch_dtype=DTYPE
)
tokenizer = CLIPTokenizer.from_pretrained(SD2_MODEL, subfolder="tokenizer")
scheduler = DDIMScheduler.from_pretrained(SD2_MODEL, subfolder="scheduler")

# Training uses timestep_spacing="trailing" and rescale_betas_zero_snr=True
# (see script/hybrid_controlnet_restoration/run.py::_fix_scheduler_config).
scheduler = DDIMScheduler.from_config(
    scheduler.config,
    timestep_spacing="trailing",
    rescale_betas_zero_snr=True,
)

print("Loading hybrid-003 UNet (8ch) and ControlNet...")
unet = UNet2DConditionModel.from_pretrained(
    os.path.join(CHECKPOINT_DIR, "unet"), torch_dtype=DTYPE
)
controlnet = ControlNetModel.from_pretrained(
    os.path.join(CHECKPOINT_DIR, "controlnet"), torch_dtype=DTYPE
)

print("Building restoration pipeline...")
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
# Keep GPU memory usage bounded for ensembles (1 prediction at a time)
pipe = pipe.to("cuda")
print("Pipeline ready.")


def _estimate_duration(image, steps, seed, *args, **kwargs):
    # Rough upper bound: ~15 s overhead + ~15 s per denoising step (per patch).
    return min(180, 30 + int(steps) * 15)


@spaces.GPU(duration=_estimate_duration)
def restore(image, steps, seed):
    """Restore a single degraded image."""
    import traceback

    img = Image.fromarray(image).convert("RGB")

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    try:
        out = pipe(
            input_image=img,
            denoising_steps=int(steps),
            ensemble_size=1,
            processing_res=768,  # training resolution
            match_input_res=True,
            show_progress_bar=False,
            generator=generator,
            guidance_scale=1.0,
        )
        return out.restored_img
    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown(
        """
# IRis — Blind Image Restoration

Restore a degraded image (noise, blur, JPEG artifacts, upscaling defects) with a
diffusion model trained for blind image restoration.

**IRis (Image Restoration via latent diffusion)** repurposes a Stable Diffusion 2
UNet (expanded to 8 input channels) together with a ControlNet branch, trained
jointly on synthetic degradations. See the [project repository](https://github.com/ccalzerano72/IRis)
for details, metrics and the thesis results.
        """
    )
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Degraded input", sources=["upload", "clipboard"])
            steps = gr.Slider(1, 50, value=5, step=1, label="Denoising steps")
            seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
            run_btn = gr.Button("Restore", variant="primary")
        with gr.Column():
            output_img = gr.Image(type="pil", label="Restored output")

    run_btn.click(
        fn=restore,
        inputs=[input_img, steps, seed],
        outputs=output_img,
    )

    gr.Markdown(
        """
- **Runs on shared ZeroGPU hardware** — the first request after a sleep can be
  slower while the model is streamed to the GPU.
- **Free daily GPU quota** applies per user (5 minutes). Fewer denoising steps
  are faster and cheaper; 5 steps already give strong results.
- Trained on 768×768 patches; the output is resized back to the input size.
        """
    )

demo.queue().launch()