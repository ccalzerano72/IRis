#!/usr/bin/env python3
"""
StableSR Inference Wrapper for Evaluation.

Wraps StableSR's inference pipeline to handle arbitrary image sizes correctly.
Fixes the latent-space tiling bug in the original canvas script where images
with latent dimensions smaller than tile_size cause tensor size mismatches.

Solution: pad the upscaled image so that BOTH latent dimensions are >= tile_size
before encoding. After sampling and decoding, crop back to the correct output size.

Must be run from the StableSR directory with taming-transformers on PYTHONPATH:
    cd external/StableSR
    PYTHONPATH=.:path/to/taming-transformers python path/to/infer_stablesr.py ...

Based on: scripts/sr_val_ddim_text_T_negativeprompt_canvas.py
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


# ---------------------------------------------------------------------------
# Helper functions — copied exactly from StableSR scripts
# ---------------------------------------------------------------------------

def space_timesteps(num_timesteps, section_counts):
    """
    Copied from scripts/sr_val_ddim_text_T_negativeprompt_canvas.py lines 56-109.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def load_model_from_config(config, ckpt, verbose=False):
    """
    Copied from scripts/sr_val_ddim_text_T_negativeprompt_canvas.py lines 118-135.
    """
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


def load_img(path):
    """
    Load image at ORIGINAL size (no rounding). Returns [-1, 1] tensor.
    Unlike the original StableSR load_img which rounds down to 32-multiple,
    we preserve the exact dimensions and handle padding separately in
    process_single_image. This ensures the output matches GT dimensions.
    """
    image = Image.open(path).convert("RGB")
    w, h = image.size
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_single_image(
    img_path,
    model,
    vq_model,
    sampler,
    ddim_timesteps,
    device,
    upscale=4.0,
    input_size=768,
    tile_overlap=48,
    use_negative_prompt=True,
    scale=7.0,
    colorfix_type="wavelet",
):
    """
    Process a single image through StableSR.

    Handles the latent-padding fix: if the upscaled image produces a latent
    with any dimension < tile_size (input_size//8), we pad the pixel image
    with reflection so that the latent is large enough for tiling. After
    decoding we crop back to the correct output size.

    Flow copied from sr_val_ddim_text_T_negativeprompt_canvas.py lines 356-400.
    """
    tile_size = input_size // 8  # 768 -> 96

    # 1. Load at original size and upscale (upscale=1 for same-res restoration)
    cur_image = load_img(img_path).to(device)
    if upscale != 1.0:
        cur_image = F.interpolate(
            cur_image,
            size=(int(cur_image.size(-2) * upscale),
                  int(cur_image.size(-1) * upscale)),
            mode='bicubic',
        )
    init_image = cur_image  # 1 x 3 x H x W, [-1, 1]
    # This is the exact target output size (must match GT)
    target_h, target_w = init_image.shape[2], init_image.shape[3]

    # 2. Pad UP to multiple of 32 (never round down — preserves all pixels)
    pad_h = (32 - target_h % 32) % 32
    pad_w = (32 - target_w % 32) % 32

    # 3. LATENT PADDING FIX: ensure both latent dims >= tile_size.
    #    After pad-to-32, the latent is (target_h+pad_h)/8 x (target_w+pad_w)/8.
    #    If either < tile_size, add more pixel padding so latent >= tile_size.
    latent_h_after_32pad = (target_h + pad_h) // 8
    latent_w_after_32pad = (target_w + pad_w) // 8

    extra_pad_h = 0
    extra_pad_w = 0
    if latent_h_after_32pad < tile_size:
        # Need latent_h >= tile_size, so pixel_h >= tile_size * 8
        target_pixel_h = tile_size * 8
        extra_pad_h = target_pixel_h - (target_h + pad_h)
    if latent_w_after_32pad < tile_size:
        target_pixel_w = tile_size * 8
        extra_pad_w = target_pixel_w - (target_w + pad_w)

    total_pad_h = pad_h + extra_pad_h
    total_pad_w = pad_w + extra_pad_w

    if total_pad_h > 0 or total_pad_w > 0:
        init_image = F.pad(init_image, (0, total_pad_w, 0, total_pad_h), mode='replicate')

    # 4. Encode to latent (canvas script lines 367-368)
    init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
    init_latent = model.get_first_stage_encoding(init_latent_generator)

    # 5. Text conditioning (canvas script lines 370-383)
    text_init = [''] * init_image.size(0)
    semantic_c = model.cond_stage_model(text_init)

    nega_semantic_c = None
    if use_negative_prompt:
        negative_text_init = [
            '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)'
        ] * init_image.size(0)
        nega_semantic_c = model.cond_stage_model(negative_text_init)

    # 6. Sample (canvas script lines 385-398)
    #    x_T = None means start from pure noise (canvas script sets x_T = None)
    samples, _ = sampler.ddim_sampling_sr_t_canvas(
        cond=semantic_c,
        struct_cond=init_latent,
        shape=init_latent.shape,
        unconditional_conditioning=nega_semantic_c if use_negative_prompt else None,
        unconditional_guidance_scale=scale if use_negative_prompt else None,
        timesteps=np.array(ddim_timesteps),
        x_T=None,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        batch_size=1,
    )

    # 7. Decode (canvas script line 400)
    x_samples = vq_model.decode(samples * 1.0 / model.scale_factor, enc_fea_lq)

    # 8. Color fix (canvas script lines 401-404)
    if colorfix_type == 'adain':
        x_samples = adaptive_instance_normalization(x_samples, init_image)
    elif colorfix_type == 'wavelet':
        x_samples = wavelet_reconstruction(x_samples, init_image)

    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    # 9. Crop back to exact target output size (remove all padding)
    x_samples = x_samples[:, :, :target_h, :target_w]

    # 10. Convert to numpy uint8
    x_sample = 255.0 * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
    return x_sample.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="StableSR Inference Wrapper for Evaluation"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="configs/stableSRNew/v2-finetune_text_T_768v.yaml")
    parser.add_argument("--ckpt", type=str,
                        default="weights/stablesr_768v_000139.ckpt")
    parser.add_argument("--vqgan_ckpt", type=str,
                        default="weights/vqgan_cfw_00011.ckpt")
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--dec_w", type=float, default=0.0)
    parser.add_argument("--upscale", type=float, default=4.0)
    parser.add_argument("--scale", type=float, default=7.0,
                        help="Unconditional guidance scale")
    parser.add_argument("--colorfix_type", type=str, default="wavelet",
                        choices=["adain", "wavelet", "nofix"])
    parser.add_argument("--input_size", type=int, default=768)
    parser.add_argument("--tile_overlap", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=0,
                        help="Max images to process. 0 = all.")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # Load models (canvas script lines 293-308)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    model = model.to(device)

    vqgan_config = OmegaConf.load(
        "configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml"
    )
    vq_model = load_model_from_config(vqgan_config, args.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = args.dec_w

    sampler = DDIMSampler(model)
    sampler.configs = config

    # Register schedule (canvas script lines 337-339)
    # NOTE: register_schedule must be called before model.to(device) to ensure
    # all schedule tensors (betas, alphas, etc.) are created, then moved to GPU.
    model.register_schedule(
        given_betas=None, beta_schedule="linear", timesteps=1000,
        linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3,
    )
    model.num_timesteps = 1000
    model = model.to(device)

    # Build timesteps (canvas script lines 341-343)
    ddim_timesteps = sorted(space_timesteps(1000, [args.ddim_steps]))

    sampler.make_schedule(
        ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False
    )

    # Find input images
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    # Filter already-processed
    remaining = []
    for f in image_files:
        name_no_ext = os.path.splitext(f)[0]
        out_path = os.path.join(output_dir, f"{name_no_ext}.png")
        if os.path.exists(out_path):
            continue
        remaining.append(f)

    print(f"Found {len(image_files)} images, {len(image_files) - len(remaining)} already processed")

    if args.max_images > 0:
        remaining = remaining[:args.max_images]

    if not remaining:
        print("All images already processed.")
        return

    print(f"Processing {len(remaining)} images...")

    # Process images
    with torch.no_grad():
        with model.ema_scope():
            for img_name in tqdm(remaining, desc="StableSR"):
                try:
                    seed_everything(args.seed)
                    img_path = os.path.join(input_dir, img_name)
                    result = process_single_image(
                        img_path=img_path,
                        model=model,
                        vq_model=vq_model,
                        sampler=sampler,
                        ddim_timesteps=ddim_timesteps,
                        device=device,
                        upscale=args.upscale,
                        input_size=args.input_size,
                        tile_overlap=args.tile_overlap,
                        use_negative_prompt=True,
                        scale=args.scale,
                        colorfix_type=args.colorfix_type,
                    )
                    name_no_ext = os.path.splitext(img_name)[0]
                    out_path = os.path.join(output_dir, f"{name_no_ext}.png")
                    Image.fromarray(result).save(out_path)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
