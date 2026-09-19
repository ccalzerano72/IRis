# Real-ESRGAN backend (RealESRGAN_x4plus, outscale=1 as in the thesis eval).
# Minimal RRDBNet with BasicSR-compatible keys; official weights from GitHub releases.
# The x4 output is downscaled back to the input resolution (LANCZOS),
# mimicking `inference_realesrgan.py --outscale 1`.

import logging

import numpy as np
import torch
from PIL import Image

from .common import load_pth_weights, tiled_forward
from .rrdbnet import RRDBNet

TILE = 256
OVERLAP = 32
SCALE = 4


def load_realesrgan(ckpt_path: str, device: torch.device):
    logging.info(f"Loading Real-ESRGAN x4plus from {ckpt_path} ...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=SCALE, num_feat=64, num_block=23)
    load_pth_weights(model, ckpt_path)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Real-ESRGAN x4plus loaded ({n_params:.1f}M params).")
    return model


def run_realesrgan(model, image: Image.Image) -> Image.Image:
    """Run Real-ESRGAN x4plus with outscale=1. Returns PIL RGB at input resolution."""
    img = image.convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    out = tiled_forward(model, ten, tile=TILE, overlap=OVERLAP, scale=SCALE)
    out_np = (out.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    out_img = Image.fromarray(out_np)
    if out_img.size != (w, h):
        out_img = out_img.resize((w, h), Image.LANCZOS)
    return out_img
