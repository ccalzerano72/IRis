# Restormer backend (Real_Denoising task, as in the thesis eval).
# Architecture vendored from swz30/Restormer (MIT); weights: deepinv/Restormer real_denoising.pth.
# Single forward pass, same-resolution in/out.

import logging

import numpy as np
import torch
from PIL import Image

from .common import load_pth_weights, tiled_forward
from .restormer_arch import Restormer

WEIGHT_FILE = "real_denoising.pth"
TILE = 256
OVERLAP = 32


def load_restormer(ckpt_path: str, device: torch.device):
    logging.info(f"Loading Restormer from {ckpt_path} ...")
    # Denoising checkpoints are trained with BiasFree LayerNorm (no norm biases
    # in the state dict); deblurring ones use WithBias. Try both.
    last_error = None
    for norm_type in ("WithBias", "BiasFree"):
        try:
            model = Restormer(LayerNorm_type=norm_type)
            load_pth_weights(model, ckpt_path)
            logging.info(f"Restormer loaded (LayerNorm_type={norm_type}).")
            return model.to(device)
        except RuntimeError as e:
            last_error = e
            logging.info(f"LayerNorm_type={norm_type} did not match, retrying...")
    raise RuntimeError(f"Could not load Restormer weights: {last_error}")


def run_restormer(model, image: Image.Image) -> Image.Image:
    """Run Restormer Real_Denoising. Returns PIL RGB at input resolution."""
    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W] in [0,1]
    out = tiled_forward(model, ten, tile=TILE, overlap=OVERLAP, scale=1)
    out_np = (out.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(out_np)
