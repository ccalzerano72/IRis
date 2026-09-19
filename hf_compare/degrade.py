# Synthetic degradation module for the IRis-compare Space.
#
# Default mode: presets (Mild/Medium/Strong) + 4 basic controls.
# Advanced mode: sliders replicating the two-stage RealESRGAN-style pipeline
# used to train IRis (blur -> resize -> noise -> JPEG, optionally twice).
# Works on CPU with PIL + numpy only.

import io

import numpy as np
from PIL import Image, ImageFilter

PRESETS = {
    "Mild": {"blur": 1.0, "noise": 8.0, "jpeg": 80, "scale": 1.0,
             "stage2": False, "noise2": 0.0, "jpeg2": 100},
    "Medium": {"blur": 2.0, "noise": 15.0, "jpeg": 60, "scale": 2.0,
               "stage2": False, "noise2": 0.0, "jpeg2": 100},
    "Strong": {"blur": 3.0, "noise": 25.0, "jpeg": 40, "scale": 2.0,
               "stage2": True, "noise2": 10.0, "jpeg2": 60},
}

PARAM_RANGES = {
    "blur": (0.0, 5.0, 0.1),
    "noise": (0.0, 50.0, 1.0),
    "jpeg": (10, 100, 1),
    "scale": (1.0, 4.0, 0.5),
    "noise2": (0.0, 50.0, 1.0),
    "jpeg2": (10, 100, 1),
}


def _gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma is None or sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _resize_degrade(img: Image.Image, scale: float) -> Image.Image:
    if scale is None or scale <= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, int(w / scale)), max(1, int(h / scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def _gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    if sigma is None or sigma <= 0:
        return img
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0.0, float(sigma), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    if quality is None or quality >= 100:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_degradation(image: Image.Image, params: dict, seed: int = 0) -> Image.Image:
    """Apply blur -> resize -> noise -> JPEG (+ optional second stage).

    `params` keys: blur, noise, jpeg, scale, stage2 (bool), noise2, jpeg2.
    Output has the same size as the input.
    """
    rng = np.random.default_rng(seed)
    img = image.convert("RGB")

    # Stage 1
    img = _gaussian_blur(img, params.get("blur", 0.0))
    img = _resize_degrade(img, params.get("scale", 1.0))
    img = _gaussian_noise(img, params.get("noise", 0.0), rng)
    img = _jpeg(img, params.get("jpeg", 100))

    # Stage 2 (optional, lighter)
    if params.get("stage2", False):
        img = _gaussian_noise(img, params.get("noise2", 0.0), rng)
        img = _jpeg(img, params.get("jpeg2", 100))

    return img


def describe_params(params: dict) -> str:
    parts = [
        f"blur σ={params.get('blur', 0)}",
        f"noise σ={params.get('noise', 0)}",
        f"JPEG q={params.get('jpeg', 100)}",
        f"scale ×{params.get('scale', 1)}",
    ]
    if params.get("stage2", False):
        parts.append(f"stage2 (noise σ={params.get('noise2', 0)}, JPEG q={params.get('jpeg2', 100)})")
    return ", ".join(parts)
