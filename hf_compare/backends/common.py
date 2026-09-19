# Shared helpers for the local (non-diffusion) backends:
# tiled inference with overlap averaging + robust .pth weight loading.

import torch
import torch.nn.functional as F

# Official checkpoints may store the state dict under different keys.
_STATE_DICT_KEYS = ("params_ema", "params", "state_dict", "model", "net")


def load_pth_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a .pth checkpoint into `model`, trying common wrapper keys.

    Raises RuntimeError if no compatible state dict is found.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        for key in _STATE_DICT_KEYS:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unrecognized checkpoint format in {ckpt_path}")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        # Fall back to strict to surface the real mismatch
        model.load_state_dict(ckpt, strict=True)
    model.eval()


@torch.no_grad()
def tiled_forward(
    model: torch.nn.Module,
    img: torch.Tensor,
    tile: int = 256,
    overlap: int = 32,
    scale: int = 1,
) -> torch.Tensor:
    """Run `model` on tiles of `img` ([1, 3, H, W] in [0, 1]) and blend overlaps.

    `scale` is the model output scale (1 for Restormer, 4 for Real-ESRGAN x4).
    Returns [1, 3, H*scale, W*scale] in [0, 1].
    """
    _, _, h, w = img.shape
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if h <= tile and w <= tile:
        out = model(img.to(device=device, dtype=dtype)).clamp(0, 1)
        return out.detach().cpu().float()

    stride = tile - overlap
    out_acc = torch.zeros(1, 3, h * scale, w * scale, dtype=torch.float32)
    w_acc = torch.zeros(1, 1, h * scale, w * scale, dtype=torch.float32)

    # 1D triangular blending weights
    ramp = torch.linspace(0, 1, overlap + 2)[1:-1]

    ys = list(range(0, h - tile + 1, stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    xs = list(range(0, w - tile + 1, stride))
    if xs[-1] + tile < w:
        xs.append(w - tile)

    for y in ys:
        wy = torch.ones(tile * scale, dtype=torch.float32)
        if y > 0:
            wy[: overlap * scale] = ramp.repeat_interleave(scale)
        if y + tile < h:
            wy[-overlap * scale :] = ramp.flip(0).repeat_interleave(scale)
        for x in xs:
            wx = torch.ones(tile * scale, dtype=torch.float32)
            if x > 0:
                wx[: overlap * scale] = ramp.repeat_interleave(scale)
            if x + tile < w:
                wx[-overlap * scale :] = ramp.flip(0).repeat_interleave(scale)
            weight = (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)

            patch = img[:, :, y : y + tile, x : x + tile]
            pred = model(patch.to(device=device, dtype=dtype)).clamp(0, 1).detach().cpu().float()
            out_acc[:, :, y * scale : (y + tile) * scale, x * scale : (x + tile) * scale] += (
                pred * weight
            )
            w_acc[:, :, y * scale : (y + tile) * scale, x * scale : (x + tile) * scale] += weight

    return (out_acc / w_acc.clamp_min(1e-8)).clamp(0, 1)
