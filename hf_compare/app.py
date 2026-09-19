# IRis-compare: degrade a clean image, restore it with IRis / Restormer /
# Real-ESRGAN, and compare full-reference + no-reference metrics.
# HuggingFace Spaces (ZeroGPU) app. All three backends run locally:
# IRis + Restormer on GPU workers, Real-ESRGAN on CPU.

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # must be imported before torch

import logging
import tempfile
import traceback

import torch
from PIL import Image

from backends import iris_backend, realesrgan_backend, restormer_backend
from degrade import PRESETS, apply_degradation, describe_params
from metrics import LightMetrics

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Module-scope model loading (ZeroGPU "pack" pattern).
# ---------------------------------------------------------------------------
print("Loading IRis pipeline...")
iris_pipe = iris_backend.load_iris(dtype=torch.float16)
iris_pipe = iris_pipe.to("cuda")
print("IRis ready.")

print("Loading Restormer (Real_Denoising)...")
restormer_model = None
try:
    from huggingface_hub import snapshot_download

    restormer_dir = snapshot_download(
        repo_id="deepinv/Restormer", allow_patterns=["real_denoising.pth"]
    )
    restormer_model = restormer_backend.load_restormer(
        os.path.join(restormer_dir, "real_denoising.pth"), torch.device("cuda")
    )
    restormer_model = restormer_model.to("cuda")
    print("Restormer ready.")
except Exception:
    traceback.print_exc()
    print("Restormer unavailable.")

print("Loading Real-ESRGAN x4plus (CPU)...")
realesrgan_model = None
try:
    _here = os.path.dirname(os.path.abspath(__file__))
    realesrgan_model = realesrgan_backend.load_realesrgan(
        os.path.join(_here, "weights", "RealESRGAN_x4plus.pth"), torch.device("cpu")
    )
    print("Real-ESRGAN ready.")
except Exception:
    traceback.print_exc()
    print("Real-ESRGAN unavailable.")

print("Initializing metrics (CPU)...")
metrics = LightMetrics(device="cpu")
print("Metrics ready.")


# ---------------------------------------------------------------------------
# Tab 1: degradation
# ---------------------------------------------------------------------------
def degrade_fn(clean, mode, preset, blur, noise, jpeg, scale, use_stage2, noise2, jpeg2, seed):
    if clean is None:
        raise ValueError("Upload a clean image first.")
    if preset in PRESETS and mode == "Default (presets)":
        params = dict(PRESETS[preset])
    else:
        params = {
            "blur": float(blur),
            "noise": float(noise),
            "jpeg": int(jpeg),
            "scale": float(scale),
            "stage2": bool(use_stage2),
            "noise2": float(noise2),
            "jpeg2": int(jpeg2),
        }
    degraded = apply_degradation(Image.fromarray(clean).convert("RGB"), params, seed=int(seed))
    return degraded, describe_params(params)


def preset_to_sliders(preset):
    p = PRESETS.get(preset, PRESETS["Medium"])
    return p["blur"], p["noise"], p["jpeg"], p["scale"]


# ---------------------------------------------------------------------------
# Tab 2: restoration backends
# ---------------------------------------------------------------------------
def _estimate_iris_duration(image, steps, seed, *args, **kwargs):
    return min(180, 30 + int(steps) * 15)


@spaces.GPU(duration=_estimate_iris_duration)
def restore_iris(image, steps, seed):
    if image is None:
        raise ValueError("Provide a degraded image first.")
    try:
        return iris_backend.run_iris(
            iris_pipe, Image.fromarray(image).convert("RGB"), steps=int(steps), seed=int(seed)
        )
    except Exception:
        traceback.print_exc()
        raise


@spaces.GPU(duration=120)
def restore_restormer(image):
    if restormer_model is None:
        raise RuntimeError("Restormer backend unavailable.")
    if image is None:
        raise ValueError("Provide a degraded image first.")
    try:
        return restormer_backend.run_restormer(
            restormer_model, Image.fromarray(image).convert("RGB")
        )
    except Exception:
        traceback.print_exc()
        raise


def restore_realesrgan(image):
    if realesrgan_model is None:
        raise RuntimeError("Real-ESRGAN backend unavailable.")
    if image is None:
        raise ValueError("Provide a degraded image first.")
    try:
        return realesrgan_backend.run_realesrgan(
            realesrgan_model, Image.fromarray(image).convert("RGB")
        )
    except Exception:
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Tab 3: metrics
# ---------------------------------------------------------------------------
def compute_metrics(clean, degraded, out_iris, out_restormer, out_realesrgan):
    import pandas as pd

    if clean is None:
        raise ValueError("Provide the clean reference image.")
    clean_img = Image.fromarray(clean).convert("RGB")

    rows = []
    candidates = [
        ("Degraded", degraded),
        ("IRis", out_iris),
        ("Restormer", out_restormer),
        ("Real-ESRGAN", out_realesrgan),
    ]
    for name, arr in candidates:
        if arr is None:
            continue
        img = Image.fromarray(arr).convert("RGB")
        row = {"image": name}
        row.update(metrics.full_reference(clean_img, img))
        row.update(metrics.no_reference(img))
        rows.append(row)

    if not rows:
        raise ValueError("No images to evaluate.")

    df = pd.DataFrame(rows, columns=["image", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "NIQE ↓", "MUSIQ ↑"])
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    df.to_csv(csv_path, index=False)
    return df, csv_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown(
        """
# IRis-compare — Restore & Compare

Upload a **clean** image, degrade it, restore it with three engines
(**IRis**, **Restormer**, **Real-ESRGAN**) and compare full-reference
(PSNR/SSIM/LPIPS) and no-reference (NIQE/MUSIQ) metrics.
Engines run locally in this Space: IRis and Restormer on GPU, Real-ESRGAN on CPU.
        """
    )

    with gr.Tab("1 · Degrade"):
        with gr.Row():
            with gr.Column():
                clean_in = gr.Image(type="numpy", label="Clean image", sources=["upload", "clipboard"])
                mode = gr.Radio(
                    ["Default (presets)", "Advanced (thesis pipeline)"],
                    value="Default (presets)",
                    label="Mode",
                )
                preset = gr.Dropdown(list(PRESETS.keys()), value="Medium", label="Preset")
                blur = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Blur σ")
                noise = gr.Slider(0.0, 50.0, value=15.0, step=1.0, label="Noise σ")
                jpeg = gr.Slider(10, 100, value=60, step=1, label="JPEG quality")
                scale = gr.Slider(1.0, 4.0, value=2.0, step=0.5, label="Downscale ×")
                with gr.Row(visible=False) as adv_row:
                    use_stage2 = gr.Checkbox(value=False, label="Second stage")
                    noise2 = gr.Slider(0.0, 50.0, value=10.0, step=1.0, label="Stage-2 noise σ")
                    jpeg2 = gr.Slider(10, 100, value=60, step=1, label="Stage-2 JPEG quality")
                seed_deg = gr.Number(value=0, label="Degradation seed", precision=0)
                degrade_btn = gr.Button("Apply degradation", variant="primary")
            with gr.Column():
                degraded_out = gr.Image(type="pil", label="Degraded image")
                params_text = gr.Textbox(label="Degradation parameters")
                to_compare_btn = gr.Button("Send to Compare →")

        preset.change(fn=preset_to_sliders, inputs=[preset], outputs=[blur, noise, jpeg, scale])
        mode.change(
            fn=lambda m: gr.update(visible=(m == "Advanced (thesis pipeline)")),
            inputs=[mode],
            outputs=[adv_row],
        )
        degrade_btn.click(
            fn=degrade_fn,
            inputs=[clean_in, mode, preset, blur, noise, jpeg, scale, use_stage2, noise2, jpeg2, seed_deg],
            outputs=[degraded_out, params_text],
        )

    with gr.Tab("2 · Compare"):
        with gr.Row():
            compare_in = gr.Image(type="numpy", label="Degraded image", sources=["upload", "clipboard"])
        with gr.Row():
            with gr.Column():
                gr.Markdown("### IRis (diffusion, GPU)")
                iris_steps = gr.Slider(1, 50, value=5, step=1, label="Denoising steps")
                iris_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                iris_btn = gr.Button("Restore with IRis", variant="primary")
                iris_out = gr.Image(type="numpy", label="IRis output")
            with gr.Column():
                gr.Markdown("### Restormer (GPU)")
                restormer_btn = gr.Button("Restore with Restormer", variant="primary")
                restormer_out = gr.Image(type="numpy", label="Restormer output")
            with gr.Column():
                gr.Markdown("### Real-ESRGAN x4plus (CPU)")
                realesrgan_btn = gr.Button("Restore with Real-ESRGAN", variant="primary")
                realesrgan_out = gr.Image(type="numpy", label="Real-ESRGAN output")
        to_metrics_btn = gr.Button("Send all to Metrics →")

        to_compare_btn.click(fn=lambda x: x, inputs=[degraded_out], outputs=[compare_in])
        iris_btn.click(fn=restore_iris, inputs=[compare_in, iris_steps, iris_seed], outputs=[iris_out])
        restormer_btn.click(fn=restore_restormer, inputs=[compare_in], outputs=[restormer_out])
        realesrgan_btn.click(fn=restore_realesrgan, inputs=[compare_in], outputs=[realesrgan_out])

    with gr.Tab("3 · Metrics"):
        with gr.Row():
            m_clean = gr.Image(type="numpy", label="Clean reference")
            m_degraded = gr.Image(type="numpy", label="Degraded")
        with gr.Row():
            m_iris = gr.Image(type="numpy", label="IRis")
            m_restormer = gr.Image(type="numpy", label="Restormer")
            m_realesrgan = gr.Image(type="numpy", label="Real-ESRGAN")
        metrics_btn = gr.Button("Compute metrics", variant="primary")
        metrics_table = gr.Dataframe(label="Results (None = metric unavailable)")
        metrics_csv = gr.File(label="Download CSV")

        to_metrics_btn.click(
            fn=lambda c, d, i, r, e: (c, d, i, r, e),
            inputs=[clean_in, degraded_out, iris_out, restormer_out, realesrgan_out],
            outputs=[m_clean, m_degraded, m_iris, m_restormer, m_realesrgan],
        )
        metrics_btn.click(
            fn=compute_metrics,
            inputs=[m_clean, m_degraded, m_iris, m_restormer, m_realesrgan],
            outputs=[metrics_table, metrics_csv],
        )

    gr.Markdown(
        """
- Runs on shared ZeroGPU hardware: the first request after a sleep is slower
  while models are streamed to the GPU. Free daily GPU quota applies per user.
- Restormer: Real_Denoising task (as in the thesis); Real-ESRGAN: x4plus with
  outscale 1 (same-resolution restoration).
        """
    )

demo.queue().launch()