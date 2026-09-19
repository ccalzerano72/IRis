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
from metrics import LightMetrics, style_best

logging.basicConfig(level=logging.INFO)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_HERE, "examples")

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
    return style_best(df), csv_path


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

### About IRis
**IRis** (Image Restoration via latent diffusion, [thesis project — University of Pisa](https://github.com/ccalzerano72/IRis))
repurposes Stable Diffusion 2 for blind image restoration: an 8-channel UNet
conditioned on the degraded image in latent space plus a ControlNet branch,
trained jointly on synthetic degradations. It ranks 1st on all metrics
(PSNR/SSIM/ΔE/LPIPS) on both synthetic (DIV2K) and real-world (RealSR)
degradations against Real-ESRGAN, Restormer, DiffBIR and HyPIR.

- 🖼️ [Single-image IRis demo](https://huggingface.co/spaces/ccalzerano72/IRis)
- 💻 [GitHub repository](https://github.com/ccalzerano72/IRis)
- ⚖️ [Model weights](https://huggingface.co/ccalzerano72/IRis-hybrid-003)
        """
    )

    with gr.Tabs() as tabs:
        with gr.Tab("1 · Degrade", id="degrade"):
            with gr.Row():
                with gr.Column():
                    clean_in = gr.Image(type="numpy", label="Clean image", sources=["upload", "clipboard"])
                    gr.Examples(
                        examples=[
                            [os.path.join(_EXAMPLES_DIR, "clean_urban100_011.jpg")],
                            [os.path.join(_EXAMPLES_DIR, "clean_urban100_062.jpg")],
                            [os.path.join(_EXAMPLES_DIR, "clean_div2k_0801.jpg")],
                        ],
                        inputs=[clean_in],
                        label="Try an example (Urban100 / DIV2K)",
                    )
                    mode = gr.Radio(
                        ["Default (presets)", "Advanced (thesis pipeline)"],
                        value="Default (presets)",
                        label="Mode",
                    )
                    preset = gr.Dropdown(list(PRESETS.keys()), value="Medium", label="Preset")
                    with gr.Column(visible=False) as basic_row:
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
                    to_compare_btn = gr.Button("Send to Compare →", interactive=False)

            preset.change(fn=preset_to_sliders, inputs=[preset], outputs=[blur, noise, jpeg, scale])

            def _toggle_mode(m):
                adv = m == "Advanced (thesis pipeline)"
                return (
                    gr.update(visible=not adv),  # preset
                    gr.update(visible=adv),  # basic sliders
                    gr.update(visible=adv),  # stage-2 row
                )

            mode.change(
                fn=_toggle_mode,
                inputs=[mode],
                outputs=[preset, basic_row, adv_row],
            )
            degrade_btn.click(
                fn=degrade_fn,
                inputs=[clean_in, mode, preset, blur, noise, jpeg, scale, use_stage2, noise2, jpeg2, seed_deg],
                outputs=[degraded_out, params_text],
            )
            degraded_out.change(
                fn=lambda x: gr.update(interactive=x is not None),
                inputs=[degraded_out],
                outputs=[to_compare_btn],
            )

        with gr.Tab("2 · Compare", id="compare"):
            with gr.Row():
                compare_in = gr.Image(type="numpy", label="Degraded image", sources=["upload", "clipboard"])
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### IRis (diffusion, GPU)")
                    iris_btn = gr.Button("Restore with IRis", variant="primary")
                    iris_out = gr.Image(type="numpy", label="IRis output")
                    iris_steps = gr.Slider(1, 50, value=5, step=1, label="Denoising steps")
                    iris_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                with gr.Column():
                    gr.Markdown("### Restormer (GPU)")
                    restormer_btn = gr.Button("Restore with Restormer", variant="primary")
                    restormer_out = gr.Image(type="numpy", label="Restormer output")
                with gr.Column():
                    gr.Markdown("### Real-ESRGAN x4plus (CPU)")
                    realesrgan_btn = gr.Button("Restore with Real-ESRGAN", variant="primary")
                    realesrgan_out = gr.Image(type="numpy", label="Real-ESRGAN output")
            to_metrics_btn = gr.Button("Send all to Metrics →", interactive=False)

            def _toggle_metrics_btn(i, r, e):
                return gr.update(interactive=any(v is not None for v in (i, r, e)))

            to_compare_btn.click(
                fn=lambda x: (x, gr.Tabs(selected="compare")),
                inputs=[degraded_out],
                outputs=[compare_in, tabs],
            )
            iris_btn.click(fn=restore_iris, inputs=[compare_in, iris_steps, iris_seed], outputs=[iris_out])
            restormer_btn.click(fn=restore_restormer, inputs=[compare_in], outputs=[restormer_out])
            realesrgan_btn.click(fn=restore_realesrgan, inputs=[compare_in], outputs=[realesrgan_out])
            for _out in (iris_out, restormer_out, realesrgan_out):
                _out.change(
                    fn=_toggle_metrics_btn,
                    inputs=[iris_out, restormer_out, realesrgan_out],
                    outputs=[to_metrics_btn],
                )

        with gr.Tab("3 · Metrics", id="metrics"):
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
                fn=lambda c, d, i, r, e: (c, d, i, r, e, gr.Tabs(selected="metrics")),
                inputs=[clean_in, degraded_out, iris_out, restormer_out, realesrgan_out],
                outputs=[m_clean, m_degraded, m_iris, m_restormer, m_realesrgan, tabs],
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
- About IRis: [single-image demo](https://huggingface.co/spaces/ccalzerano72/IRis) ·
  [GitHub repository](https://github.com/ccalzerano72/IRis) ·
  [model weights](https://huggingface.co/ccalzerano72/IRis-hybrid-003)
- Test data: example images from [Urban100](https://github.com/jbhuang0604/SelfExSR)
  ([HF mirror](https://huggingface.co/datasets/eugenesiow/Urban100), CC-BY-4.0) and
  [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
  ([HF mirror](https://huggingface.co/datasets/eugenesiow/Div2k)).
        """
    )

demo.queue().launch()