#!/usr/bin/env python3
"""
Ablation study for ControlNet Feature Recycling.

Runs inference with the hybrid-003 pipeline at multiple recycle_start_step
values (plus a no-recycle baseline), saves restored images in per-run
subdirectories, computes per-image and aggregate metrics, and produces a
final cross-run comparison table.

Fixed inference parameters:
    denoise_steps    = 10
    ensemble_size    = 1
    recycle_interval = 1  (recycle every step after start)

Usage example:
    python script/hybrid_controlnet_restoration/test_recycle_ablation.py \
        --checkpoint output/hybrid_re_007_lowLR/train_marigold_hybrid_controlnet_restoration_003/checkpoint/latest \
        --clean_dir   comparison_realsr/RealSR/RealSR-Canon-2/clean \
        --degraded_dir comparison_realsr/RealSR/RealSR-Canon-2/degraded_1x \
        --output_dir  output/recycle_ablation \
        --recycle_steps 1 2 3 4 5 \
        --max_images 10 \
        --seed 2024
"""

import sys
import os
import importlib.util

# Force offline mode to avoid HF Hub 404 errors with transformers 5.x
# (list_repo_templates call fails on SD2 repo). All models are cached locally.
os.environ["HF_HUB_OFFLINE"] = "1"

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import logging
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import helpers by file path (filenames start with digits, not importable
# as regular Python modules).
# ---------------------------------------------------------------------------

def _import_from_file(module_name: str, file_path: str):
    """Import a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pipeline loader — reuse load_hybrid_003_pipeline from run.py
_run_mod = _import_from_file(
    "hybrid_run",
    os.path.join(PROJECT_ROOT, "script", "hybrid_controlnet_restoration", "run.py"),
)
load_hybrid_003_pipeline = _run_mod.load_hybrid_003_pipeline

# Metrics utilities — reuse from 04_calculate_metrics.py
_metrics_mod = _import_from_file(
    "calculate_metrics",
    os.path.join(PROJECT_ROOT, "script", "restoration", "eval", "04_calculate_metrics.py"),
)
MetricsCalculator = _metrics_mod.MetricsCalculator
find_matching_files = _metrics_mod.find_matching_files
calculate_statistics = _metrics_mod.calculate_statistics
write_summary_file = _metrics_mod.write_summary_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

# Fixed inference parameters
DENOISE_STEPS = 10
ENSEMBLE_SIZE = 1
RECYCLE_INTERVAL = 1


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    pipe,
    degraded_paths: List[str],
    output_restored_dir: str,
    recycle_start_step: Optional[int],
    seed: int,
    device: torch.device,
) -> None:
    """
    Run restoration inference on a list of degraded images and save results.

    Output naming follows run.py convention: {stem}_restored.png
    so that find_matching_files can match them against clean images.

    Args:
        pipe: Loaded hybrid-003 pipeline (already on device).
        degraded_paths: Sorted list of degraded image file paths.
        output_restored_dir: Directory to save restored PNGs.
        recycle_start_step: None or 0 = baseline (no recycling).
        seed: RNG seed for reproducibility.
        device: Torch device.
    """
    os.makedirs(output_restored_dir, exist_ok=True)

    label = "baseline" if (recycle_start_step is None or recycle_start_step == 0) else f"recycle_s{recycle_start_step}"
    logger.info(
        f"[{label}] Running inference: {len(degraded_paths)} images, "
        f"steps={DENOISE_STEPS}, ensemble={ENSEMBLE_SIZE}, "
        f"recycle_start_step={recycle_start_step}, recycle_interval={RECYCLE_INTERVAL}"
    )

    for rgb_path in tqdm(degraded_paths, desc=f"Inference ({label})", leave=True):
        rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
        restored_save_path = os.path.join(output_restored_dir, f"{rgb_name_base}_restored.png")

        # Skip if already exists (allows resuming)
        if os.path.exists(restored_save_path):
            continue

        input_image = Image.open(rgb_path).convert("RGB")

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        pipe_out = pipe(
            input_image,
            denoising_steps=DENOISE_STEPS,
            ensemble_size=ENSEMBLE_SIZE,
            processing_res=0,          # native resolution
            match_input_res=True,
            batch_size=0,              # auto
            show_progress_bar=False,
            generator=generator,
            guidance_scale=1.0,        # no CFG
            recycle_start_step=recycle_start_step,
            recycle_interval=RECYCLE_INTERVAL,
        )

        restored_img: Image.Image = pipe_out.restored_img
        restored_img.save(restored_save_path)

    logger.info(f"[{label}] Inference complete → {output_restored_dir}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRIC_COLUMNS = ["psnr", "ssim", "lpips", "delta_e", "arniqa", "brisque", "niqe", "maniqa", "musiq"]


def compute_metrics_for_run(
    calculator: MetricsCalculator,
    clean_dir: str,
    restored_dir: str,
    metrics_dir: str,
    run_name: str,
) -> Optional[pd.DataFrame]:
    """
    Compute per-image metrics for one run and save CSV + summary TXT.

    Returns the per-image DataFrame (or None if no matches found).
    """
    os.makedirs(metrics_dir, exist_ok=True)

    matches = find_matching_files(clean_dir, restored_dir)
    if not matches:
        logger.warning(f"[{run_name}] No matching image pairs found — skipping metrics.")
        return None

    logger.info(f"[{run_name}] Computing metrics for {len(matches)} image pairs …")

    results = []
    for clean_path, restored_path, filename in tqdm(matches, desc=f"Metrics ({run_name})", leave=True):
        try:
            clean_pil = Image.open(clean_path).convert("RGB")
            restored_pil = Image.open(restored_path).convert("RGB")
            metrics = calculator.calculate_all_metrics(clean_pil, restored_pil)
            metrics["filename"] = filename
            results.append(metrics)
        except Exception as e:
            logger.warning(f"[{run_name}] Error processing {filename}: {e}")

    if not results:
        logger.warning(f"[{run_name}] No metrics computed.")
        return None

    df = pd.DataFrame(results)
    cols = ["filename"] + [c for c in METRIC_COLUMNS if c in df.columns]
    df = df[cols]

    # Per-image CSV
    csv_path = os.path.join(metrics_dir, f"metrics_{run_name}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"[{run_name}] Per-image CSV → {csv_path}")

    # Summary TXT
    stats = calculate_statistics(df, METRIC_COLUMNS)
    summary_path = Path(metrics_dir) / f"summary_{run_name}.txt"
    write_summary_file(stats, run_name, summary_path, len(results))
    logger.info(f"[{run_name}] Summary → {summary_path}")

    return df


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    run_dataframes: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """
    Build a cross-run comparison table with mean ± std for every metric
    and save as CSV + human-readable TXT.
    """
    rows = []
    for run_name, df in run_dataframes.items():
        row = {"run": run_name}
        for metric in METRIC_COLUMNS:
            if metric in df.columns:
                vals = df[metric].dropna()
                if len(vals) > 0:
                    row[f"{metric}_mean"] = round(float(vals.mean()), 4)
                    row[f"{metric}_std"] = round(float(vals.std()), 4)
        rows.append(row)

    comp_df = pd.DataFrame(rows)

    # CSV
    csv_path = os.path.join(output_dir, "comparison_recycle_ablation.csv")
    comp_df.to_csv(csv_path, index=False)
    logger.info(f"Comparison CSV → {csv_path}")

    # Human-readable TXT
    txt_path = os.path.join(output_dir, "comparison_recycle_ablation.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("CONTROLNET FEATURE RECYCLING — ABLATION COMPARISON\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Inference: steps={DENOISE_STEPS}, ensemble={ENSEMBLE_SIZE}, interval={RECYCLE_INTERVAL}\n")
        f.write("=" * 90 + "\n\n")

        # Header
        header = f"{'Run':<20}"
        for m in METRIC_COLUMNS:
            header += f"  {m.upper():>16}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for _, row in comp_df.iterrows():
            line = f"{row['run']:<20}"
            for m in METRIC_COLUMNS:
                mean_key = f"{m}_mean"
                std_key = f"{m}_std"
                if mean_key in row and pd.notna(row[mean_key]):
                    line += f"  {row[mean_key]:>7.4f}±{row[std_key]:<7.4f}"
                else:
                    line += f"  {'N/A':>16}"
            f.write(line + "\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("↑ = higher is better: PSNR, SSIM, ARNIQA, MANIQA, MUSIQ\n")
        f.write("↓ = lower is better:  LPIPS, Delta-E, BRISQUE, NIQE\n")
        f.write("=" * 90 + "\n")

    logger.info(f"Comparison TXT → {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for ControlNet Feature Recycling"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to hybrid-003 checkpoint (directory with unet/ and controlnet/).",
    )
    parser.add_argument(
        "--clean_dir", type=str, required=True,
        help="Directory with clean / ground-truth images.",
    )
    parser.add_argument(
        "--degraded_dir", type=str, required=True,
        help="Directory with corresponding degraded images.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Root output directory. Sub-folders are created per run.",
    )
    parser.add_argument(
        "--recycle_steps", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="List of recycle_start_step values to test. Default: 1 2 3 4 5",
    )
    parser.add_argument(
        "--max_images", type=int, default=10,
        help="Maximum number of images to process. 0 = all. Default: 10.",
    )
    parser.add_argument(
        "--seed", type=int, default=2024,
        help="RNG seed for reproducibility. Default: 2024.",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Load model in float16 (saves VRAM).",
    )
    parser.add_argument(
        "--base_model", type=str, default="stabilityai/stable-diffusion-2",
        help="Base SD2 model for frozen components. Default: stabilityai/stable-diffusion-2",
    )
    parser.add_argument(
        "--scheduler", type=str, choices=["ddim", "lcm"], default="ddim",
        help="Scheduler type. Default: ddim.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- Collect degraded images ----
    degraded_paths = sorted(
        f for f in glob(os.path.join(args.degraded_dir, "*"))
        if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    )
    if not degraded_paths:
        logger.error(f"No images found in {args.degraded_dir}")
        sys.exit(1)

    if args.max_images > 0 and len(degraded_paths) > args.max_images:
        degraded_paths = degraded_paths[: args.max_images]
        logger.info(f"Limited to {args.max_images} images")

    logger.info(f"Will process {len(degraded_paths)} images")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA not available — running on CPU will be very slow.")
    logger.info(f"Device: {device}")

    # ---- Load pipeline once ----
    dtype = torch.float16 if args.fp16 else torch.float32
    logger.info(f"Loading hybrid-003 pipeline from {args.checkpoint} (dtype={dtype})")
    pipe = load_hybrid_003_pipeline(
        checkpoint_path=args.checkpoint,
        sd2_model=args.base_model,
        dtype=dtype,
        scheduler_type=args.scheduler,
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass
    pipe = pipe.to(device)
    logger.info("Pipeline ready")

    # ---- Build list of runs: baseline + each recycle_start_step ----
    runs: List[Tuple[str, Optional[int]]] = [("baseline", None)]
    for s in sorted(set(args.recycle_steps)):
        if s > 0:
            runs.append((f"recycle_s{s}", s))

    logger.info(f"Runs planned: {[name for name, _ in runs]}")

    # ---- Phase 1: Inference ----
    logger.info("=" * 60)
    logger.info("PHASE 1 — INFERENCE")
    logger.info("=" * 60)

    run_restored_dirs: Dict[str, str] = {}
    for run_name, recycle_start in runs:
        restored_dir = os.path.join(output_dir, run_name, "restored")
        run_restored_dirs[run_name] = restored_dir

        with torch.no_grad():
            run_inference(
                pipe=pipe,
                degraded_paths=degraded_paths,
                output_restored_dir=restored_dir,
                recycle_start_step=recycle_start,
                seed=args.seed,
                device=device,
            )

    # ---- Free pipeline VRAM before metrics ----
    del pipe
    torch.cuda.empty_cache()
    logger.info("Pipeline unloaded, VRAM freed for metrics computation")

    # ---- Phase 2: Metrics ----
    logger.info("=" * 60)
    logger.info("PHASE 2 — METRICS")
    logger.info("=" * 60)

    calculator = MetricsCalculator(device="cuda" if torch.cuda.is_available() else "cpu")

    run_dataframes: Dict[str, pd.DataFrame] = {}
    for run_name, _ in runs:
        restored_dir = run_restored_dirs[run_name]
        metrics_dir = os.path.join(output_dir, run_name, "metrics")

        df = compute_metrics_for_run(
            calculator=calculator,
            clean_dir=args.clean_dir,
            restored_dir=restored_dir,
            metrics_dir=metrics_dir,
            run_name=run_name,
        )
        if df is not None:
            run_dataframes[run_name] = df

    # ---- Phase 3: Comparison ----
    if len(run_dataframes) >= 2:
        logger.info("=" * 60)
        logger.info("PHASE 3 — COMPARISON")
        logger.info("=" * 60)
        build_comparison_table(run_dataframes, output_dir)
    else:
        logger.warning("Not enough runs with results to build comparison table.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
