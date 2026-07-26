#!/usr/bin/env python3
"""
Baseline Metrics Calculation for Restoration Model Evaluation.

Computes two sets of baseline metrics for a given dataset results folder:

1. baseline_degraded: Treats degraded images as "restored" output.
   - Full-reference metrics: degraded vs clean (the quality floor)
   - No-reference metrics: computed on degraded images

2. baseline_clean: Metrics on clean images themselves.
   - Full-reference metrics: set to theoretical perfect values
     (PSNR=100, SSIM=1.0, LPIPS=0.0, Delta-E=0.0)
   - No-reference metrics: computed on clean images (the quality ceiling)

These baselines provide reference points for interpreting model comparison results.

Usage:
    # Single dataset folder
    python script/restoration/eval/05_calculate_baselines.py \
        --results_dir comparison/results/BSDS100

    # Multiple dataset folders
    python script/restoration/eval/05_calculate_baselines.py \
        --results_dir comparison/results/BSDS100 \
        --results_dir comparison/results/Urban100

    # Specify device
    python script/restoration/eval/05_calculate_baselines.py \
        --results_dir comparison/results/BSDS100 \
        --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import from 04_calculate_metrics.py (numeric prefix prevents normal import)
# ---------------------------------------------------------------------------
_METRICS_SCRIPT = Path(__file__).parent / "04_calculate_metrics.py"
_spec = importlib.util.spec_from_file_location("calculate_metrics_04", _METRICS_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MetricsCalculator = _mod.MetricsCalculator
calculate_statistics = _mod.calculate_statistics
write_summary_file = _mod.write_summary_file
SUPPORTED_EXTENSIONS = _mod.SUPPORTED_EXTENSIONS


# Theoretical perfect values for full-reference metrics (clean vs clean)
PERFECT_FR_METRICS = {
    'psnr': 100.0,     # Finite cap so charts render properly (real inf is problematic)
    'ssim': 1.0,       # Identical images
    'lpips': 0.0,      # Identical images
    'delta_e': 0.0,    # Identical images
}

METRIC_COLUMNS = ['psnr', 'ssim', 'lpips', 'delta_e', 'arniqa', 'brisque', 'niqe', 'maniqa', 'musiq']


def find_images(directory: Path) -> List[Path]:
    """
    Find all supported image files in a directory.
    Returns sorted list of image paths.
    """
    images = []
    for f in directory.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(f)
    return sorted(images, key=lambda p: p.stem)


def calculate_baseline_degraded(
    clean_dir: Path,
    degraded_dir: Path,
    metrics_dir: Path,
    calculator: MetricsCalculator,
) -> bool:
    """
    Calculate baseline metrics treating degraded images as "restored" output.

    Full-reference metrics compare degraded vs clean.
    No-reference metrics are computed on degraded images.

    Uses the same calculate_all_metrics() method from MetricsCalculator
    (verified in 04_calculate_metrics.py lines 390-420).

    Returns True if successful.
    """
    model_name = "baseline_degraded"
    csv_path = metrics_dir / f"metrics_{model_name}.csv"
    summary_path = metrics_dir / f"summary_{model_name}.txt"

    # Check if already computed
    if csv_path.exists() and summary_path.exists():
        logger.info(f"  Baseline degraded already exists, skipping: {csv_path}")
        return True

    # Find clean images
    clean_images = find_images(clean_dir)
    if not clean_images:
        logger.error(f"  No images found in clean dir: {clean_dir}")
        return False

    # Match clean images to degraded images
    results = []
    for clean_path in tqdm(clean_images, desc="  baseline_degraded"):
        # Degraded images use same filename as clean
        degraded_path = degraded_dir / clean_path.name
        if not degraded_path.exists():
            logger.warning(f"  No matching degraded image for {clean_path.name}, skipping")
            continue

        try:
            clean_pil = Image.open(clean_path).convert('RGB')
            degraded_pil = Image.open(degraded_path).convert('RGB')

            # calculate_all_metrics(clean_pil, restored_pil) — here "restored" = degraded
            metrics = calculator.calculate_all_metrics(clean_pil, degraded_pil)
            metrics['filename'] = clean_path.stem
            metrics['clean_path'] = str(clean_path)
            metrics['restored_path'] = str(degraded_path)

            results.append(metrics)

        except Exception as e:
            logger.warning(f"  Error processing {clean_path.name}: {str(e)}")
            continue

    if not results:
        logger.error("  No results generated for baseline_degraded")
        return False

    # Create DataFrame — same column order as 04_calculate_metrics.py main() lines 640-643
    df = pd.DataFrame(results)
    cols = ['filename'] + METRIC_COLUMNS + ['clean_path', 'restored_path']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Save CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path} ({len(results)} images)")

    # Calculate statistics and write summary
    stats = calculate_statistics(df, METRIC_COLUMNS)
    write_summary_file(stats, model_name, summary_path, len(results))
    logger.info(f"  Saved: {summary_path}")

    return True


def calculate_baseline_clean(
    clean_dir: Path,
    metrics_dir: Path,
    calculator: MetricsCalculator,
) -> bool:
    """
    Calculate baseline metrics on clean images.

    Full-reference metrics are set to theoretical perfect values.
    No-reference metrics are computed on clean images.

    Returns True if successful.
    """
    model_name = "baseline_clean"
    csv_path = metrics_dir / f"metrics_{model_name}.csv"
    summary_path = metrics_dir / f"summary_{model_name}.txt"

    # Check if already computed
    if csv_path.exists() and summary_path.exists():
        logger.info(f"  Baseline clean already exists, skipping: {csv_path}")
        return True

    # Find clean images
    clean_images = find_images(clean_dir)
    if not clean_images:
        logger.error(f"  No images found in clean dir: {clean_dir}")
        return False

    results = []
    for clean_path in tqdm(clean_images, desc="  baseline_clean"):
        try:
            clean_pil = Image.open(clean_path).convert('RGB')

            # No-reference metrics on clean image
            # Using individual calculate_* methods verified in 04_calculate_metrics.py:
            #   calculate_arniqa(img_pil) — line 240
            #   calculate_brisque(img_pil) — line 260
            #   calculate_niqe(img_pil) — line 282
            #   calculate_maniqa(img_pil) — line 304
            #   calculate_musiq(img_pil) — line 326
            metrics = {
                # Perfect full-reference values
                'psnr': PERFECT_FR_METRICS['psnr'],
                'ssim': PERFECT_FR_METRICS['ssim'],
                'lpips': PERFECT_FR_METRICS['lpips'],
                'delta_e': PERFECT_FR_METRICS['delta_e'],
                # Real no-reference metrics on clean image
                'arniqa': calculator.calculate_arniqa(clean_pil),
                'brisque': calculator.calculate_brisque(clean_pil),
                'niqe': calculator.calculate_niqe(clean_pil),
                'maniqa': calculator.calculate_maniqa(clean_pil),
                'musiq': calculator.calculate_musiq(clean_pil),
            }
            metrics['filename'] = clean_path.stem
            metrics['clean_path'] = str(clean_path)
            metrics['restored_path'] = str(clean_path)  # "restored" = clean itself

            results.append(metrics)

        except Exception as e:
            logger.warning(f"  Error processing {clean_path.name}: {str(e)}")
            continue

    if not results:
        logger.error("  No results generated for baseline_clean")
        return False

    # Create DataFrame
    df = pd.DataFrame(results)
    cols = ['filename'] + METRIC_COLUMNS + ['clean_path', 'restored_path']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Save CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path} ({len(results)} images)")

    # Calculate statistics and write summary
    stats = calculate_statistics(df, METRIC_COLUMNS)
    write_summary_file(stats, model_name, summary_path, len(results))
    logger.info(f"  Saved: {summary_path}")

    return True


def process_results_folder(results_dir: Path, device: str) -> bool:
    """
    Process a single results folder, generating both baselines.

    Expects the folder to contain:
      - clean/    (symlink or directory with clean images)
      - degraded/ (symlink or directory with degraded images)
      - metrics/  (directory for output files)

    Returns True if both baselines were generated successfully.
    """
    clean_dir = results_dir / "clean"
    degraded_dir = results_dir / "degraded"
    metrics_dir = results_dir / "metrics"

    # Validate directories
    if not clean_dir.exists():
        logger.error(f"Clean directory not found: {clean_dir}")
        return False
    if not degraded_dir.exists():
        logger.error(f"Degraded directory not found: {degraded_dir}")
        return False

    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {results_dir.name}")
    logger.info(f"  Clean dir:    {clean_dir} -> {clean_dir.resolve()}")
    logger.info(f"  Degraded dir: {degraded_dir} -> {degraded_dir.resolve()}")
    logger.info(f"  Metrics dir:  {metrics_dir}")

    # Initialize calculator once for both baselines
    calculator = MetricsCalculator(device=device)

    # Calculate both baselines
    ok_degraded = calculate_baseline_degraded(clean_dir, degraded_dir, metrics_dir, calculator)
    ok_clean = calculate_baseline_clean(clean_dir, metrics_dir, calculator)

    return ok_degraded and ok_clean


def main():
    parser = argparse.ArgumentParser(
        description="Calculate baseline metrics (degraded and clean) for restoration evaluation"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        action="append",
        required=True,
        help="Results folder(s) containing clean/, degraded/, metrics/ "
             "(can be specified multiple times)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu). Default: cuda",
    )

    args = parser.parse_args()

    success_count = 0
    fail_count = 0

    for results_dir_str in args.results_dir:
        results_dir = Path(results_dir_str)
        if not results_dir.is_dir():
            logger.error(f"Not a directory: {results_dir}")
            fail_count += 1
            continue

        ok = process_results_folder(results_dir, args.device)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    logger.info("")
    logger.info(f"Done. Success: {success_count}, Failed: {fail_count}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
