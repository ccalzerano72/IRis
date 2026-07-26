#!/usr/bin/env python3
"""
Recalculate SSIM with Wang et al. 2004 standard parameters:
  - gaussian_weights=True
  - sigma=1.5
  - win_size auto-computed to 11

This script scans comparison/results/ and comparison_realsr/results/ for
existing metrics_*.csv files, reads the clean_path and restored_path columns,
recomputes SSIM with the standard configuration, and outputs:
  - ssim_11/metrics_<model>.csv  (per-image SSIM)
  - ssim_11/summary_<model>.txt  (statistics)

Parallelized at model level: each CSV is processed by an independent worker.

Usage:
    python script/restoration/eval/recalculate_ssim_11.py [--dry-run] [--workers N]
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def compute_ssim_wang2004(clean_path: str, restored_path: str) -> float:
    """
    Compute SSIM between two images using Wang et al. 2004 parameters:
    gaussian_weights=True, sigma=1.5 (auto win_size=11).
    """
    clean_img = np.array(Image.open(clean_path).convert("RGB"))
    restored_img = np.array(Image.open(restored_path).convert("RGB"))

    # Resize restored to clean size if they differ
    if clean_img.shape != restored_img.shape:
        restored_pil = Image.open(restored_path).convert("RGB")
        restored_pil = restored_pil.resize(
            (clean_img.shape[1], clean_img.shape[0]), Image.Resampling.LANCZOS
        )
        restored_img = np.array(restored_pil)

    ssim_value = structural_similarity(
        clean_img,
        restored_img,
        data_range=255,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
    )
    return round(float(ssim_value), 4)


def write_summary(output_path: Path, model_name: str, filenames: list, ssim_values: list):
    """Write a summary .txt file with SSIM statistics."""
    arr = np.array(ssim_values)
    best_idx = int(np.argmax(arr))
    worst_idx = int(np.argmin(arr))

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"SSIM (Wang 2004, Gaussian 11x11, sigma=1.5) - {model_name.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of images: {len(ssim_values)}\n")
        f.write("=" * 70 + "\n")
        f.write("SSIM (gaussian_weights=True, sigma=1.5, win_size=11)\n")
        f.write(f"  Mean:   {arr.mean():.4f} +/- {arr.std():.4f}\n")
        f.write(f"  Median: {np.median(arr):.4f}\n")
        f.write(f"  Range:  [{arr.min():.4f}, {arr.max():.4f}]\n")
        f.write(f"  Best:   {arr[best_idx]:.4f} ({filenames[best_idx]})\n")
        f.write(f"  Worst:  {arr[worst_idx]:.4f} ({filenames[worst_idx]})\n")
        f.write("=" * 70 + "\n")


def process_single_csv(csv_path_str: str, output_dir_str: str) -> str:
    """
    Worker function: process a single metrics CSV.
    Returns a status string for logging.
    Must be a top-level function for multiprocessing pickling.
    """
    csv_path = Path(csv_path_str)
    output_dir = Path(output_dir_str)

    csv_name = csv_path.stem
    model_name = csv_name.replace("metrics_", "", 1)

    out_csv = output_dir / f"metrics_{model_name}.csv"
    out_summary = output_dir / f"summary_{model_name}.txt"

    # Skip if already computed
    if out_csv.exists() and out_summary.exists():
        return f"[SKIP] {model_name}"

    # Read existing CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return f"[WARN] Empty CSV: {csv_path}"

    if "clean_path" not in rows[0] or "restored_path" not in rows[0]:
        return f"[WARN] Missing clean_path/restored_path: {csv_path}"

    filenames = []
    ssim_values = []
    errors = 0

    for row in rows:
        filename = row["filename"]
        clean_path = row["clean_path"]
        restored_path = row["restored_path"]

        if not os.path.isfile(clean_path) or not os.path.isfile(restored_path):
            errors += 1
            continue

        try:
            ssim_val = compute_ssim_wang2004(clean_path, restored_path)
            filenames.append(filename)
            ssim_values.append(ssim_val)
        except Exception:
            errors += 1

    if not ssim_values:
        return f"[WARN] No valid results for {model_name} ({errors} errors)"

    # Write output CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ssim"])
        for fname, val in zip(filenames, ssim_values):
            writer.writerow([fname, val])

    # Write summary
    write_summary(out_summary, model_name, filenames, ssim_values)

    err_msg = f", {errors} errors" if errors else ""
    return f"[OK] {model_name}: mean={np.mean(ssim_values):.4f} ({len(ssim_values)} images{err_msg})"


def find_metrics_dirs(base_paths: list) -> list:
    """Find all 'metrics' directories under the given base paths."""
    metrics_dirs = []
    for base in base_paths:
        base_path = Path(base)
        if not base_path.exists():
            print(f"[WARN] Base path does not exist: {base}", flush=True)
            continue
        for metrics_dir in sorted(base_path.rglob("metrics")):
            if metrics_dir.is_dir():
                metrics_dirs.append(metrics_dir)
    return metrics_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate SSIM with Wang 2004 standard (Gaussian 11x11, sigma=1.5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without computing"
    )
    parser.add_argument(
        "--workers", type=int, default=20,
        help="Number of parallel workers (default: 20)"
    )
    parser.add_argument(
        "--base-dirs", nargs="+",
        default=["comparison/results", "comparison_realsr/results"],
        help="Base directories to scan for metrics/ folders"
    )
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else 20

    print("=" * 70, flush=True)
    print("SSIM Recalculation - Wang et al. 2004 Standard", flush=True)
    print(f"  gaussian_weights=True, sigma=1.5, win_size=11 (auto)", flush=True)
    print(f"  Workers: {n_workers}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    metrics_dirs = find_metrics_dirs(args.base_dirs)

    if not metrics_dirs:
        print("[ERROR] No metrics directories found.", flush=True)
        sys.exit(1)

    # Collect all (csv_path, output_dir) jobs
    jobs = []
    for metrics_dir in metrics_dirs:
        ssim_dir = metrics_dir.parent / "ssim_11"
        csv_files = sorted(metrics_dir.glob("metrics_*.csv"))
        for csv_path in csv_files:
            jobs.append((str(csv_path), str(ssim_dir)))

    print(f"Found {len(metrics_dirs)} metrics directories, {len(jobs)} total models", flush=True)
    for d in metrics_dirs:
        n_csv = len(list(d.glob("metrics_*.csv")))
        print(f"  {d} ({n_csv} models)", flush=True)
    print(flush=True)

    if args.dry_run:
        for csv_path_str, output_dir_str in jobs:
            model_name = Path(csv_path_str).stem.replace("metrics_", "", 1)
            out_csv = Path(output_dir_str) / f"metrics_{model_name}.csv"
            out_summary = Path(output_dir_str) / f"summary_{model_name}.txt"
            if out_csv.exists() and out_summary.exists():
                print(f"  [SKIP] {model_name}", flush=True)
            else:
                print(f"  [DRY-RUN] Would process: {model_name}", flush=True)
        print(f"\nDry-run complete. {len(jobs)} models found.", flush=True)
        return

    # Process in parallel
    completed = 0
    skipped = 0
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_job = {
            executor.submit(process_single_csv, csv_path, output_dir): (csv_path, output_dir)
            for csv_path, output_dir in jobs
        }

        for future in as_completed(future_to_job):
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                csv_path, _ = future_to_job[future]
                result = f"[ERROR] {Path(csv_path).stem}: {e}"

            if "[SKIP]" in result:
                skipped += 1
            else:
                # Only print non-skip results to reduce noise
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"  [{completed}/{len(jobs)}] ({elapsed:.0f}s) {result}", flush=True)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(flush=True)
    print("=" * 70, flush=True)
    print(f"Done in {elapsed:.1f}s. Total: {len(jobs)}, Skipped: {skipped}, Computed: {completed - skipped}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
