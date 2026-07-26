#!/usr/bin/env python3
"""
Extract all metrics from summary_*.txt files and produce a single CSV per dataset.
Output: thesis-docs/notes/all_metrics_{dataset}.csv

Usage:
    python script/restoration/eval/extract_all_metrics_csv.py
"""

import csv
import os
import re
from pathlib import Path

# Directories to scan
SYNTHETIC_BASE = Path("comparison/results")
REALSR_BASE = Path("comparison_realsr/results")

METRICS_ORDER = ["PSNR", "SSIM", "LPIPS", "DELTA_E", "ARNIQA", "BRISQUE", "NIQE", "MANIQA", "MUSIQ"]


def parse_summary_file(filepath: Path) -> dict:
    """Parse a summary_*.txt file and return {metric_name: mean_value}."""
    result = {}
    n_images = None
    with open(filepath, "r") as f:
        text = f.read()

    # Extract number of images
    m = re.search(r"Number of images:\s*(\d+)", text)
    if m:
        n_images = int(m.group(1))

    # Extract mean values for each metric
    # Pattern: METRIC_NAME ↑/↓\n  Mean:   VALUE ± STD
    for metric in METRICS_ORDER:
        pattern = rf"{metric}\s*[↑↓]\s*\n\s*Mean:\s*([-\d.]+)\s*±\s*([-\d.]+)"
        m = re.search(pattern, text)
        if m:
            result[metric] = float(m.group(1))
            result[f"{metric}_std"] = float(m.group(2))

    return result, n_images


def model_name_from_filename(filename: str) -> str:
    """Convert summary filename to a readable model name."""
    name = filename.replace("summary_", "").replace(".txt", "")
    return name


def process_dataset_dir(metrics_dir: Path) -> list:
    """Process all summary files in a metrics directory."""
    rows = []
    if not metrics_dir.is_dir():
        return rows

    for f in sorted(metrics_dir.glob("summary_*.txt")):
        model_name = model_name_from_filename(f.name)
        metrics, n_images = parse_summary_file(f)
        if not metrics:
            continue
        row = {"model": model_name, "n_images": n_images}
        for metric in METRICS_ORDER:
            row[metric] = metrics.get(metric, "")
            row[f"{metric}_std"] = metrics.get(f"{metric}_std", "")
        rows.append(row)

    return rows


def write_csv(rows: list, output_path: Path):
    """Write rows to CSV."""
    if not rows:
        return
    fieldnames = ["model", "n_images"]
    for metric in METRICS_ORDER:
        fieldnames.append(metric)
        fieldnames.append(f"{metric}_std")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {output_path} ({len(rows)} models)")


def main():
    output_dir = Path("thesis-docs/notes")

    # Synthetic benchmarks
    print("=== Synthetic Benchmarks ===")
    for dataset in sorted(SYNTHETIC_BASE.iterdir()):
        if not dataset.is_dir():
            continue
        metrics_dir = dataset / "metrics"
        rows = process_dataset_dir(metrics_dir)
        if rows:
            out = output_dir / f"all_metrics_{dataset.name}.csv"
            write_csv(rows, out)

    # RealSR benchmarks
    print("\n=== RealSR Benchmarks ===")
    for dataset in sorted(REALSR_BASE.iterdir()):
        if not dataset.is_dir():
            continue
        metrics_dir = dataset / "metrics"
        rows = process_dataset_dir(metrics_dir)
        if rows:
            out = output_dir / f"all_metrics_{dataset.name}.csv"
            write_csv(rows, out)


if __name__ == "__main__":
    main()
