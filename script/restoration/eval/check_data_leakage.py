#!/usr/bin/env python3
"""
Data Leakage Check: CLIP-based Near-Duplicate Detection

Verifies that no near-duplicate images exist between the LAION training subset
and the evaluation datasets (DIV2K, Urban100, BSDS100).

Approach:
  1. Extract CLIP image embeddings for all training images (batched, GPU)
  2. Extract CLIP image embeddings for all evaluation images
  3. Compute cosine similarity between every (eval, train) pair
  4. Flag pairs exceeding the similarity threshold as potential duplicates
  5. Report results per evaluation dataset

The CLIP ViT-L/14 model produces 768-dim embeddings that are resolution-invariant
(images are resized to 224x224 internally), making it suitable for comparing
images at different resolutions (training: 768x768, eval: variable).

Usage:
    python script/restoration/eval/check_data_leakage.py

    # Custom threshold and paths
    python script/restoration/eval/check_data_leakage.py \
        --threshold 0.95 \
        --train_dir data/laion/train/clean \
        --eval_dirs comparison/results/DIV2K/clean \
                    comparison/results/Urban100/clean \
                    comparison/results/BSDS100/clean

    # Save embeddings for reuse
    python script/restoration/eval/check_data_leakage.py --save_embeddings

    # Use precomputed embeddings
    python script/restoration/eval/check_data_leakage.py \
        --load_embeddings output/leakage_check/embeddings.pt
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Default paths (relative to project root)
DEFAULT_TRAIN_DIR = "data/laion/train/clean"
DEFAULT_EVAL_DIRS = [
    "comparison/results/DIV2K/clean",
    "comparison/results/Urban100/clean",
    "comparison/results/BSDS100/clean",
]
DEFAULT_OUTPUT_DIR = "output/leakage_check"
DEFAULT_THRESHOLD = 0.93  # Cosine similarity threshold for near-duplicates
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def find_images(directory: Path) -> List[Path]:
    """Find all supported image files in a directory, sorted."""
    images = []
    for f in directory.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(f)
    return sorted(images)


@torch.no_grad()
def extract_clip_embeddings(
    image_paths: List[Path],
    model,
    processor,
    device: torch.device,
    batch_size: int = 64,
    desc: str = "Extracting embeddings",
) -> torch.Tensor:
    """
    Extract normalized CLIP image embeddings for a list of images.

    Returns:
        Tensor of shape (N, embed_dim) with L2-normalized embeddings.
    """
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc=desc):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")
                # Use a blank image as placeholder to maintain alignment
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        # Process batch through CLIP
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.get_image_features(**inputs)
        # Handle both tensor returns (older transformers) and object returns (5.x+)
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[1]
        # L2 normalize for cosine similarity
        embeddings = F.normalize(outputs, p=2, dim=-1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def compute_max_similarities(
    eval_embeddings: torch.Tensor,
    train_embeddings: torch.Tensor,
    chunk_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the maximum cosine similarity between each eval image and
    all training images. Uses chunked computation to avoid OOM.

    Returns:
        max_sims: (N_eval,) tensor of max similarities
        max_indices: (N_eval,) tensor of indices into train set
    """
    n_eval = eval_embeddings.shape[0]
    max_sims = torch.zeros(n_eval)
    max_indices = torch.zeros(n_eval, dtype=torch.long)

    # Process training embeddings in chunks to limit memory
    n_train = train_embeddings.shape[0]

    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        train_chunk = train_embeddings[start:end]  # (chunk, dim)

        # Cosine similarity: (N_eval, chunk)
        sims = eval_embeddings @ train_chunk.T

        chunk_max_sims, chunk_max_idx = sims.max(dim=1)
        # Adjust indices to global training set
        chunk_max_idx += start

        # Update global max
        better = chunk_max_sims > max_sims
        max_sims[better] = chunk_max_sims[better]
        max_indices[better] = chunk_max_idx[better]

    return max_sims, max_indices


def run_leakage_check(args):
    """Main leakage check pipeline."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Similarity threshold: {args.threshold}")

    # -------------------------------------------------------------------------
    # Load or compute embeddings
    # -------------------------------------------------------------------------
    if args.load_embeddings and Path(args.load_embeddings).exists():
        logger.info(f"Loading precomputed embeddings from {args.load_embeddings}")
        data = torch.load(args.load_embeddings, map_location="cpu")
        train_embeddings = data["train_embeddings"]
        train_paths = data["train_paths"]
        eval_data = data["eval_data"]
        logger.info(
            f"Loaded: {len(train_paths)} train embeddings, "
            f"{len(eval_data)} eval datasets"
        )
    else:
        # Load CLIP model
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        model.eval()

        # Extract training embeddings
        train_dir = Path(args.train_dir)
        assert train_dir.exists(), f"Training directory not found: {train_dir}"
        train_paths = find_images(train_dir)
        logger.info(f"Found {len(train_paths)} training images in {train_dir}")

        train_embeddings = extract_clip_embeddings(
            train_paths,
            model,
            processor,
            device,
            batch_size=args.batch_size,
            desc="Training set embeddings",
        )
        logger.info(f"Training embeddings shape: {train_embeddings.shape}")

        # Extract eval embeddings per dataset
        eval_data = {}
        for eval_dir_str in args.eval_dirs:
            eval_dir = Path(eval_dir_str)
            if not eval_dir.exists():
                logger.warning(f"Eval directory not found, skipping: {eval_dir}")
                continue

            dataset_name = eval_dir.parent.name  # e.g., "DIV2K", "Urban100"
            eval_paths = find_images(eval_dir)
            logger.info(f"Found {len(eval_paths)} images in {dataset_name}")

            eval_embeddings = extract_clip_embeddings(
                eval_paths,
                model,
                processor,
                device,
                batch_size=args.batch_size,
                desc=f"{dataset_name} embeddings",
            )

            eval_data[dataset_name] = {
                "embeddings": eval_embeddings,
                "paths": [str(p) for p in eval_paths],
            }

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Optionally save embeddings
        if args.save_embeddings:
            save_path = output_dir / "embeddings.pt"
            torch.save(
                {
                    "train_embeddings": train_embeddings,
                    "train_paths": [str(p) for p in train_paths],
                    "eval_data": eval_data,
                },
                save_path,
            )
            logger.info(f"Saved embeddings to {save_path}")

    # -------------------------------------------------------------------------
    # Compute similarities and detect near-duplicates
    # -------------------------------------------------------------------------
    results_summary = []
    all_flagged = []

    for dataset_name, data in eval_data.items():
        eval_embeddings = data["embeddings"]
        eval_paths = data["paths"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Checking {dataset_name} ({len(eval_paths)} images)")
        logger.info(f"{'='*60}")

        max_sims, max_indices = compute_max_similarities(
            eval_embeddings, train_embeddings, chunk_size=2000
        )

        # Statistics
        mean_sim = max_sims.mean().item()
        std_sim = max_sims.std().item()
        max_sim = max_sims.max().item()
        min_sim = max_sims.min().item()

        logger.info(f"Similarity statistics:")
        logger.info(f"  Mean: {mean_sim:.4f} +/- {std_sim:.4f}")
        logger.info(f"  Range: [{min_sim:.4f}, {max_sim:.4f}]")

        # Flag potential duplicates
        flagged_mask = max_sims > args.threshold
        n_flagged = flagged_mask.sum().item()

        logger.info(
            f"  Flagged (>{args.threshold}): {n_flagged}/{len(eval_paths)} images"
        )

        dataset_result = {
            "dataset": dataset_name,
            "n_images": len(eval_paths),
            "n_flagged": n_flagged,
            "mean_sim": mean_sim,
            "std_sim": std_sim,
            "max_sim": max_sim,
            "min_sim": min_sim,
            "threshold": args.threshold,
        }
        results_summary.append(dataset_result)

        # Detail flagged pairs
        if n_flagged > 0:
            flagged_indices = torch.where(flagged_mask)[0]
            for idx in flagged_indices:
                eval_path = eval_paths[idx.item()]
                train_idx = max_indices[idx.item()].item()
                sim_value = max_sims[idx.item()].item()

                if isinstance(train_paths[0], str):
                    train_path = train_paths[train_idx]
                else:
                    train_path = str(train_paths[train_idx])

                flagged_entry = {
                    "dataset": dataset_name,
                    "eval_image": Path(eval_path).name,
                    "train_image": Path(train_path).name,
                    "similarity": sim_value,
                }
                all_flagged.append(flagged_entry)
                logger.warning(
                    f"  POTENTIAL DUPLICATE: {Path(eval_path).name} <-> "
                    f"{Path(train_path).name} (sim={sim_value:.4f})"
                )

    # -------------------------------------------------------------------------
    # Write report
    # -------------------------------------------------------------------------
    report_path = output_dir / "leakage_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DATA LEAKAGE CHECK REPORT\n")
        f.write(
            "CLIP-based Near-Duplicate Detection between Training and Eval Sets\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(f"Model: {CLIP_MODEL_NAME}\n")
        f.write(f"Similarity threshold: {args.threshold}\n")
        f.write(f"Training set: {args.train_dir} ({len(train_paths)} images)\n\n")

        f.write("-" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n\n")

        total_flagged = sum(r["n_flagged"] for r in results_summary)
        total_eval = sum(r["n_images"] for r in results_summary)

        f.write(f"Total evaluation images checked: {total_eval}\n")
        f.write(f"Total potential duplicates found: {total_flagged}\n\n")

        for r in results_summary:
            f.write(
                f"  {r['dataset']:12s}: {r['n_flagged']:3d}/{r['n_images']:3d} flagged "
                f"| max_sim={r['max_sim']:.4f} | mean={r['mean_sim']:.4f} "
                f"+/- {r['std_sim']:.4f}\n"
            )

        f.write("\n")

        if total_flagged == 0:
            f.write("-" * 70 + "\n")
            f.write("CONCLUSION: NO DATA LEAKAGE DETECTED\n")
            f.write("-" * 70 + "\n\n")
            f.write(
                "No near-duplicate images were found between the LAION training\n"
                "subset and the evaluation datasets at the specified threshold.\n"
                "This provides empirical evidence that the quantitative evaluation\n"
                "metrics are not inflated by training-test overlap.\n"
            )
        else:
            f.write("-" * 70 + "\n")
            f.write("FLAGGED PAIRS (require manual inspection)\n")
            f.write("-" * 70 + "\n\n")
            for entry in all_flagged:
                f.write(
                    f"  [{entry['dataset']}] {entry['eval_image']} <-> "
                    f"{entry['train_image']} (cosine_sim={entry['similarity']:.4f})\n"
                )
            f.write(
                "\nNote: High similarity does not guarantee exact duplication.\n"
                "Visual inspection is recommended for flagged pairs.\n"
            )

        # Histogram of similarities for each dataset
        f.write("\n" + "-" * 70 + "\n")
        f.write("SIMILARITY DISTRIBUTION (top-1 per eval image)\n")
        f.write("-" * 70 + "\n\n")

        for dataset_name, data in eval_data.items():
            eval_embeddings = data["embeddings"]
            max_sims, _ = compute_max_similarities(
                eval_embeddings, train_embeddings, chunk_size=2000
            )
            sims_np = max_sims.numpy()

            # Histogram bins
            bins = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 1.0]
            f.write(f"  {dataset_name}:\n")
            for i in range(len(bins) - 1):
                count = np.sum((sims_np >= bins[i]) & (sims_np < bins[i + 1]))
                bar = "#" * count
                f.write(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:3d} {bar}\n")
            f.write("\n")

    logger.info(f"\nReport saved to: {report_path}")

    # Also save per-image similarities as CSV for further analysis
    csv_path = output_dir / "per_image_similarities.csv"
    rows = []
    for dataset_name, data in eval_data.items():
        eval_embeddings = data["embeddings"]
        eval_paths_list = data["paths"]
        max_sims, max_indices = compute_max_similarities(
            eval_embeddings, train_embeddings, chunk_size=2000
        )
        for i in range(len(eval_paths_list)):
            train_idx = max_indices[i].item()
            if isinstance(train_paths[0], str):
                train_path = train_paths[train_idx]
            else:
                train_path = str(train_paths[train_idx])
            rows.append(
                {
                    "dataset": dataset_name,
                    "eval_image": Path(eval_paths_list[i]).name,
                    "most_similar_train_image": Path(train_path).name,
                    "cosine_similarity": max_sims[i].item(),
                }
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Per-image similarities saved to: {csv_path}")

    # Print final verdict
    print("\n" + "=" * 70)
    if total_flagged == 0:
        print(f"RESULT: NO DATA LEAKAGE DETECTED (threshold={args.threshold})")
        print(f"All {total_eval} evaluation images have max similarity < {args.threshold}")
    else:
        print(f"RESULT: {total_flagged} POTENTIAL DUPLICATES FOUND")
        print("Manual inspection recommended for flagged pairs.")
    print("=" * 70)

    return total_flagged


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check for data leakage between training and evaluation sets "
        "using CLIP embedding similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=DEFAULT_TRAIN_DIR,
        help=f"Directory of clean training images (default: {DEFAULT_TRAIN_DIR})",
    )
    parser.add_argument(
        "--eval_dirs",
        type=str,
        nargs="+",
        default=DEFAULT_EVAL_DIRS,
        help="Directories of clean evaluation images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for flagging (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for report (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for CLIP inference (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (default: cuda)",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save computed embeddings to disk for reuse",
    )
    parser.add_argument(
        "--load_embeddings",
        type=str,
        default=None,
        help="Path to precomputed embeddings file (.pt)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_flagged = run_leakage_check(args)
    sys.exit(0 if n_flagged == 0 else 1)
