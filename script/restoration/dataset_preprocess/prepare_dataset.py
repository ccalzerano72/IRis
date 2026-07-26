#!/usr/bin/env python
"""
Dataset preparation script for restoration training.

This script:
1. Reads configuration from config/dataset_restoration/dataset_base.yaml
2. Processes images from source directory
3. Generates degraded versions
4. Creates file lists for Marigold

Usage:
    python script/restoration/dataset_preprocess/prepare_dataset.py --source_dir /path/to/images
"""

import argparse
import os
import sys
import random
from pathlib import Path
from tqdm import tqdm

# Add project root to path
# __file__ is in: script/restoration/dataset_preprocess/prepare_dataset.py
# We need to go up 3 levels to reach project root
script_dir = Path(__file__).resolve().parent  # script/restoration/dataset_preprocess
project_root = script_dir.parent.parent.parent  # Go up 3 levels
sys.path.insert(0, str(project_root))

# Debug: print paths if import fails
try:
    from omegaconf import OmegaConf
    from src.util.config_util import recursive_load_config
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print(f"Script dir: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    print(f"\nPlease run the script from the project root directory:")
    print(f"  cd {project_root}")
    print(f"  python script/restoration/dataset_preprocess/prepare_dataset.py --source_dir /path/to/images")
    sys.exit(1)

# Configuration constants
TRAIN_SPLIT = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42   # For reproducible splits


def load_config():
    """Load configuration from dataset_base.yaml with recursive base_config loading."""
    config_path = project_root / "config" / "dataset_restoration" / "dataset_base.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Use recursive_load_config to merge base_config files
    cfg = recursive_load_config(str(config_path))
    
    # Resolve variable interpolations
    OmegaConf.resolve(cfg)
    
    return cfg


def find_images(source_dir):
    """Find all image files in source directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    source_path = Path(source_dir)
    
    # Find all image files (use set to avoid duplicates on Windows)
    image_files_set = set()
    for ext in image_extensions:
        image_files_set.update(source_path.rglob(f'*{ext}'))
        image_files_set.update(source_path.rglob(f'*{ext.upper()}'))
    
    return sorted(list(image_files_set))


def split_train_val(image_files, train_ratio, seed):
    """Split image files into train and validation sets."""
    random.seed(seed)
    
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)
    
    split_idx = int(len(shuffled_files) * train_ratio)
    
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    return train_files, val_files


def get_train_val_images(source_dir, split_config):
    """
    Get train and val images based on split configuration.
    
    Args:
        source_dir: Source directory path
        split_config: Split configuration from dataset_base.yaml
        
    Returns:
        tuple: (train_images, val_images)
    """
    split_mode = split_config.mode
    
    if split_mode == 'presplit':
        # Mode 1: Separate train/val folders
        train_subfolder = split_config.train_subfolder
        val_subfolder = split_config.val_subfolder
        
        train_dir = os.path.join(source_dir, train_subfolder)
        val_dir = os.path.join(source_dir, val_subfolder)
        
        print(f"  Mode: presplit")
        print(f"  Train folder: {train_subfolder}")
        print(f"  Val folder: {val_subfolder}")
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Val directory not found: {val_dir}")
        
        train_images = find_images(train_dir)
        val_images = find_images(val_dir)
        
        # Store source dirs for later use
        return train_images, val_images, train_dir, val_dir
        
    elif split_mode == 'auto':
        # Mode 2: Single folder, auto split
        train_ratio = split_config.train_ratio
        random_seed = split_config.random_seed
        
        print(f"  Mode: auto")
        print(f"  Train ratio: {train_ratio} ({int(train_ratio*100)}%)")
        print(f"  Random seed: {random_seed}")
        
        all_images = find_images(source_dir)
        train_images, val_images = split_train_val(all_images, train_ratio, random_seed)
        
        # Both use same source dir
        return train_images, val_images, source_dir, source_dir
        
    else:
        raise ValueError(f"Unknown split mode: {split_mode}. Use 'presplit' or 'auto'")


def copy_and_preprocess_images(image_files, source_dir, dest_dir, target_h, target_w, force=False):
    """Copy images to destination and preprocess them.
    
    Args:
        image_files: List of image file paths to process
        source_dir: Source directory path
        dest_dir: Destination directory path
        target_h: Target height
        target_w: Target width
        force: If True, overwrite existing files. If False (default), skip existing files.
    
    Returns:
        tuple: (processed_count, skipped_count)
    """
    import cv2
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    for img_path in tqdm(image_files, desc=f"Processing to {dest_path.name}"):
        # Create relative path structure
        rel_path = img_path.relative_to(source_path)
        dest_file = dest_path / rel_path
        
        # Skip if file exists and force=False
        if dest_file.exists() and not force:
            skipped_count += 1
            continue
        
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        h, w = image.shape[:2]
        
        # Resize so min dimension matches target min dimension
        target_min = min(target_h, target_w)
        if h < w:
            new_h = target_min
            new_w = int(w * (target_min / h))
        else:
            new_w = target_min
            new_h = int(h * (target_min / w))
        
        # Choose interpolation
        if new_w < w or new_h < h:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center crop to target dimensions
        h_resized, w_resized = resized.shape[:2]
        start_y = (h_resized - target_h) // 2
        start_x = (w_resized - target_w) // 2
        cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]
        
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with maximum quality
        if dest_file.suffix.lower() in ['.png']:
            cv2.imwrite(str(dest_file), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif dest_file.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(dest_file), cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            cv2.imwrite(str(dest_file), cropped)
        
        processed_count += 1
    
    return processed_count, skipped_count


def generate_degraded_images(clean_dir, degraded_dir, distortion_name, level, force=False):
    """Generate degraded images from clean images.
    
    Args:
        clean_dir: Clean images directory
        degraded_dir: Degraded images output directory
        distortion_name: ARNIQA distortion name
        level: Distortion severity level (0-4)
        force: If True, overwrite existing files. If False (default), skip existing files.
    
    Returns:
        tuple: (processed_count, skipped_count)
    """
    from src.ARNIQA.degradation import ImageDistorter
    
    clean_path = Path(clean_dir)
    degraded_path = Path(degraded_dir)
    degraded_path.mkdir(parents=True, exist_ok=True)
    
    distorter = ImageDistorter()
    image_files = find_images(clean_dir)
    
    processed_count = 0
    skipped_count = 0
    
    for img_path in tqdm(image_files, desc=f"Generating {distortion_name} level {level}"):
        try:
            rel_path = img_path.relative_to(clean_path)
            dest_file = degraded_path / rel_path
            
            # Skip if file exists and force=False
            if dest_file.exists() and not force:
                skipped_count += 1
                continue
            
            image_tensor = distorter.load_image(str(img_path))
            degraded_tensor = distorter.apply_distortion_to_tensor(
                image_tensor, distortion_name, level
            )
            
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            distorter.save_image(degraded_tensor, str(dest_file))
            processed_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    return processed_count, skipped_count


def generate_file_lists(clean_dir, degraded_dir, output_dir, dataset_name, split_name):
    """Generate file lists for Marigold dataset loading."""
    clean_path = Path(clean_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(find_images(clean_dir))
    file_list_path = output_path / f"{dataset_name}_{split_name}.txt"
    
    with open(file_list_path, 'w') as f:
        for img_path in image_files:
            rel_path = img_path.relative_to(clean_path)
            f.write(f"{rel_path} {rel_path}\n")
    
    print(f"Generated file list: {file_list_path} ({len(image_files)} entries)")
    return file_list_path


def generate_vis_sample_list(clean_dir, output_dir, dataset_name, num_samples=10):
    """Generate visualization sample file list."""
    clean_path = Path(clean_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(find_images(clean_dir))
    
    random.seed(RANDOM_SEED)
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    sample_files = sorted(sample_files)
    
    file_list_path = output_path / f"{dataset_name}_vis_sample.txt"
    
    with open(file_list_path, 'w') as f:
        for img_path in sample_files:
            rel_path = img_path.relative_to(clean_path)
            f.write(f"{rel_path} {rel_path}\n")
    
    print(f"Generated vis sample list: {file_list_path} ({len(sample_files)} entries)")
    return file_list_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for restoration training (reads config automatically)"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to source images directory (e.g., /path/to/DIV2K_train_HR)"
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=None,
        help="Base data directory (default: uses BASE_DATA_DIR env var or ./data)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files (default: skip existing files)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("=" * 60)
    print("Loading configuration from dataset_base.yaml...")
    print("=" * 60)
    
    cfg = load_config()
    
    # Extract config values
    dataset_name = cfg.dataset_name
    dataset_root = cfg.dataset_root
    target_h, target_w = cfg.resize_to_hw  # [height, width]
    
    # Split config
    split_config = cfg.split_config
    
    # Degradation config
    deg_types = cfg.degradation_config.types
    deg_levels = cfg.degradation_config.levels
    
    # Use first degradation type and level for now
    distortion_name = deg_types[0]
    level = deg_levels[0]
    
    # Determine base data directory
    if args.base_data_dir:
        base_data_dir = args.base_data_dir
    elif 'BASE_DATA_DIR' in os.environ:
        base_data_dir = os.environ['BASE_DATA_DIR']
    else:
        base_data_dir = str(project_root / "data")
        print(f"⚠ BASE_DATA_DIR not set, using default: {base_data_dir}")
    
    dest_dir = os.path.join(base_data_dir, dataset_root)
    
    # Get degradation mode
    degradation_mode = cfg.degradation_config.mode
    
    print(f"\n✓ Configuration loaded:")
    print(f"  - Dataset name: {dataset_name}")
    print(f"  - Dataset root: {dataset_root}")
    print(f"  - Target size: {target_h}x{target_w} (HxW)")
    print(f"  - Degradation mode: {degradation_mode}")
    if degradation_mode == 'pre_generated':
        print(f"  - Degradation type: {distortion_name}, level: {level}")
    elif degradation_mode == 'online':
        print(f"  - Degradation types: {', '.join(cfg.degradation_config.types)}")
        print(f"  - Degradation levels: {cfg.degradation_config.levels}")
    print(f"  - Base data dir: {base_data_dir}")
    print(f"  - Destination: {dest_dir}")
    print(f"  - Force overwrite: {args.force}")
    
    print("\n" + "=" * 60)
    print("Dataset Preparation for Image Restoration")
    print("=" * 60)
    print(f"Source directory: {args.source_dir}")
    print(f"Destination directory: {dest_dir}")
    print(f"Force overwrite: {'Yes' if args.force else 'No (skip existing)'}")
    print("=" * 60)
    
    # Step 1 & 2: Get train/val images based on split config
    print("\n[1/7] Getting train/val images...")
    train_images, val_images, train_source_dir, val_source_dir = get_train_val_images(
        args.source_dir, split_config
    )
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    
    if len(train_images) == 0:
        print("Error: No train images found")
        return 1
    if len(val_images) == 0:
        print("Error: No val images found")
        return 1
    
    # Step 2: Preprocess train
    print("\n[2/7] Preprocessing clean train images...")
    train_clean_dir = os.path.join(dest_dir, "train", "clean")
    train_count, train_skipped = copy_and_preprocess_images(
        train_images, train_source_dir, train_clean_dir, target_h, target_w, force=args.force
    )
    print(f"Processed {train_count} train images" + (f" (skipped {train_skipped} existing)" if train_skipped > 0 else ""))
    
    # Step 3: Preprocess val
    print("\n[3/7] Preprocessing clean validation images...")
    val_clean_dir = os.path.join(dest_dir, "val", "clean")
    val_count, val_skipped = copy_and_preprocess_images(
        val_images, val_source_dir, val_clean_dir, target_h, target_w, force=args.force
    )
    print(f"Processed {val_count} validation images" + (f" (skipped {val_skipped} existing)" if val_skipped > 0 else ""))
    
    # Step 4: Generate degraded (ONLY if mode=pre_generated)
    degradation_mode = cfg.degradation_config.mode
    
    if degradation_mode == 'pre_generated':
        print("\n[4/7] Generating degraded images (mode=pre_generated)...")
        
        print("  - Train degraded images...")
        train_degraded_dir = os.path.join(dest_dir, "train", "degraded")
        train_deg_count, train_deg_skipped = generate_degraded_images(
            train_clean_dir, train_degraded_dir, distortion_name, level, force=args.force
        )
        print(f"  Generated {train_deg_count} train degraded images" + (f" (skipped {train_deg_skipped} existing)" if train_deg_skipped > 0 else ""))
        
        print("  - Validation degraded images...")
        val_degraded_dir = os.path.join(dest_dir, "val", "degraded")
        val_deg_count, val_deg_skipped = generate_degraded_images(
            val_clean_dir, val_degraded_dir, distortion_name, level, force=args.force
        )
        print(f"  Generated {val_deg_count} validation degraded images" + (f" (skipped {val_deg_skipped} existing)" if val_deg_skipped > 0 else ""))
        
    elif degradation_mode == 'online':
        print("\n[4/7] Skipping degraded generation (mode=online)")
        print("  ℹ️  Degradations will be generated on-the-fly during training")
        print("  ℹ️  This saves storage and provides infinite degradation variety")
        
        # Set dummy values for summary
        train_degraded_dir = None
        val_degraded_dir = None
        train_deg_count = 0
        val_deg_count = 0
        
    else:
        raise ValueError(f"Unknown degradation mode: {degradation_mode}. Use 'pre_generated' or 'online'")
    
    # Step 5: Generate file lists
    print("\n[5/7] Generating file lists for Marigold...")
    file_list_dir = str(project_root / "data_split" / "restoration")
    
    print("  - Train file list...")
    train_list = generate_file_lists(
        train_clean_dir, train_degraded_dir, file_list_dir, dataset_name, "train"
    )
    
    print("  - Validation file list...")
    val_list = generate_file_lists(
        val_clean_dir, val_degraded_dir, file_list_dir, dataset_name, "val"
    )
    
    # Step 6: Generate vis sample
    print("\n[6/7] Generating visualization sample list...")
    vis_list = generate_vis_sample_list(
        val_clean_dir, file_list_dir, dataset_name, num_samples=10
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"Train set: {train_count} clean images")
    print(f"Val set: {val_count} clean images")
    
    if degradation_mode == 'pre_generated':
        print(f"  + {train_deg_count} train degraded")
        print(f"  + {val_deg_count} val degraded")
        print(f"\nDegradation mode: pre_generated")
        print(f"  ℹ️  Degraded images saved to disk")
        print(f"\nOutput structure:")
        print(f"  {dest_dir}/")
        print(f"    ├── train/")
        print(f"    │   ├── clean/")
        print(f"    │   └── degraded/")
        print(f"    └── val/")
        print(f"        ├── clean/")
        print(f"        └── degraded/")
    elif degradation_mode == 'online':
        print(f"\nDegradation mode: online")
        print(f"  ℹ️  Degradations will be generated during training")
        print(f"  ℹ️  Types: {', '.join(cfg.degradation_config.types)}")
        print(f"  ℹ️  Levels: {cfg.degradation_config.levels}")
        print(f"\nOutput structure:")
        print(f"  {dest_dir}/")
        print(f"    ├── train/")
        print(f"    │   └── clean/")
        print(f"    └── val/")
        print(f"        └── clean/")
    
    print(f"\nFile lists generated:")
    print(f"  {train_list}")
    print(f"  {val_list}")
    print(f"  {vis_list}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
