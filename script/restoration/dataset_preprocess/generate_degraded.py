# Generate degraded images from clean reference images
# Uses ARNIQA degradation framework

import argparse
import torch
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path to import ARNIQA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.ARNIQA.degradation import ImageDistorter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate degraded images from clean reference images"
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        required=True,
        help="Path to clean reference images directory"
    )
    parser.add_argument(
        "--degraded_dir",
        type=str,
        required=True,
        help="Path to output degraded images directory"
    )
    parser.add_argument(
        "--distortion_name",
        type=str,
        required=True,
        help="Name of distortion to apply (e.g., whitenoise, jpeg, gaublur)"
    )
    parser.add_argument(
        "--level",
        type=int,
        required=True,
        help="Distortion level (0-4)"
    )
    
    args = parser.parse_args()
    
    clean_dir = args.clean_dir
    degraded_dir = args.degraded_dir
    distortion_name = args.distortion_name
    level = args.level
    
    # Validate level
    if not 0 <= level <= 4:
        print(f"Error: Level must be between 0 and 4, received: {level}")
        exit(1)
    
    # Initialize ImageDistorter
    print("Initializing ImageDistorter...")
    distorter = ImageDistorter()
    
    # Validate distortion name
    if distortion_name not in distorter.distortion_functions:
        available = ", ".join(distorter.distortion_functions.keys())
        print(f"Error: Distortion '{distortion_name}' not supported.")
        print(f"Available distortions: {available}")
        exit(1)
    
    # Image extensions to process
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    clean_path = Path(clean_dir)
    degraded_path = Path(degraded_dir)
    
    # Create output directory
    os.makedirs(degraded_dir, exist_ok=True)
    
    # Find all image files recursively (use set to avoid duplicates on Windows)
    image_files_set = set()
    for ext in image_extensions:
        image_files_set.update(clean_path.rglob(f'*{ext}'))
        image_files_set.update(clean_path.rglob(f'*{ext.upper()}'))
    image_files = sorted(list(image_files_set))
    
    print(f"Found {len(image_files)} images in {clean_dir}")
    print(f"Applying distortion: {distortion_name}, level: {level}")
    
    if len(image_files) == 0:
        print(f"Warning: No images found")
        exit(0)
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for img_path in tqdm(image_files, desc="Generating degraded images"):
        try:
            # Load image as tensor using ARNIQA's method
            image_tensor = distorter.load_image(str(img_path))
            
            # Apply distortion
            degraded_tensor = distorter.apply_distortion_to_tensor(
                image_tensor, distortion_name, level
            )
            
            # Create relative path structure in destination
            rel_path = img_path.relative_to(clean_path)
            dest_file = degraded_path / rel_path
            
            # Create subdirectories if needed
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save degraded image using ARNIQA's method
            distorter.save_image(degraded_tensor, str(dest_file))
            processed_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            skipped_count += 1
            continue
    
    print(f"\nProcessing complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(image_files)}")
    
    # Create filename list
    filename_list_path = os.path.join(degraded_dir, "filename_list.txt")
    with open(filename_list_path, 'w') as f:
        for img_path in sorted(degraded_path.rglob('*.png')):
            rel_path = img_path.relative_to(degraded_path)
            f.write(f"{rel_path}\n")
        for img_path in sorted(degraded_path.rglob('*.jpg')):
            rel_path = img_path.relative_to(degraded_path)
            f.write(f"{rel_path}\n")
    
    print(f"Filename list saved to: {filename_list_path}")
    print("Generation finished")
