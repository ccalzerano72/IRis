# Dataset preprocessing for image restoration task
# Resizes and crops images to square format for training

import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm


def resize_and_crop(image, target_size=512):
    """
    Resize and crop image to square format.
    
    Process:
    1. Resize so that min(width, height) = target_size
    2. Center crop to target_size x target_size
    
    Args:
        image: Input image (numpy array)
        target_size: Target size for output (default: 512)
    
    Returns:
        Processed image of shape (target_size, target_size, channels)
    """
    h, w = image.shape[:2]
    
    # Step 1: Resize so min dimension = target_size
    if h < w:
        # Height is smaller, resize based on height
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        # Width is smaller or equal, resize based on width
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    # Resize image using appropriate interpolation
    # INTER_AREA for downsampling (best quality)
    # INTER_LANCZOS4 for upsampling (best quality)
    if new_w < w or new_h < h:
        # Downsampling
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # Upsampling
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 2: Center crop to target_size x target_size
    h_resized, w_resized = resized.shape[:2]
    
    # Calculate crop coordinates (center crop)
    start_y = (h_resized - target_size) // 2
    start_x = (w_resized - target_size) // 2
    
    cropped = resized[start_y:start_y + target_size, start_x:start_x + target_size]
    
    return cropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess images for restoration task: resize and crop to square format"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to source image directory"
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Path to destination directory"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target size for output images (default: 512)"
    )
    
    args = parser.parse_args()
    
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    target_size = args.target_size
    
    # Image extensions to process
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all image files recursively (use set to avoid duplicates on Windows)
    image_files_set = set()
    for ext in image_extensions:
        image_files_set.update(source_path.rglob(f'*{ext}'))
        image_files_set.update(source_path.rglob(f'*{ext.upper()}'))
    image_files = sorted(list(image_files_set))
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    if len(image_files) == 0:
        print(f"Warning: No images found")
        exit(0)
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Warning: Could not read {img_path}")
            skipped_count += 1
            continue
        
        # Process image (resize and crop to target_size x target_size)
        processed = resize_and_crop(image, target_size)
        
        # Create relative path structure in destination
        rel_path = img_path.relative_to(source_path)
        dest_file = dest_path / rel_path
        
        # Create subdirectories if needed
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed image with maximum quality
        # For PNG: compression level 0 (no compression, max quality)
        # For JPEG: quality 100 (max quality)
        if dest_file.suffix.lower() in ['.png']:
            cv2.imwrite(str(dest_file), processed, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif dest_file.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(dest_file), processed, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            cv2.imwrite(str(dest_file), processed)
        processed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(image_files)}")
    
    # Create filename list
    filename_list_path = os.path.join(dest_dir, "filename_list.txt")
    with open(filename_list_path, 'w') as f:
        for img_path in sorted(dest_path.rglob('*.png')):
            rel_path = img_path.relative_to(dest_path)
            f.write(f"{rel_path}\n")
        for img_path in sorted(dest_path.rglob('*.jpg')):
            rel_path = img_path.relative_to(dest_path)
            f.write(f"{rel_path}\n")
    
    print(f"Filename list saved to: {filename_list_path}")
    print("Preprocess finished")
