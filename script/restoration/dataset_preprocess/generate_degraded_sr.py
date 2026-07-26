# Generate degraded images using super-resolution degradation (downscale-upscale)
# Reads configuration from dataset_base.yaml sr_config section
#
# This script replicates the exact SR degradation logic from base_restoration_dataset.py
# to generate offline datasets with the same degradation distribution as online training.

import argparse
import numpy as np
import os
import random
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from omegaconf import OmegaConf
from src.ARNIQA.degradation import ImageDistorter, downscale_upscale


class SRDegradationGenerator:
    """Generate degraded images using super-resolution degradation from config"""
    
    def __init__(self, config_path: str, scale_factor_override: float = None):
        """Initialize generator with config file
        
        Args:
            config_path: Path to dataset_base.yaml or similar config
            scale_factor_override: If provided, use this fixed scale factor instead of range
        """
        # Load config
        self.cfg = OmegaConf.load(config_path)
        self.degradation_config = self.cfg.degradation_config
        
        # Extract SR config
        self.sr_config = self.degradation_config.get('sr_config', {})
        
        # Scale factor: either fixed override or range from config
        self.scale_factor_override = scale_factor_override
        scale_range = self.sr_config.get('scale_factor', [2.0, 4.0])
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]
        
        # Interpolation modes
        self.downscale_mode = self.sr_config.get('downscale_mode', 'random')
        self.upscale_mode = self.sr_config.get('upscale_mode', 'random')
        
        # Base seed for deterministic generation
        self.base_seed = self.degradation_config.get('base_seed', 42)
        
        # Initialize ARNIQA distorter for image loading/saving
        self.distorter = ImageDistorter()
        
        # Print config summary
        print(f"SR Degradation Generator initialized:")
        print(f"  Base seed: {self.base_seed}")
        if self.scale_factor_override:
            print(f"  Scale factor: {self.scale_factor_override} (fixed override)")
        else:
            print(f"  Scale factor: [{self.scale_min}, {self.scale_max}] (random)")
        print(f"  Downscale mode: {self.downscale_mode}")
        print(f"  Upscale mode: {self.upscale_mode}")
    
    def set_seed(self, index: int, seed_offset: int = 0):
        """Set deterministic seed for given image index
        
        Args:
            index: Image index
            seed_offset: Additional offset (e.g., for different runs)
        """
        seed = self.base_seed + index + seed_offset
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def apply_sr_degradation(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply super-resolution degradation (downscale-upscale)
        
        Replicates _apply_sr_degradation from base_restoration_dataset.py
        
        Args:
            image_tensor: Clean image [C, H, W] in [0, 1]
            
        Returns:
            Degraded image [C, H, W] in [0, 1]
        """
        # Determine scale factor
        if self.scale_factor_override:
            scale_factor = self.scale_factor_override
        else:
            scale_factor = random.uniform(self.scale_min, self.scale_max)
        
        # Apply downscale-upscale degradation
        degraded_tensor = downscale_upscale(
            image_tensor,
            scale_factor=scale_factor,
            downscale_mode=self.downscale_mode,
            upscale_mode=self.upscale_mode
        )
        
        return degraded_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Generate degraded images using SR degradation (downscale-upscale) from config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset_restoration/dataset_base.yaml",
        help="Path to config file with degradation_config.sr_config"
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        required=True,
        help="Directory containing clean images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save degraded images"
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Offset added to seed (for generating different degradation sets)"
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help="Fixed scale factor override (e.g., 4.0). If not provided, samples from config range"
    )
    args = parser.parse_args()
    
    # Initialize generator
    generator = SRDegradationGenerator(args.config, args.scale_factor)
    
    # Find all images
    clean_path = Path(args.clean_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files_set = set()
    for ext in image_extensions:
        image_files_set.update(clean_path.rglob(f'*{ext}'))
        image_files_set.update(clean_path.rglob(f'*{ext.upper()}'))
    image_files = sorted(list(image_files_set))
    
    print(f"Found {len(image_files)} images in {args.clean_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed offset: {args.seed_offset}")
    
    if len(image_files) == 0:
        print("No images found, exiting")
        return
    
    # Process images
    processed = 0
    skipped = 0
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Generating SR degraded images")):
        try:
            # Set deterministic seed
            generator.set_seed(idx, args.seed_offset)
            
            # Load image
            image_tensor = generator.distorter.load_image(str(img_path))
            
            # Apply SR degradation
            degraded_tensor = generator.apply_sr_degradation(image_tensor)
            
            # Create output path (preserve directory structure)
            rel_path = img_path.relative_to(clean_path)
            dest_file = output_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save
            generator.distorter.save_image(degraded_tensor, str(dest_file))
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            skipped += 1
    
    print(f"\nComplete: {processed} processed, {skipped} skipped")


if __name__ == "__main__":
    main()
