# Generate degraded images using pipeline degradation mode
# Reads configuration from dataset_base.yaml and applies sequential degradation:
# Blur → Noise → Compression (each with configurable probability)
#
# This script replicates the exact degradation logic from base_restoration_dataset.py
# to generate offline datasets with the same degradation distribution as online training.

import argparse
import numpy as np
import os
import random
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from omegaconf import OmegaConf
from src.ARNIQA.degradation import ImageDistorter


class PipelineDegradationGenerator:
    """Generate degraded images using pipeline mode from config"""
    
    def __init__(self, config_path: str):
        """Initialize generator with config file
        
        Args:
            config_path: Path to dataset_base.yaml or similar config
        """
        # Load config
        self.cfg = OmegaConf.load(config_path)
        self.degradation_config = self.cfg.degradation_config
        
        # Verify pipeline mode
        mode = self.degradation_config.get('mode', 'pre_generated')
        if mode != 'pipeline':
            print(f"Warning: Config mode is '{mode}', but this script uses pipeline logic")
        
        # Extract pipeline config
        self.pipeline_config = self.degradation_config.get('pipeline_config', {})
        self.blur_config = self.pipeline_config.get('blur', {})
        self.noise_config = self.pipeline_config.get('noise', {})
        self.compression_config = self.pipeline_config.get('compression', {})
        
        # Base seed for deterministic generation
        self.base_seed = self.degradation_config.get('base_seed', 42)
        
        # Initialize ARNIQA distorter
        self.distorter = ImageDistorter()
        
        # Print config summary
        print(f"Pipeline Degradation Generator initialized:")
        print(f"  Base seed: {self.base_seed}")
        print(f"  Blur: p={self.blur_config.get('probability', 0.5)}")
        print(f"  Noise: p={self.noise_config.get('probability', 0.5)}")
        print(f"  Compression: p={self.compression_config.get('probability', 0.5)}")
    
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
    
    def apply_pipeline(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply sequential probabilistic degradation pipeline
        
        Replicates _apply_pipeline_degradation from base_restoration_dataset.py
        
        Args:
            image_tensor: Clean image [C, H, W] in [0, 1]
            
        Returns:
            Degraded image [C, H, W] in [0, 1]
        """
        degraded = image_tensor.clone()
        
        # Stage 1: BLUR
        if random.random() < self.blur_config.get('probability', 0.5):
            degraded = self._apply_blur_stage(degraded)
        
        # Stage 2: NOISE
        if random.random() < self.noise_config.get('probability', 0.5):
            degraded = self._apply_noise_stage(degraded)
        
        # Stage 3: COMPRESSION
        if random.random() < self.compression_config.get('probability', 0.5):
            degraded = self._apply_compression_stage(degraded)
        
        return degraded
    
    def _apply_blur_stage(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply blur degradation (Gaussian or Motion)"""
        blur_types = self.blur_config.get('types', {})
        
        gaussian_weight = blur_types.get('gaussian', {}).get('weight', 0.7)
        motion_weight = blur_types.get('motion', {}).get('weight', 0.3)
        
        if random.random() < gaussian_weight / (gaussian_weight + motion_weight):
            # Gaussian blur
            gaussian_config = blur_types.get('gaussian', {})
            available_levels = gaussian_config.get('levels', [0, 1, 2, 3, 4])
            level = random.choice(available_levels)
            return self.distorter.apply_distortion_to_tensor(image_tensor, 'gaublur', level)
        else:
            # Motion blur
            motion_config = blur_types.get('motion', {})
            available_levels = motion_config.get('levels', [0, 1, 2, 3, 4])
            level = random.choice(available_levels)
            return self.distorter.apply_distortion_to_tensor(image_tensor, 'motionblur', level)
    
    def _apply_noise_stage(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply noise degradation (Gaussian or Multiplicative)"""
        noise_types = self.noise_config.get('types', {})
        
        gaussian_weight = noise_types.get('gaussian', {}).get('weight', 0.8)
        mult_weight = noise_types.get('multiplicative', {}).get('weight', 0.2)
        
        if random.random() < gaussian_weight / (gaussian_weight + mult_weight):
            # Gaussian (white) noise
            gaussian_config = noise_types.get('gaussian', {})
            available_levels = gaussian_config.get('levels', [0, 1, 2, 3, 4])
            level = random.choice(available_levels)
            return self.distorter.apply_distortion_to_tensor(image_tensor, 'whitenoise', level)
        else:
            # Multiplicative noise
            mult_config = noise_types.get('multiplicative', {})
            available_levels = mult_config.get('levels', [0, 1, 2, 3, 4])
            level = random.choice(available_levels)
            return self.distorter.apply_distortion_to_tensor(image_tensor, 'multnoise', level)
    
    def _apply_compression_stage(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply JPEG compression degradation"""
        available_levels = self.compression_config.get('levels', [0, 1, 2, 3, 4])
        level = random.choice(available_levels)
        return self.distorter.apply_distortion_to_tensor(image_tensor, 'jpeg', level)


def main():
    parser = argparse.ArgumentParser(
        description="Generate degraded images using pipeline mode from config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset_restoration/dataset_base.yaml",
        help="Path to config file with degradation_config.pipeline_config"
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
    args = parser.parse_args()
    
    # Initialize generator
    generator = PipelineDegradationGenerator(args.config)
    
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
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Generating degraded images")):
        try:
            # Set deterministic seed
            generator.set_seed(idx, args.seed_offset)
            
            # Load image
            image_tensor = generator.distorter.load_image(str(img_path))
            
            # Apply pipeline degradation
            degraded_tensor = generator.apply_pipeline(image_tensor)
            
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
