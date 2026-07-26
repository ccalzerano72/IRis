#!/usr/bin/env python3
"""
Independent module for applying single distortions to images.

This module allows applying a specific distortion to an image
and saving the result in an organized folder structure.

Usage:
    python distort.py --image path/to/image.jpg --distortion gaublur --level 2
    
    or as a module:
    from distort import ImageDistorter
    distorter = ImageDistorter()
    distorter.apply_distortion("path/to/image.jpg", "gaublur", 2)
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import sys
import os
from typing import Dict, List, Tuple, Union

import torch.nn.functional as F
import random as py_random
from torchvision.transforms import GaussianBlur

from .utils.distortions import (
    gaussian_blur, lens_blur, motion_blur, color_diffusion, color_shift,
    color_saturation1, color_saturation2, jpeg2000, jpeg, white_noise,
    white_noise_cc, impulse_noise, multiplicative_noise, brighten, darken,
    mean_shift, jitter, non_eccentricity_patch, pixelate, quantization,
    color_block, high_sharpen, linear_contrast_change, non_linear_contrast_change
)


def downscale_upscale(image_tensor: torch.Tensor, scale_factor: float, 
                      downscale_mode: str = "random", upscale_mode: str = "random") -> torch.Tensor:
    """
    Super-resolution degradation: downscale then upscale back to original size.
    
    This simulates the information loss that occurs when a low-resolution image
    is upscaled, creating the typical SR artifacts (blurriness, aliasing).
    
    Args:
        image_tensor: Input image tensor [C, H, W] in [0, 1]
        scale_factor: Downscale factor (e.g., 2.0 means reduce to 1/2 size)
        downscale_mode: Downscale interpolation mode. 
                        "random" (default) -> randomly choose between 'area' and 'bicubic'
                        "area" -> use area interpolation
                        "bicubic" -> use bicubic interpolation
        upscale_mode: Upscale interpolation mode.
                      "random" (default) -> randomly choose between 'bicubic' and 'bilinear'
                      "bicubic" -> use bicubic interpolation
                      "bilinear" -> use bilinear interpolation
    
    Returns:
        Degraded image tensor [C, H, W] in [0, 1] at original resolution
    """
    if scale_factor <= 1.0:
        return image_tensor
    
    # Optional light blur before downscaling (80% probability)
    if py_random.random() < 0.8:
        k_size = py_random.choice([3, 5])
        sigma = py_random.uniform(0.2, 0.7)
        image_tensor = GaussianBlur(kernel_size=k_size, sigma=sigma)(image_tensor)
    
    # Get original size
    c, h, w = image_tensor.shape
    
    # Calculate downscaled size
    new_h = max(1, int(h / scale_factor))
    new_w = max(1, int(w / scale_factor))
    
    # Add batch dimension for F.interpolate: [C, H, W] -> [1, C, H, W]
    img_4d = image_tensor.unsqueeze(0)
    
    # Choose downscale mode randomly if "random"
    if downscale_mode == "random" or downscale_mode is None:
        downscale_mode = py_random.choice(['area', 'bicubic'])
    
    # Downscale
    if downscale_mode == 'area':
        downscaled = F.interpolate(
            img_4d, 
            size=(new_h, new_w), 
            mode='area'
        )
    else:  # bicubic
        downscaled = F.interpolate(
            img_4d, 
            size=(new_h, new_w), 
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
    
    # Choose upscale mode randomly if "random"
    if upscale_mode == "random" or upscale_mode is None:
        upscale_mode = py_random.choice(['bicubic', 'bilinear'])
    
    # Upscale back to original size
    upscaled = F.interpolate(
        downscaled,
        size=(h, w),
        mode=upscale_mode,
        align_corners=False if upscale_mode in ['bicubic', 'bilinear'] else None,
        antialias=True
    )
    
    # Remove batch dimension: [1, C, H, W] -> [C, H, W]
    result = upscaled[0]
    
    # Add fine grain noise (50% probability) to help UNet generate texture
    if py_random.random() < 0.5:
        strength = py_random.uniform(0.005, 0.015)
        noise = torch.randn_like(result) * strength
        result = result + noise
    
    # Clamp to valid range
    return torch.clamp(result, 0.0, 1.0)


class ImageDistorter:
    """
    A class for applying various distortions to images.
    
    This class provides methods to apply single distortions or generate
    all possible distortions for a given image.
    """
    
    def __init__(self):
        """Initialize the ImageDistorter with available distortions and levels."""
        self._initialize_distortions()
    
    def _initialize_distortions(self):
        """Initialize distortion functions and their parameter levels."""

        self.distortion_functions = {
            "gaublur": gaussian_blur,
            "lensblur": lens_blur,
            "motionblur": motion_blur,
            "colordiff": color_diffusion,
            "colorshift": color_shift,
            "colorsat1": color_saturation1,
            "colorsat2": color_saturation2,
            "jpeg2000": jpeg2000,
            "jpeg": jpeg,
            "whitenoise": white_noise,
            "whitenoiseCC": white_noise_cc,
            "impulsenoise": impulse_noise,
            "multnoise": multiplicative_noise,
            "brighten": brighten,
            "darken": darken,
            "meanshift": mean_shift,
            "jitter": jitter,
            "noneccpatch": non_eccentricity_patch,
            "pixelate": pixelate,
            "quantization": quantization,
            "colorblock": color_block,
            "highsharpen": high_sharpen,
            "lincontrchange": linear_contrast_change,
            "nonlincontrchange": non_linear_contrast_change,
            "downscale_upscale": downscale_upscale,  # Super-resolution degradation
        }
        
        # Parameters for each distortion level (0-4)
        self.distortion_levels = {
            "gaublur": [0.1, 0.5, 1, 2, 5],
            "lensblur": [1, 2, 4, 6, 8],
            "motionblur": [1, 2, 4, 6, 10],
            "colordiff": [1, 3, 6, 8, 12],
            "colorshift": [1, 3, 6, 8, 12],
            "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
            "colorsat2": [1, 2, 3, 6, 9],
            "jpeg2000": [16, 32, 45, 120, 170],
            "jpeg": [43, 36, 24, 7, 4],
            "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
            "whitenoiseCC": [0.0001, 0.0005, 0.001, 0.002, 0.003],
            "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],
            "multnoise": [0.001, 0.005, 0.01, 0.02, 0.05],
            "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
            "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
            "meanshift": [0, 0.08, -0.08, 0.15, -0.15],
            "jitter": [0.05, 0.1, 0.2, 0.5, 1],
            "noneccpatch": [20, 40, 60, 80, 100],
            "pixelate": [0.01, 0.05, 0.1, 0.2, 0.5],
            "quantization": [20, 16, 13, 10, 7],
            "colorblock": [2, 4, 6, 8, 10],
            "highsharpen": [1, 2, 3, 6, 12],
            "lincontrchange": [0., 0.15, -0.4, 0.3, -0.6],
            "nonlincontrchange": [0.4, 0.3, 0.2, 0.1, 0.05],
            "downscale_upscale": (2.0, 4.0),  # Continuous range (min, max) for SR
        }

    
    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load an image and convert it to a tensor.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            torch.Tensor: Image tensor in CHW format, normalized to [0, 1]
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be loaded
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
            # print(f"Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
        
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return image_tensor
    
    def save_image(self, image_tensor: torch.Tensor, output_path: Union[str, Path]) -> str:
        """
        Save a tensor as an image file.
        
        Args:
            image_tensor: Image tensor in CHW format, values in [0, 1]
            output_path: Path where to save the image
            
        Returns:
            str: Path of the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensor to PIL image
        image_tensor = torch.clamp(image_tensor, 0, 1)
        distorted_array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        distorted_image = Image.fromarray(distorted_array)
        
        # Save the image
        distorted_image.save(output_path, "PNG")
        # print(f"Image saved: {output_path}")
        
        return str(output_path)
    
    def apply_distortion_to_tensor(self, image_tensor: torch.Tensor, 
                                 distortion_name: str, level: int) -> torch.Tensor:
        """
        Apply a specific distortion to an image tensor.
        
        Args:
            image_tensor: Input image tensor in CHW format
            distortion_name: Name of the distortion to apply
            level: Distortion level (0-4)
            
        Returns:
            torch.Tensor: Distorted image tensor
            
        Raises:
            ValueError: If distortion name or level is invalid
            RuntimeError: If distortion application fails
        """

        if distortion_name not in self.distortion_functions:
            available = ", ".join(self.distortion_functions.keys())
            raise ValueError(f"Distortion '{distortion_name}' not supported. Available: {available}")
        
        if not 0 <= level <= 4:
            raise ValueError(f"Level must be between 0 and 4, received: {level}")
        

        distortion_func = self.distortion_functions[distortion_name]
        level_config = self.distortion_levels[distortion_name]
        
        # Handle continuous range (tuple) vs discrete levels (list)
        if isinstance(level_config, tuple):
            # Continuous range: sample uniformly between min and max
            min_val, max_val = level_config
            distortion_param = py_random.uniform(min_val, max_val)
        else:
            # Discrete levels
            distortion_param = level_config[level]
        
        # print(f"Applying distortion '{distortion_name}' level {level} (parameter: {distortion_param})")
        
        try:
        	# tensor clone needed because some distortion filters operate in-place
            distorted_tensor = distortion_func(image_tensor.clone(), distortion_param)
            # print(distortion_name, level, distorted_tensor.min().item(), distorted_tensor.max().item())
            
            # Ensure tensor is in range [0, 1]
            distorted_tensor = torch.clamp(distorted_tensor, 0, 1)
            return distorted_tensor
            
        except Exception as e:
            raise RuntimeError(f"Error applying distortion: {e}")
    
    def apply_distortion(self, image_path: str, distortion_name: str, level: int, 
                        output_dir: str = "./distortions") -> str:
        """
        Apply a specific distortion to an image and save the result.
        
        Args:
            image_path: Path to the input image
            distortion_name: Name of the distortion to apply
            level: Distortion level (0-4)
            output_dir: Base output directory
            
        Returns:
            str: Path of the saved output file
        """
        image_tensor = self.load_image(image_path)
        distorted_tensor = self.apply_distortion_to_tensor(image_tensor, distortion_name, level)
        
        image_path = Path(image_path)
        image_name = image_path.stem
        output_path = Path(output_dir)
        output_filename = f"{image_name}.{distortion_name}.{level}.png"
        output_filepath = output_path / output_filename
        
        return self.save_image(distorted_tensor, output_filepath)
    
    def generate_all_distortions(self, image_path: str, output_dir: str = "./distortions") -> List[str]:
        """
        Generate all possible distortions and levels for a given image.
        
        Args:
            image_path: Path to the input image
            output_dir: Base output directory
            
        Returns:
            List[str]: List of paths of all generated distorted images
        """
        # Load the original image once
        image_tensor = self.load_image(image_path)
        
        image_path = Path(image_path)
        image_name = image_path.stem
        output_base_path = Path(output_dir)
        
        generated_files = []
        total_distortions = len(self.distortion_functions) * 5  # 5 levels per distortion
        current_count = 0
        
        print(f"Generating {total_distortions} distorted images for '{image_name}'...")
        
        # Iterate through all distortions and levels
        for distortion_name in self.distortion_functions.keys():
            for level in range(5):  # Levels 0-4
                try:
                    distorted_tensor = self.apply_distortion_to_tensor(
                        image_tensor, distortion_name, level
                    )
                    
                    # print(distortion_name, level, distorted_tensor.min().item(), distorted_tensor.max().item())

                    # Create output filename: imagename.distortiontype.level.png
                    output_filename = f"{image_name}.{distortion_name}.{level}.jpg"
                    output_filepath = output_base_path / output_filename
                    
                    saved_path = self.save_image(distorted_tensor, output_filepath)
                    generated_files.append(saved_path)
                    
                    current_count += 1
                    if current_count % 10 == 0:  # Progress update every 10 images
                        print(f"Progress: {current_count}/{total_distortions} images generated")
                    
                except Exception as e:
                    print(f"Warning: Failed to generate {distortion_name} level {level}: {e}")
                    continue
        
        print(f"✓ Generated {len(generated_files)} distorted images in '{output_dir}'")
        return generated_files
    
    def process_image_directory(self, image_dir: str, distortion_name: str = None, 
                              level: int = None, output_dir: str = "./distortions", 
                              all_distortions: bool = False) -> List[str]:
        """
        Process all JPEG/PNG images in a directory with specified distortion(s).
        
        Args:
            image_dir: Path to directory containing images
            distortion_name: Name of distortion to apply (if not all_distortions)
            level: Distortion level (if not all_distortions)
            output_dir: Base output directory
            all_distortions: If True, generate all distortions for each image
            
        Returns:
            List[str]: List of paths of all generated distorted images
        """
        image_dir = Path(image_dir)
        if not image_dir.exists() or not image_dir.is_dir():
            raise ValueError(f"Directory not found or not a directory: {image_dir}")
        
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No JPEG/PNG images found in directory: {image_dir}")
            return []
        
        print(f"Found {len(image_files)} images in directory: {image_dir}")
        
        all_generated_files = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\nProcessing image {i}/{len(image_files)}: {image_file.name}")
            
            try:
                if all_distortions:
                    # Generate all distortions for this image
                    generated_files = self.generate_all_distortions(str(image_file), output_dir)
                    all_generated_files.extend(generated_files)
                else:
                    # Apply single distortion to this image
                    output_path = self.apply_distortion(str(image_file), distortion_name, level, output_dir)
                    all_generated_files.append(output_path)
                    
            except Exception as e:
                print(f"Warning: Failed to process {image_file.name}: {e}")
                continue
        
        print(f"\n✓ Directory processing completed!")
        print(f"  Processed: {len(image_files)} input images")
        print(f"  Generated: {len(all_generated_files)} distorted images")
        print(f"  Output directory: {output_dir}")
        
        return all_generated_files


def main():
    """Main function for command line usage."""
    distorter = ImageDistorter()
    
    parser = argparse.ArgumentParser(
        description="Apply single distortions to images using ARNIQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available distortions:
{', '.join(sorted(distorter.distortion_functions.keys()))}

Examples:
  # Single image processing
  python distort.py --image photo.jpg --distortion gaublur --level 2
  python distort.py --image /path/to/image.png --distortion brighten --level 3 --output ./my_distortions
  python distort.py --image photo.jpg --all --output ./all_distortions
  
  # Directory processing
  python distort.py --imagedir ./photos --distortion gaublur --level 2
  python distort.py --imagedir /path/to/images --all --output ./batch_distortions
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", "-i", 
                           help="Path to a single input image")
    input_group.add_argument("--imagedir", 
                           help="Path to directory containing images (JPEG/PNG)")
    
    # Distortion parameters
    parser.add_argument("--distortion", "-d",
                       choices=list(distorter.distortion_functions.keys()),
                       help="Type of distortion to apply")
    parser.add_argument("--level", "-l", type=int,
                       choices=range(5), metavar="0-4",
                       help="Distortion level (0=minimum, 4=maximum)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Generate all possible distortions and levels")
    parser.add_argument("--output", "-o", default="./distortions",
                       help="Output directory (default: ./distortions)")
    
    args = parser.parse_args()
    
    # Validate distortion parameters
    if not args.all and (args.distortion is None or args.level is None):
        parser.error("Either use --all or specify both --distortion and --level")
    
    if args.all and (args.distortion is not None or args.level is not None):
        parser.error("Cannot use --all with --distortion or --level")
    
    try:
        if args.imagedir:
            # Process directory of images
            if args.all:
                generated_files = distorter.process_image_directory(
                    args.imagedir, output_dir=args.output, all_distortions=True
                )
                print(f"\n✓ All distortions generated for directory!")
                print(f"  Input:  {args.imagedir}")
                print(f"  Output: {args.output}")
                print(f"  Files:  {len(generated_files)} images generated")
            else:
                generated_files = distorter.process_image_directory(
                    args.imagedir, args.distortion, args.level, args.output
                )
                print(f"\n✓ Distortion applied to directory!")
                print(f"  Input:  {args.imagedir}")
                print(f"  Distortion: {args.distortion} level {args.level}")
                print(f"  Output: {args.output}")
                print(f"  Files:  {len(generated_files)} images generated")
        else:
            # Process single image
            if args.all:
                generated_files = distorter.generate_all_distortions(args.image, args.output)
                print(f"\n✓ All distortions generated successfully!")
                print(f"  Input:  {args.image}")
                print(f"  Output: {args.output}")
                print(f"  Files:  {len(generated_files)} images generated")
            else:
                output_path = distorter.apply_distortion(args.image, args.distortion, args.level, args.output)
                print(f"\n✓ Distortion applied successfully!")
                print(f"  Input:  {args.image}")
                print(f"  Output: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()