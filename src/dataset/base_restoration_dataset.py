# BaseRestorationDataset - Thesis Implementation
# 
# Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction:
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# Base dataset class for image restoration tasks.
# Handles loading of clean and degraded image pairs.
# --------------------------------------------------------------------------

import ctypes
import gc
import io
import numpy as np
import os
import random
import tarfile
import torch
from PIL import Image
from enum import Enum
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
from typing import Union, Dict, Any
from pathlib import Path


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


class BaseRestorationDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        clean_dir: str,
        degraded_dir: str,
        disp_name: str,
        degradation_config: Dict[str, Any] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  # [0, 255] -> [-1, 1]
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        
        # Resolve paths with BASE_DATA_DIR
        self.filename_ls_path = self._resolve_path(filename_ls_path)
        self.clean_dir = self._resolve_path(clean_dir)
        self.degraded_dir = self._resolve_path(degraded_dir)
        self.disp_name = disp_name
        
        # Degradation config
        self.degradation_config = degradation_config or {}
        self.degradation_mode = self.degradation_config.get('mode', 'pre_generated')
        
        # Deterministic degradation seeds
        self.degradation_base_seed = self.degradation_config.get('base_seed', 42)
        self.current_epoch = 0  # Will be updated by trainer for TRAIN mode
        
        # Verify directories exist based on mode
        assert os.path.exists(self.clean_dir), f"Clean directory does not exist: {self.clean_dir}"
        if self.degradation_mode == 'pre_generated':
            assert os.path.exists(self.degraded_dir), f"Degraded directory does not exist: {self.degraded_dir}"
        
        # Training arguments
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform
        
        # Initialize online degradation if needed
        if self.degradation_mode in ['online', 'pipeline', 'sr']:
            self._init_online_degradation()
        elif self.degradation_mode in ['diffbir_codeformer', 'diffbir_realesrgan']:
            self._init_diffbir_degradation()
        else:
            self.distorter = None
        
        # Load filenames
        self._load_filenames()
        
        # Tar dataset support (for consistency with Marigold)
        self.tar_obj = None
        self.is_tar_clean = self._check_tar(self.clean_dir)
        self.is_tar_degraded = self._check_tar(self.degraded_dir) if self.degradation_mode == 'pre_generated' else False

    def _resolve_path(self, path):
        """Resolve path with BASE_DATA_DIR if path is relative"""
        if os.path.isabs(path):
            return path
        
        # Special handling for data_split paths (should remain relative to project root)
        if path.startswith('data_split/'):
            return path
        
        # For dataset directories, use BASE_DATA_DIR
        base_data_dir = os.environ.get('BASE_DATA_DIR')
        if base_data_dir:
            return os.path.join(base_data_dir, path)
        
        # Fallback to ./data for dataset directories
        return os.path.join('data', path)

    def _check_tar(self, directory):
        """Check if directory is a tar file"""
        return (
            os.path.isfile(directory) and tarfile.is_tarfile(directory)
            if directory else False
        )
    
    def _init_online_degradation(self):
        """Initialize ARNIQA degradation system for online/pipeline/sr mode"""
        try:
            from src.ARNIQA.degradation import ImageDistorter, downscale_upscale
            self.distorter = ImageDistorter()
            self.downscale_upscale_fn = downscale_upscale
            
            if self.degradation_mode == 'sr':
                # Super-resolution mode: downscale-upscale degradation
                self.sr_config = self.degradation_config.get('sr_config', {})
                
                # Scale factor range (uniform sampling between min and max)
                scale_range = self.sr_config.get('scale_factor', [2.0, 4.0])
                self.sr_scale_min = scale_range[0]
                self.sr_scale_max = scale_range[1]
                
                # Interpolation modes
                self.sr_downscale_mode = self.sr_config.get('downscale_mode', 'random')
                self.sr_upscale_mode = self.sr_config.get('upscale_mode', 'random')
                
                print(f"[SR Degradation] Initialized super-resolution degradation:")
                print(f"  - Scale factor: {self.sr_scale_min}x - {self.sr_scale_max}x (uniform)")
                print(f"  - Downscale mode: {self.sr_downscale_mode}")
                print(f"  - Upscale mode: {self.sr_upscale_mode}")
                
            elif self.degradation_mode == 'pipeline':
                # Pipeline mode: Sequential probabilistic degradation
                self.pipeline_config = self.degradation_config.get('pipeline_config', {})
                
                # Extract pipeline configuration
                self.blur_config = self.pipeline_config.get('blur', {})
                self.noise_config = self.pipeline_config.get('noise', {})
                self.compression_config = self.pipeline_config.get('compression', {})
                
                print(f"[Pipeline Degradation] Initialized sequential pipeline:")
                print(f"  - Blur: p={self.blur_config.get('probability', 0.5)}")
                print(f"  - Noise: p={self.noise_config.get('probability', 0.5)}")
                print(f"  - Compression: p={self.compression_config.get('probability', 0.5)}")
                print(f"  - Expected clean: ~12.5%")
                
            else:
                # Legacy online mode: Simple single/mixed degradation
                self.degradation_types = self.degradation_config.get('types', ['whitenoise'])
                self.degradation_levels = self.degradation_config.get('levels', [2])
                self.mixed_prob = self.degradation_config.get('mixed_prob', 0.0)
                self.degradation_weights = self.degradation_config.get('weights', None)
                
                # Validate degradation types
                available_types = list(self.distorter.distortion_functions.keys())
                for deg_type in self.degradation_types:
                    if deg_type not in available_types:
                        raise ValueError(f"Unknown degradation type: {deg_type}. Available: {available_types}")
                
                print(f"[Online Degradation] Initialized with {len(self.degradation_types)} types, "
                      f"{len(self.degradation_levels)} levels, mixed_prob={self.mixed_prob}")
            
        except ImportError as e:
            raise ImportError(f"Failed to import ARNIQA for online degradation: {e}")

    def _init_diffbir_degradation(self):
        """Initialize DiffBIR degradation system for diffbir_codeformer/diffbir_realesrgan mode.

        Imports degradation functions from generate_degraded_diffbir.py which wraps
        DiffBIR's degradation pipeline (diffbir/dataset/degradation.py, batch_transform.py).
        """
        try:
            from script.restoration.dataset_preprocess.generate_degraded_diffbir import (
                apply_codeformer_degradation,
                generate_realesrgan_kernels,
                apply_realesrgan_degradation,
            )
            from diffbir.dataset.diffjpeg import DiffJPEG

            self.diffbir_apply_codeformer = apply_codeformer_degradation
            self.diffbir_generate_kernels = generate_realesrgan_kernels
            self.diffbir_apply_realesrgan = apply_realesrgan_degradation

            # Read config sections (with DiffBIR v2.1 defaults)
            self.diffbir_cf_config = self.degradation_config.get('diffbir_codeformer_config', {})
            self.diffbir_re_config = self.degradation_config.get('diffbir_realesrgan_config', {})

            if self.degradation_mode == 'diffbir_realesrgan':
                # DiffJPEG on CPU (DataLoader workers cannot share CUDA context)
                self.diffbir_jpeger = DiffJPEG(differentiable=False)
                # Pre-create reusable pulse tensor (avoids torch.zeros(21,21) every call)
                self.diffbir_pulse_tensor = torch.zeros(21, 21).float()
                self.diffbir_pulse_tensor[10, 10] = 1
                # glibc malloc_trim handle for forcing memory return to OS
                try:
                    self._libc = ctypes.CDLL('libc.so.6')
                except OSError:
                    self._libc = None
                stage2_scale = self.diffbir_re_config.get('stage2_scale', 1.0)
                print(f"[DiffBIR RealESRGAN Degradation] Initialized two-stage pipeline:")
                print(f"  - stage2_scale: {stage2_scale}")
                print(f"  - Device: cpu (DataLoader compatible)")
                print(f"  - Memory management: numpy round-trip + gc + malloc_trim")
            else:
                self.diffbir_jpeger = None
                self.diffbir_pulse_tensor = None
                self._libc = None
                print(f"[DiffBIR Codeformer Degradation] Initialized single-stage pipeline")

        except ImportError as e:
            raise ImportError(f"Failed to import DiffBIR degradation modules: {e}")

    def _load_filenames(self):
        """Load filename pairs from file list"""
        assert os.path.exists(self.filename_ls_path), f"File list does not exist: {self.filename_ls_path}"
        
        with open(self.filename_ls_path, "r") as f:
            lines = f.readlines()
        
        self.filenames = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                # Format: clean_path degraded_path
                clean_rel_path = parts[0]
                degraded_rel_path = parts[1]
                self.filenames.append([clean_rel_path, degraded_rel_path])
            elif len(parts) == 1:
                # Format: same_path (clean and degraded have same relative path)
                rel_path = parts[0]
                self.filenames.append([rel_path, rel_path])
            else:
                raise ValueError(f"Invalid line format in {self.filename_ls_path}: {line}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Set deterministic seed for degradation generation
        self._set_degradation_seed(index)
        
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # Merge
        outputs = rasters
        outputs.update(other)
        return outputs
    
    def _set_degradation_seed(self, index):
        """Set deterministic seed for degradation generation
        
        For EVAL/RGB_ONLY: seed = base_seed + index (always same degradation)
        For TRAIN: seed = base_seed + index + epoch * 1000000 (different each epoch)
        """
        if self.degradation_mode not in ['online', 'pipeline', 'sr', 'diffbir_codeformer', 'diffbir_realesrgan']:
            return  # No need for seed if using pre-generated degradations
        
        if self.mode in [DatasetMode.EVAL, DatasetMode.RGB_ONLY]:
            # EVAL: Always same degradation for same image
            seed = self.degradation_base_seed + index
        else:
            # TRAIN: Different degradation each epoch, but deterministic
            seed = self.degradation_base_seed + index + (self.current_epoch * 1000000)
        
        # Set all random seeds for degradation generation
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def set_epoch(self, epoch: int):
        """Update current epoch for deterministic-variable degradations
        
        Called by trainer at the start of each epoch for TRAIN mode.
        """
        self.current_epoch = epoch

    def _get_data_item(self, index):
        """Get data item for given index"""
        clean_rel_path, degraded_rel_path = self._get_data_paths(index)
        
        rasters = {}
        
        # Load clean RGB data
        clean_data = self._load_rgb_data(clean_rel_path, self.clean_dir, self.is_tar_clean, "clean")
        
        # For online/pipeline/sr/diffbir modes: resize BEFORE degradation generation
        # This ensures consistent dimensions and is more efficient
        if self.degradation_mode in ['online', 'pipeline', 'sr', 'diffbir_codeformer', 'diffbir_realesrgan'] and self.resize_to_hw is not None:
            clean_data = self._resize_data(clean_data)
        
        rasters.update(clean_data)
        
        # Load or generate degraded RGB data based on mode
        if self.degradation_mode == 'online':
            # Generate degradation online from clean image (already resized)
            degraded_data = self._generate_degraded_online(clean_data)
            degraded_rel_path = f"{clean_rel_path}_online"  # Virtual path for tracking
        elif self.degradation_mode == 'pipeline':
            # Generate degradation using realistic pipeline from clean image (already resized)
            degraded_data = self._generate_degraded_online(clean_data)
            degraded_rel_path = f"{clean_rel_path}_pipeline"  # Virtual path for tracking
        elif self.degradation_mode == 'sr':
            # Generate SR degradation (downscale-upscale) from clean image (already resized)
            degraded_data = self._generate_degraded_online(clean_data)
            degraded_rel_path = f"{clean_rel_path}_sr"  # Virtual path for tracking
        elif self.degradation_mode == 'diffbir_codeformer':
            # Generate DiffBIR Codeformer degradation from clean image (already resized)
            degraded_data = self._generate_degraded_online(clean_data)
            degraded_rel_path = f"{clean_rel_path}_diffbir_cf"  # Virtual path for tracking
        elif self.degradation_mode == 'diffbir_realesrgan':
            # Generate DiffBIR RealESRGAN degradation from clean image (already resized)
            degraded_data = self._generate_degraded_online(clean_data)
            degraded_rel_path = f"{clean_rel_path}_diffbir_re"  # Virtual path for tracking
        else:
            # Load pre-generated degraded image (original behavior)
            degraded_data = self._load_rgb_data(degraded_rel_path, self.degraded_dir, self.is_tar_degraded, "degraded")
        
        rasters.update(degraded_data)
        
        other = {
            "index": index,
            "rgb_relative_path": clean_rel_path,
            "clean_relative_path": clean_rel_path,
            "degraded_relative_path": degraded_rel_path,
        }
        
        return rasters, other

    def _get_data_paths(self, index):
        """Get clean and degraded paths for given index"""
        filename_line = self.filenames[index]
        clean_rel_path = filename_line[0]
        degraded_rel_path = filename_line[1]
        return clean_rel_path, degraded_rel_path

    def _load_rgb_data(self, rel_path, base_dir, is_tar, prefix):
        """Load RGB data from file"""
        # Read RGB data
        rgb = self._read_rgb_file(rel_path, base_dir, is_tar)  # [H, W, C]
        
        # Transpose in numpy (creates a copy via astype), not in PyTorch
        # This avoids "storage not resizable" errors in DataLoader workers
        # Pattern from base_depth_dataset.py line 209
        rgb_chw = np.transpose(rgb, (2, 0, 1)).astype(int)  # [H, W, C] -> [C, H, W]
        rgb_norm = rgb_chw / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        
        outputs = {
            f"{prefix}_rgb_int": torch.from_numpy(rgb_chw).int(),
            f"{prefix}_rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _read_rgb_file(self, rel_path, base_dir, is_tar):
        """Read RGB image file"""
        if is_tar:
            return self._read_image_from_tar(rel_path, base_dir)
        else:
            return self._read_image_from_dir(rel_path, base_dir)

    def _read_image_from_tar(self, rel_path, tar_path):
        """Read image from tar file"""
        if self.tar_obj is None:
            self.tar_obj = tarfile.open(tar_path)
        
        image_to_read = self.tar_obj.extractfile("./" + rel_path)
        image_to_read = image_to_read.read()
        image_to_read = io.BytesIO(image_to_read)
        
        image = Image.open(image_to_read)
        image = image.convert("RGB")
        rgb = np.array(image)
        
        return rgb

    def _read_image_from_dir(self, rel_path, base_dir):
        """Read image from directory"""
        img_path = os.path.join(base_dir, rel_path)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path)
        image = image.convert("RGB")
        rgb = np.array(image)
        
        return rgb

    def _generate_degraded_online(self, clean_data):
        """Generate degraded image online using ARNIQA
        
        Args:
            clean_data: Dictionary with clean_rgb_int and clean_rgb_norm tensors
            
        Returns:
            Dictionary with degraded_rgb_int and degraded_rgb_norm tensors
        """
        # Get clean image in [0, 1] format for ARNIQA
        clean_rgb_int = clean_data['clean_rgb_int']  # [C, H, W] in [0, 255]
        clean_tensor = clean_rgb_int.float() / 255.0  # [C, H, W] in [0, 1]
        
        # Apply degradation based on mode
        if self.degradation_mode == 'sr':
            # Super-resolution degradation (downscale-upscale)
            degraded_tensor = self._apply_sr_degradation(clean_tensor)
        elif self.degradation_mode == 'pipeline':
            # Sequential probabilistic pipeline (blur → noise → compression)
            degraded_tensor = self._apply_pipeline_degradation(clean_tensor)
        elif self.degradation_mode == 'diffbir_codeformer':
            # DiffBIR Codeformer single-stage degradation (numpy-based)
            degraded_tensor = self._apply_diffbir_codeformer_degradation(clean_tensor)
        elif self.degradation_mode == 'diffbir_realesrgan':
            # DiffBIR RealESRGAN two-stage degradation (PyTorch-based)
            degraded_tensor = self._apply_diffbir_realesrgan_degradation(clean_tensor)
        else:
            # Legacy mode: single or mixed degradation
            if random.random() < self.mixed_prob:
                degraded_tensor = self._apply_mixed_degradation(clean_tensor)
            else:
                degraded_tensor = self._apply_single_degradation(clean_tensor)
        
        # Convert back to [0, 255] and [-1, 1] formats
        degraded_rgb_int = (degraded_tensor * 255.0).to(torch.int32)
        degraded_rgb_norm = degraded_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        return {
            'degraded_rgb_int': degraded_rgb_int,
            'degraded_rgb_norm': degraded_rgb_norm,
        }
    
    def _apply_single_degradation(self, clean_tensor):
        """Apply a single random degradation"""
        # Select degradation type
        if self.degradation_weights:
            # Weighted sampling
            types = list(self.degradation_weights.keys())
            weights = list(self.degradation_weights.values())
            deg_type = random.choices(types, weights=weights, k=1)[0]
        else:
            # Uniform sampling
            deg_type = random.choice(self.degradation_types)
        
        # Select degradation level
        deg_level = random.choice(self.degradation_levels)
        
        # Apply degradation
        degraded_tensor = self.distorter.apply_distortion_to_tensor(
            clean_tensor, deg_type, deg_level
        )
        
        return degraded_tensor
    
    def _apply_mixed_degradation(self, clean_tensor):
        """Apply multiple degradations sequentially"""
        num_degradations = random.randint(2, 3)  # 2-3 degradations
        
        # Select random degradation types (without replacement)
        if len(self.degradation_types) < num_degradations:
            num_degradations = len(self.degradation_types)
        
        selected_types = random.sample(self.degradation_types, num_degradations)
        
        # Apply degradations sequentially
        degraded_tensor = clean_tensor.clone()
        for deg_type in selected_types:
            deg_level = random.choice(self.degradation_levels)
            degraded_tensor = self.distorter.apply_distortion_to_tensor(
                degraded_tensor, deg_type, deg_level
            )
        
        return degraded_tensor
    
    def _apply_sr_degradation(self, clean_tensor):
        """Apply super-resolution degradation (downscale-upscale)
        
        Simulates low-resolution image upscaling artifacts:
        - Downscales image by random factor in configured range
        - Upscales back to original resolution
        - Creates typical SR artifacts (blurriness, aliasing, loss of detail)
        
        Args:
            clean_tensor: Clean image tensor [C, H, W] in [0, 1]
            
        Returns:
            Degraded image tensor [C, H, W] in [0, 1]
        """
        # Sample scale factor uniformly from configured range
        scale_factor = random.uniform(self.sr_scale_min, self.sr_scale_max)
        
        # Apply downscale-upscale degradation
        degraded_tensor = self.downscale_upscale_fn(
            clean_tensor,
            scale_factor=scale_factor,
            downscale_mode=self.sr_downscale_mode,
            upscale_mode=self.sr_upscale_mode
        )
        
        return degraded_tensor
    
    def _apply_pipeline_degradation(self, clean_tensor):
        """Apply sequential probabilistic degradation pipeline
        
        Following Real-ESRGAN/BSRGAN approach:
        1. Blur (50% probability) - Simulates optical imperfections
        2. Noise (50% probability) - Simulates sensor noise
        3. Compression (50% probability) - Simulates JPEG artifacts
        
        This creates realistic degradation combinations:
        - 12.5% get all 3 (hardest samples)
        - 37.5% get 2 degradations
        - 37.5% get 1 degradation
        - 12.5% stay clean (identity mapping)
        
        Args:
            clean_tensor: Clean image tensor [C, H, W] in [0, 1]
            
        Returns:
            Degraded image tensor [C, H, W] in [0, 1]
        """
        degraded = clean_tensor.clone()
        
        # Stage 1: BLUR (applied first, simulates optical physics)
        if random.random() < self.blur_config.get('probability', 0.5):
            degraded = self._apply_blur_stage(degraded)
        
        # Stage 2: NOISE (applied second, simulates sensor)
        if random.random() < self.noise_config.get('probability', 0.5):
            degraded = self._apply_noise_stage(degraded)
        
        # Stage 3: COMPRESSION (applied last, simulates digital processing)
        if random.random() < self.compression_config.get('probability', 0.5):
            degraded = self._apply_compression_stage(degraded)
        
        return degraded
    
    def _apply_blur_stage(self, image_tensor):
        """Apply blur degradation (Gaussian or Motion)
        
        Uses ARNIQA's predefined levels directly:
        - Gaussian: [0.1, 0.5, 1, 2, 5] (sigma)
        - Motion: [1, 2, 4, 6, 10] (radius)
        """
        blur_types = self.blur_config.get('types', {})
        
        # Select blur type based on weights
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
    
    def _apply_noise_stage(self, image_tensor):
        """Apply noise degradation (Gaussian or Multiplicative)
        
        Uses ARNIQA's predefined levels directly:
        - Gaussian (white): [0.001, 0.002, 0.003, 0.005, 0.01] (variance)
        - Multiplicative: [0.001, 0.005, 0.01, 0.02, 0.05] (variance)
        """
        noise_types = self.noise_config.get('types', {})
        
        # Select noise type based on weights
        gaussian_weight = noise_types.get('gaussian', {}).get('weight', 0.8)
        mult_weight = noise_types.get('multiplicative', {}).get('weight', 0.2)
        
        if random.random() < gaussian_weight / (gaussian_weight + mult_weight):
            # Gaussian (white) noise
            gaussian_config = noise_types.get('gaussian', {})
            available_levels = gaussian_config.get('levels', [0, 1, 2, 3, 4])
            level = random.choice(available_levels)
            
            return self.distorter.apply_distortion_to_tensor(image_tensor, 'whitenoise', level)
        else:
            # Multiplicative noise (intensity-dependent, realistic for low-light)
            mult_config = noise_types.get('multiplicative', {})
            available_levels = mult_config.get('levels', [0, 1, 2, 3, 4])
            level = random.choice(available_levels)
            
            return self.distorter.apply_distortion_to_tensor(image_tensor, 'multnoise', level)
    
    def _apply_compression_stage(self, image_tensor):
        """Apply JPEG compression degradation
        
        Uses ARNIQA's predefined levels directly:
        - JPEG: [43, 36, 24, 7, 4] (quality - lower is worse)
        """
        available_levels = self.compression_config.get('levels', [0, 1, 2, 3, 4])
        level = random.choice(available_levels)
        
        return self.distorter.apply_distortion_to_tensor(image_tensor, 'jpeg', level)
    
    def _apply_diffbir_codeformer_degradation(self, clean_tensor):
        """Apply DiffBIR Codeformer single-stage degradation (numpy-based).
        
        Converts CHW RGB [0,1] tensor to HWC BGR [0,1] numpy (Codeformer convention),
        applies degradation via apply_codeformer_degradation(), converts back.
        
        Args:
            clean_tensor: Clean image tensor [C, H, W] in [0, 1] RGB
            
        Returns:
            Degraded image tensor [C, H, W] in [0, 1] RGB
        """
        # CHW RGB [0,1] tensor -> HWC RGB [0,1] numpy
        img_rgb = clean_tensor.permute(1, 2, 0).numpy()  # [H, W, 3]
        # RGB -> BGR (Codeformer convention)
        img_bgr = img_rgb[..., ::-1].copy()
        
        # Apply codeformer degradation with config params
        cfg = self.diffbir_cf_config
        img_lq = self.diffbir_apply_codeformer(
            img_bgr,
            blur_kernel_size=cfg.get('blur_kernel_size', 41),
            kernel_list=cfg.get('kernel_list', None),
            kernel_prob=cfg.get('kernel_prob', None),
            blur_sigma=cfg.get('blur_sigma', None),
            downsample_range=cfg.get('downsample_range', None),
            noise_range=cfg.get('noise_range', None),
            jpeg_range=cfg.get('jpeg_range', None),
        )
        
        # BGR -> RGB, HWC -> CHW, numpy -> tensor
        img_rgb_out = img_lq[..., ::-1].copy()  # BGR -> RGB
        degraded_tensor = torch.from_numpy(
            np.transpose(img_rgb_out, (2, 0, 1)).copy()
        ).float()
        degraded_tensor = torch.clamp(degraded_tensor, 0.0, 1.0)
        
        return degraded_tensor
    
    def _apply_diffbir_realesrgan_degradation(self, clean_tensor):
        """Apply DiffBIR RealESRGAN two-stage degradation (PyTorch-based).
        
        Adds batch dim, generates kernels, applies two-stage degradation via
        apply_realesrgan_degradation() on CPU, removes batch dim.
        
        Memory management: The RealESRGAN pipeline creates many variable-size
        intermediate tensors (F.interpolate with random scales, filter2D,
        DiffJPEG). On CPU, glibc malloc fragments memory without returning it
        to the OS. We mitigate this with:
        1. Numpy round-trip: forces a compact, contiguous allocation
        2. Explicit del: releases PyTorch storage objects immediately
        3. gc.collect(): forces Python garbage collection
        4. malloc_trim(0): forces glibc to return freed pages to the OS
        
        Args:
            clean_tensor: Clean image tensor [C, H, W] in [0, 1] RGB
            
        Returns:
            Degraded image tensor [C, H, W] in [0, 1] RGB
        """
        cfg = self.diffbir_re_config
        
        # Generate kernels (per-image, stochastic)
        # Pass pre-created pulse_tensor to avoid torch.zeros(21,21) each call
        kernel1, kernel2, sinc_kernel = self.diffbir_generate_kernels(
            kernel_list=cfg.get('kernel_list', None),
            kernel_prob=cfg.get('kernel_prob', None),
            blur_sigma=cfg.get('blur_sigma', None),
            betag_range=cfg.get('betag_range', None),
            betap_range=cfg.get('betap_range', None),
            sinc_prob=cfg.get('sinc_prob', 0.1),
            kernel_list2=cfg.get('kernel_list2', None),
            kernel_prob2=cfg.get('kernel_prob2', None),
            blur_sigma2=cfg.get('blur_sigma2', None),
            betag_range2=cfg.get('betag_range2', None),
            betap_range2=cfg.get('betap_range2', None),
            sinc_prob2=cfg.get('sinc_prob2', 0.1),
            final_sinc_prob=cfg.get('final_sinc_prob', 0.8),
            pulse_tensor=self.diffbir_pulse_tensor,
        )
        
        # Add batch dim: (21, 21) -> (1, 21, 21)
        kernel1 = kernel1.unsqueeze(0)
        kernel2 = kernel2.unsqueeze(0)
        sinc_kernel = sinc_kernel.unsqueeze(0)
        
        # Add batch dim to image: CHW -> BCHW
        img_batch = clean_tensor.unsqueeze(0)
        
        # Apply two-stage degradation on CPU
        lq = self.diffbir_apply_realesrgan(
            img_batch, kernel1, kernel2, sinc_kernel,
            self.diffbir_jpeger,
            device='cpu',
            resize_prob=cfg.get('resize_prob', None),
            resize_range=cfg.get('resize_range', None),
            gaussian_noise_prob=cfg.get('gaussian_noise_prob', 0.5),
            noise_range=cfg.get('noise_range', None),
            poisson_scale_range=cfg.get('poisson_scale_range', None),
            gray_noise_prob=cfg.get('gray_noise_prob', 0.4),
            jpeg_range=cfg.get('jpeg_range', None),
            second_blur_prob=cfg.get('second_blur_prob', 0.8),
            stage2_scale=cfg.get('stage2_scale', 1.0),
            resize_prob2=cfg.get('resize_prob2', None),
            resize_range2=cfg.get('resize_range2', None),
            gaussian_noise_prob2=cfg.get('gaussian_noise_prob2', 0.5),
            noise_range2=cfg.get('noise_range2', None),
            poisson_scale_range2=cfg.get('poisson_scale_range2', None),
            gray_noise_prob2=cfg.get('gray_noise_prob2', 0.4),
            jpeg_range2=cfg.get('jpeg_range2', None),
        )
        
        # --- Memory management: break ties to PyTorch's fragmented allocator ---
        # Numpy round-trip: creates a compact contiguous copy, releases all
        # PyTorch intermediate storage that lq's tensor graph references
        lq_np = lq.squeeze(0).numpy().copy()  # CHW float32 numpy, contiguous
        
        # Explicitly delete all PyTorch tensors from this call
        del lq, kernel1, kernel2, sinc_kernel, img_batch
        
        # Force Python GC to collect any circular refs / deferred objects
        gc.collect()
        
        # Force glibc to return freed memory pages to the OS
        # (counters malloc fragmentation from variable-size F.interpolate calls)
        if self._libc is not None:
            self._libc.malloc_trim(0)
        
        # Rebuild tensor from compact numpy array
        degraded_tensor = torch.from_numpy(lq_np).float()
        
        return degraded_tensor
    
    def _training_preprocess(self, rasters):
        """Apply training preprocessing (augmentation only)
        
        Note: Resize is now done in _get_data_item BEFORE degradation generation
        to ensure consistent dimensions and avoid tensor storage issues.
        """
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)
        
        return rasters

    def _augment_data(self, rasters):
        """Apply data augmentation"""
        # Horizontal flip (both clean and degraded together)
        if hasattr(self.augm_args, 'lr_flip_p') and random.random() < self.augm_args.lr_flip_p:
            # Flip all tensors
            rasters = {k: v.flip(-1) if torch.is_tensor(v) else v for k, v in rasters.items()}
        
        return rasters

    def _resize_data(self, data_dict):
        """Resize RGB data dictionary using random crop (maintains aspect ratio)
        
        Strategy:
        - If image larger than target: random crop to target size
        - If image smaller than target: resize to cover target, then random crop
        
        This preserves aspect ratio and avoids stretching.
        Uses numpy operations to create fresh tensors without storage issues.
        
        Args:
            data_dict: Dictionary with {prefix}_rgb_int and {prefix}_rgb_norm tensors
            
        Returns:
            Dictionary with cropped tensors at target resolution
        """
        target_h, target_w = self.resize_to_hw
        
        # Get current size from first rgb tensor
        first_key = next(k for k in data_dict.keys() if k.endswith('_rgb_int'))
        _, curr_h, curr_w = data_dict[first_key].shape
        
        # Determine if we need to resize first (image too small)
        need_resize = curr_h < target_h or curr_w < target_w
        
        if need_resize:
            # Calculate scale to ensure image covers target area
            scale = max(target_h / curr_h, target_w / curr_w)
            new_h = int(curr_h * scale) + 1  # +1 to ensure coverage
            new_w = int(curr_w * scale) + 1
        else:
            new_h, new_w = curr_h, curr_w
        
        # Calculate random crop position (same for all tensors)
        crop_top = random.randint(0, max(0, new_h - target_h))
        crop_left = random.randint(0, max(0, new_w - target_w))
        
        result = {}
        
        for key, tensor in data_dict.items():
            if key.endswith('_rgb_int') or key.endswith('_rgb_norm'):
                if len(tensor.shape) == 3:  # [C, H, W]
                    # Convert to numpy HWC
                    if key.endswith('_rgb_int'):
                        img_np = tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                    else:
                        img_np = ((tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
                    
                    # Resize if needed (image too small)
                    if need_resize:
                        pil_img = Image.fromarray(img_np)
                        pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
                        img_np = np.array(pil_resized)
                    
                    # Random crop
                    cropped_np = img_np[crop_top:crop_top+target_h, crop_left:crop_left+target_w, :]
                    
                    # Convert back to tensor with fresh memory
                    cropped_chw = np.transpose(cropped_np, (2, 0, 1)).copy()
                    
                    if key.endswith('_rgb_int'):
                        result[key] = torch.from_numpy(cropped_chw.astype(int)).int()
                    else:
                        cropped_norm = cropped_chw.astype(np.float32) / 127.5 - 1.0
                        result[key] = torch.from_numpy(cropped_norm).float()
                else:
                    result[key] = tensor
            else:
                result[key] = tensor
        
        return result

    def __del__(self):
        """Cleanup tar objects"""
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None