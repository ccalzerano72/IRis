# Marigold Restoration Pipeline - Patched Version for Large Images
# Thesis Project: Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# This module extends MarigoldRestorationPipelineBase with patch-based processing
# for images larger than the model's native processing resolution.

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from .marigold_restoration_pipeline_base import (
    MarigoldRestorationPipelineBase,
    MarigoldRestorationOutput,
)
from .util.image_util import chw2hwc


def create_gaussian_weight_2d(size: int, sigma_factor: float = 0.3) -> torch.Tensor:
    """
    Create a 2D Gaussian weight map for patch blending.
    
    The Gaussian is centered on the patch with sigma proportional to patch size.
    This provides smooth falloff at edges for seamless blending.
    
    Args:
        size: Patch size (square patches assumed)
        sigma_factor: Sigma as fraction of size (default 0.3 gives good blending)
    
    Returns:
        2D tensor of shape [size, size] with values in (0, 1]
    """
    sigma = size * sigma_factor
    center = (size - 1) / 2.0
    
    coords = torch.arange(size, dtype=torch.float32)
    gauss_1d = torch.exp(-((coords - center) ** 2) / (2 * sigma ** 2))
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    gauss_2d = gauss_2d / gauss_2d.max()
    
    return gauss_2d


def create_linear_weight_2d(size: int, border_fraction: float = 0.25) -> torch.Tensor:
    """
    Create a 2D linear weight map for patch blending.
    
    Linear ramp from 0 at edges to 1 in the center region.
    
    Args:
        size: Patch size (square patches assumed)
        border_fraction: Fraction of size for the linear ramp region
    
    Returns:
        2D tensor of shape [size, size] with values in [0, 1]
    """
    border = int(size * border_fraction)
    if border < 1:
        border = 1
    
    weight_1d = torch.ones(size, dtype=torch.float32)
    
    for i in range(border):
        weight_1d[i] = (i + 1) / (border + 1)
    for i in range(border):
        weight_1d[size - 1 - i] = (i + 1) / (border + 1)
    
    weight_2d = torch.min(
        weight_1d.unsqueeze(1).expand(size, size),
        weight_1d.unsqueeze(0).expand(size, size)
    )
    
    return weight_2d



class MarigoldRestorationPipelinePatched(MarigoldRestorationPipelineBase):
    """
    Patch-based extension of MarigoldRestorationPipelineBase.
    
    Inherits all functionality from the base pipeline and adds patch-based processing
    for large images. When patch_size is set (> 0), images are divided into overlapping
    patches, processed individually, and reassembled with weighted blending.
    When patch_size is None or 0, behaves exactly like the base pipeline.
    
    Args:
        unet, vae, scheduler, text_encoder, tokenizer: Same as base pipeline
        default_denoising_steps: Default denoising steps
        default_processing_resolution: Default processing resolution
        patch_size: Size of square patches (None or 0 to disable patching)
        overlap_ratio: Fraction of patch_size for overlap (default 0.25)
        blend_mode: "gaussian" or "linear" (default "gaussian")
    
    Example:
        # Load with patching enabled
        pipe = MarigoldRestorationPipelinePatched.from_pretrained(...)
        pipe.set_patch_size(768)
        pipe.set_overlap_ratio(0.25)
        result = pipe(large_image)
        
        # Or disable patching to use base behavior
        pipe.set_patch_size(None)
        result = pipe(image, processing_res=768)
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        patch_size: Optional[int] = 768,
        overlap_ratio: float = 0.25,
        blend_mode: str = "gaussian",
        max_ensemble_batch_size: Optional[int] = 5,
    ):
        # Initialize base class
        super().__init__(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )
        
        # Set max_ensemble_batch_size if provided
        if max_ensemble_batch_size is not None:
            self.set_max_ensemble_batch_size(max_ensemble_batch_size)
        
        # Patch-specific attributes
        self._patch_size = None
        self._overlap_ratio = 0.25
        self._blend_mode = "gaussian"
        self._blend_weights = None
        
        # Use setters for validation
        if patch_size is not None and patch_size > 0:
            self.set_patch_size(patch_size)
        self.set_overlap_ratio(overlap_ratio)
        self.set_blend_mode(blend_mode)
    
    @property
    def patch_size(self) -> Optional[int]:
        return self._patch_size
    
    @property
    def overlap_ratio(self) -> float:
        return self._overlap_ratio
    
    @property
    def blend_mode(self) -> str:
        return self._blend_mode
    
    def set_patch_size(self, patch_size: Optional[int]):
        """
        Set the patch size for processing. Set to None or 0 to disable patching.
        
        Args:
            patch_size: Size of square patches (must be positive, or None/0 to disable)
        """
        if patch_size is None or patch_size <= 0:
            self._patch_size = None
            self._blend_weights = None
        else:
            self._patch_size = patch_size
            self._recompute_blend_weights()
    
    def set_overlap_ratio(self, overlap_ratio: float):
        """
        Set the overlap ratio between patches.
        
        Args:
            overlap_ratio: Fraction of patch_size for overlap (must be in [0, 0.5))
        """
        assert 0.0 <= overlap_ratio < 0.5, "overlap_ratio must be in [0, 0.5)"
        self._overlap_ratio = overlap_ratio
        self._recompute_blend_weights()
    
    def set_blend_mode(self, blend_mode: str):
        """
        Set the blending mode for patch reassembly.
        
        Args:
            blend_mode: "gaussian" for smooth 2D Gaussian weights, "linear" for linear ramps
        """
        assert blend_mode in ("gaussian", "linear"), "blend_mode must be 'gaussian' or 'linear'"
        self._blend_mode = blend_mode
        self._recompute_blend_weights()
    
    def _recompute_blend_weights(self):
        """Recompute blending weights when parameters change."""
        if self._patch_size is None:
            self._blend_weights = None
            return
        
        if self._blend_mode == "gaussian":
            self._blend_weights = create_gaussian_weight_2d(self._patch_size)
        else:
            self._blend_weights = create_linear_weight_2d(self._patch_size, border_fraction=self._overlap_ratio)


    def _compute_patch_positions(
        self,
        img_height: int,
        img_width: int,
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
        """
        Compute patch positions with variable overlap to cover the image.
        
        Strategy:
        1. Pad any dimension smaller than patch_size
        2. Compute patch positions with adjusted spacing to cover the padded image
        
        Args:
            img_height: Image height
            img_width: Image width
        
        Returns:
            Tuple of:
            - List of (top, left) positions for each patch
            - Padding info (pad_top, pad_bottom, pad_left, pad_right) or (0,0,0,0) if no padding
        """
        patch_size = self._patch_size
        min_overlap = int(patch_size * self._overlap_ratio)
        
        # Step 1: Compute padding for dimensions smaller than patch_size
        pad_h = max(0, patch_size - img_height)
        pad_w = max(0, patch_size - img_width)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padding_info = (pad_top, pad_bottom, pad_left, pad_right)
        
        # Step 2: Compute patch positions on the padded dimensions
        padded_height = img_height + pad_h
        padded_width = img_width + pad_w
        
        positions = []
        
        # Height dimension
        if padded_height == patch_size:
            row_positions = [0]
        else:
            max_stride = patch_size - min_overlap
            n_rows = max(1, int(np.ceil((padded_height - patch_size) / max_stride)) + 1)
            if n_rows == 1:
                row_positions = [0]
            else:
                actual_stride = (padded_height - patch_size) / (n_rows - 1)
                row_positions = [int(round(i * actual_stride)) for i in range(n_rows)]
        
        # Width dimension
        if padded_width == patch_size:
            col_positions = [0]
        else:
            max_stride = patch_size - min_overlap
            n_cols = max(1, int(np.ceil((padded_width - patch_size) / max_stride)) + 1)
            if n_cols == 1:
                col_positions = [0]
            else:
                actual_stride = (padded_width - patch_size) / (n_cols - 1)
                col_positions = [int(round(i * actual_stride)) for i in range(n_cols)]
        
        for row in row_positions:
            for col in col_positions:
                positions.append((row, col))
        
        return positions, padding_info
    
    def _extract_patches(
        self,
        image: torch.Tensor,
        positions: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Extract patches from image at given positions."""
        patches = []
        for top, left in positions:
            patch = image[:, :, top:top + self._patch_size, left:left + self._patch_size]
            patches.append(patch)
        return torch.cat(patches, dim=0)
    
    def _reassemble_patches(
        self,
        patches: torch.Tensor,
        positions: List[Tuple[int, int]],
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        """Reassemble patches into full image with weighted blending."""
        device = patches.device
        dtype = patches.dtype
        n_channels = patches.shape[1]
        
        output = torch.zeros(1, n_channels, output_height, output_width, device=device, dtype=dtype)
        weight_sum = torch.zeros(1, 1, output_height, output_width, device=device, dtype=dtype)
        
        blend_weights = self._blend_weights.to(device=device, dtype=dtype)
        blend_weights = blend_weights.unsqueeze(0).unsqueeze(0)
        
        for i, (top, left) in enumerate(positions):
            patch = patches[i:i+1]
            output[:, :, top:top + self._patch_size, left:left + self._patch_size] += patch * blend_weights
            weight_sum[:, :, top:top + self._patch_size, left:left + self._patch_size] += blend_weights
        
        output = output / (weight_sum + 1e-8)
        return output


    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        prompt: str = "",
        guidance_scale: float = 1.0,
        max_ensemble_batch_size: Optional[int] = None,
    ) -> MarigoldRestorationOutput:
        """
        Process image, using patch-based approach if patch_size is set.
        
        If patch_size is None or 0, delegates to base pipeline behavior.
        Otherwise, divides the image into overlapping patches, processes each,
        and reassembles with weighted blending.
        
        Args:
            input_image: Input degraded RGB image (PIL Image or torch Tensor)
            denoising_steps: Number of denoising steps (None for model default)
            ensemble_size: Number of predictions to ensemble per patch
            processing_res: Processing resolution (ignored when patching is enabled)
            match_input_res: Match output to input resolution
            resample_method: Resampling method
            batch_size: Batch size for ensemble processing
            generator: Random generator for reproducibility
            show_progress_bar: Show progress bars
            ensemble_kwargs: Additional ensemble settings
            prompt: Text prompt for conditioning
            guidance_scale: CFG guidance scale
            max_ensemble_batch_size: Maximum ensemble predictions to process simultaneously.
                                    Limits GPU memory usage. If None, uses pipeline default.
        
        Returns:
            MarigoldRestorationOutput with restored image
        """
        # If patching is disabled, use base pipeline behavior
        if self._patch_size is None:
            return super().__call__(
                input_image=input_image,
                denoising_steps=denoising_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                resample_method=resample_method,
                batch_size=batch_size,
                generator=generator,
                show_progress_bar=show_progress_bar,
                ensemble_kwargs=ensemble_kwargs,
                prompt=prompt,
                guidance_scale=guidance_scale,
                max_ensemble_batch_size=max_ensemble_batch_size,
            )
        
        # Patch-based processing
        from torchvision.transforms.functional import pil_to_tensor
        
        # Convert input to tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
            if rgb.dim() == 3:
                rgb = rgb.unsqueeze(0)
        else:
            raise TypeError(f"Unknown input type: {type(input_image)}")
        
        original_size = rgb.shape[-2:]
        img_height, img_width = original_size
        
        # Compute patch positions and padding
        positions, padding_info = self._compute_patch_positions(img_height, img_width)
        pad_top, pad_bottom, pad_left, pad_right = padding_info
        
        # Apply padding if needed
        if any(p > 0 for p in padding_info):
            rgb = torch.nn.functional.pad(
                rgb,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="reflect"
            )
        
        padded_height, padded_width = rgb.shape[-2:]
        
        # Normalize to [-1, 1]
        rgb_norm = rgb.float() / 255.0 * 2.0 - 1.0
        
        # Extract patches (keep on CPU to save GPU memory)
        patches = self._extract_patches(rgb_norm, positions)
        n_patches = len(positions)
        
        # Free the full normalized image
        del rgb_norm
        
        if show_progress_bar:
            print(f"  Processing {n_patches} patches ({self._patch_size}x{self._patch_size}) "
                  f"with overlap_ratio={self._overlap_ratio}")
        
        # Process patches sequentially
        restored_patches = []
        uncertainties = []
        
        if show_progress_bar:
            iterable = tqdm(range(n_patches), desc="  Processing patches", leave=False)
        else:
            iterable = range(n_patches)
        
        for patch_idx in iterable:
            single_patch = patches[patch_idx:patch_idx+1]
            
            # Convert to [0, 255] uint8 for base pipeline
            patch_255 = ((single_patch + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
            
            # Call base pipeline with processing_res=patch_size to avoid unnecessary padding
            # (processing_res=0 would pad to 768x768 due to min_size in pad_to_square)
            result = super().__call__(
                input_image=patch_255,
                denoising_steps=denoising_steps,
                ensemble_size=ensemble_size,
                processing_res=self._patch_size,
                match_input_res=True,
                resample_method=resample_method,
                batch_size=batch_size,
                generator=generator,
                show_progress_bar=False,
                ensemble_kwargs=ensemble_kwargs,
                prompt=prompt,
                guidance_scale=guidance_scale,
                max_ensemble_batch_size=max_ensemble_batch_size,
            )
            
            # Convert result back to tensor [-1, 1] and keep on CPU to save GPU memory
            restored_np = result.restored_np
            restored_tensor = torch.from_numpy(restored_np).unsqueeze(0)
            restored_tensor = restored_tensor * 2.0 - 1.0
            
            restored_patches.append(restored_tensor)  # Keep on CPU
            
            if result.uncertainty is not None:
                uncert_tensor = torch.from_numpy(result.uncertainty).unsqueeze(0)
                uncertainties.append(uncert_tensor)  # Keep on CPU
            
            # Clear GPU memory after each patch
            del patch_255, result
            torch.cuda.empty_cache()
        
        # Stack and reassemble
        restored_patches = torch.cat(restored_patches, dim=0)
        restored_patches = restored_patches.to(device=self.device, dtype=self.dtype)
        
        restored_full = self._reassemble_patches(
            restored_patches, positions, padded_height, padded_width
        )
        
        # Handle uncertainty
        if uncertainties:
            uncert_patches = torch.cat(uncertainties, dim=0)
            uncert_patches = uncert_patches.to(device=self.device, dtype=self.dtype)
            uncert_full = self._reassemble_patches(
                uncert_patches, positions, padded_height, padded_width
            )
        else:
            uncert_full = None
        
        # Remove padding
        if any(p > 0 for p in padding_info):
            h_start = pad_top
            h_end = padded_height - pad_bottom if pad_bottom > 0 else padded_height
            w_start = pad_left
            w_end = padded_width - pad_right if pad_right > 0 else padded_width
            
            restored_full = restored_full[:, :, h_start:h_end, w_start:w_end]
            if uncert_full is not None:
                uncert_full = uncert_full[:, :, h_start:h_end, w_start:w_end]
        
        # Convert to output format
        restored_full = restored_full.squeeze(0)
        restored_np = restored_full.cpu().numpy()
        restored_np = (restored_np + 1.0) / 2.0
        restored_np = restored_np.clip(0, 1)
        
        if restored_np.ndim == 3:
            restored_hwc = chw2hwc(restored_np)
        else:
            restored_hwc = restored_np
        
        restored_img_array = (restored_hwc * 255).astype(np.uint8)
        restored_img = Image.fromarray(restored_img_array)
        
        if uncert_full is not None:
            uncert_np = uncert_full.squeeze().cpu().numpy()
        else:
            uncert_np = None
        
        return MarigoldRestorationOutput(
            restored_np=restored_np,
            restored_img=restored_img,
            uncertainty=uncert_np,
        )
