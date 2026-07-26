# --------------------------------------------------------------------------
# ControlNet Pipeline - Patched Version for Large Images
# Thesis Project: Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction
#
# This module extends MarigoldControlNetRestorationPipeline with patch-based
# processing for images larger than the model's native processing resolution (768x768).
#
# Pattern copied from marigold/marigold_hybrid_controlnet_arniqa_003_pipeline_patched.py
# which does the same for the hybrid-003 pipeline.
# --------------------------------------------------------------------------

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from .marigold_controlnet_restoration_pipeline import (
    MarigoldControlNetRestorationPipeline,
    MarigoldRestorationOutput,
)
from .marigold_restoration_pipeline_patched import (
    create_gaussian_weight_2d,
    create_linear_weight_2d,
)
from .util.image_util import chw2hwc


class MarigoldControlNetRestorationPipelinePatched(
    MarigoldControlNetRestorationPipeline
):
    """
    Patch-based extension of MarigoldControlNetRestorationPipeline.

    Inherits all functionality from the ControlNet pipeline and adds patch-based
    processing for large images. When patch_size is set (> 0), images are divided
    into overlapping patches, processed individually, and reassembled with weighted
    blending. When patch_size is None or 0, behaves exactly like the base pipeline.

    Pattern copied from MarigoldHybridControlNetArniqa003PipelinePatched.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        patch_size: Optional[int] = 768,
        overlap_ratio: float = 0.25,
        blend_mode: str = "gaussian",
    ):
        super().__init__(
            unet=unet,
            controlnet=controlnet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self._patch_size = None
        self._overlap_ratio = 0.25
        self._blend_mode = "gaussian"
        self._blend_weights = None

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
        if patch_size is None or patch_size <= 0:
            self._patch_size = None
            self._blend_weights = None
        else:
            self._patch_size = patch_size
            self._recompute_blend_weights()

    def set_overlap_ratio(self, overlap_ratio: float):
        assert 0.0 <= overlap_ratio < 0.5, "overlap_ratio must be in [0, 0.5)"
        self._overlap_ratio = overlap_ratio
        self._recompute_blend_weights()

    def set_blend_mode(self, blend_mode: str):
        assert blend_mode in ("gaussian", "linear")
        self._blend_mode = blend_mode
        self._recompute_blend_weights()

    def _recompute_blend_weights(self):
        if self._patch_size is None:
            self._blend_weights = None
            return
        if self._blend_mode == "gaussian":
            self._blend_weights = create_gaussian_weight_2d(self._patch_size)
        else:
            self._blend_weights = create_linear_weight_2d(
                self._patch_size, border_fraction=self._overlap_ratio
            )

    def _compute_patch_positions(
        self, img_height: int, img_width: int,
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
        patch_size = self._patch_size
        min_overlap = int(patch_size * self._overlap_ratio)

        pad_h = max(0, patch_size - img_height)
        pad_w = max(0, patch_size - img_width)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padding_info = (pad_top, pad_bottom, pad_left, pad_right)

        padded_height = img_height + pad_h
        padded_width = img_width + pad_w

        positions = []

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
        self, image: torch.Tensor, positions: List[Tuple[int, int]],
    ) -> torch.Tensor:
        patches = []
        for top, left in positions:
            patch = image[:, :, top:top + self._patch_size, left:left + self._patch_size]
            patches.append(patch)
        return torch.cat(patches, dim=0)

    def _reassemble_patches(
        self, patches: torch.Tensor, positions: List[Tuple[int, int]],
        output_height: int, output_width: int,
    ) -> torch.Tensor:
        device = patches.device
        dtype = patches.dtype
        n_channels = patches.shape[1]

        output = torch.zeros(1, n_channels, output_height, output_width, device=device, dtype=dtype)
        weight_sum = torch.zeros(1, 1, output_height, output_width, device=device, dtype=dtype)

        blend_weights = self._blend_weights.to(device=device, dtype=dtype)
        blend_weights = blend_weights.unsqueeze(0).unsqueeze(0)

        for i, (top, left) in enumerate(positions):
            patch = patches[i:i + 1]
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
        guidance_scale: float = 1.0,
    ) -> MarigoldRestorationOutput:
        """Process image with patch-based approach if patch_size is set."""
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
                guidance_scale=guidance_scale,
            )

        from torchvision.transforms.functional import pil_to_tensor

        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image).unsqueeze(0)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
            if rgb.dim() == 3:
                rgb = rgb.unsqueeze(0)
        else:
            raise TypeError(f"Unknown input type: {type(input_image)}")

        img_height, img_width = rgb.shape[-2:]
        positions, padding_info = self._compute_patch_positions(img_height, img_width)
        pad_top, pad_bottom, pad_left, pad_right = padding_info

        if any(p > 0 for p in padding_info):
            rgb = torch.nn.functional.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

        padded_height, padded_width = rgb.shape[-2:]
        rgb_norm = rgb.float() / 255.0 * 2.0 - 1.0
        patches = self._extract_patches(rgb_norm, positions)
        n_patches = len(positions)
        del rgb_norm

        if show_progress_bar:
            print(f"  Processing {n_patches} patches ({self._patch_size}x{self._patch_size}) "
                  f"with overlap_ratio={self._overlap_ratio}")

        restored_patches = []
        uncertainties = []
        iterable = tqdm(range(n_patches), desc="  Processing patches", leave=False) if show_progress_bar else range(n_patches)

        for patch_idx in iterable:
            single_patch = patches[patch_idx:patch_idx + 1]
            patch_255 = ((single_patch + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)

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
                guidance_scale=guidance_scale,
            )

            restored_np = result.restored_np
            restored_tensor = torch.from_numpy(restored_np).unsqueeze(0) * 2.0 - 1.0
            restored_patches.append(restored_tensor)

            if result.uncertainty is not None:
                uncertainties.append(torch.from_numpy(result.uncertainty).unsqueeze(0))

            del patch_255, result
            torch.cuda.empty_cache()

        restored_patches = torch.cat(restored_patches, dim=0).to(device=self.device, dtype=self.dtype)
        restored_full = self._reassemble_patches(restored_patches, positions, padded_height, padded_width)

        uncert_full = None
        if uncertainties:
            uncert_patches = torch.cat(uncertainties, dim=0).to(device=self.device, dtype=self.dtype)
            uncert_full = self._reassemble_patches(uncert_patches, positions, padded_height, padded_width)

        if any(p > 0 for p in padding_info):
            h_end = padded_height - pad_bottom if pad_bottom > 0 else padded_height
            w_end = padded_width - pad_right if pad_right > 0 else padded_width
            restored_full = restored_full[:, :, pad_top:h_end, pad_left:w_end]
            if uncert_full is not None:
                uncert_full = uncert_full[:, :, pad_top:h_end, pad_left:w_end]

        restored_full = restored_full.squeeze(0)
        restored_np = restored_full.cpu().numpy()
        restored_np = ((restored_np + 1.0) / 2.0).clip(0, 1)

        restored_hwc = chw2hwc(restored_np) if restored_np.ndim == 3 else restored_np
        restored_img = Image.fromarray((restored_hwc * 255).astype(np.uint8))
        uncert_np = uncert_full.squeeze().cpu().numpy() if uncert_full is not None else None

        return MarigoldRestorationOutput(
            restored_np=restored_np,
            restored_img=restored_img,
            uncertainty=uncert_np,
        )
