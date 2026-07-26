# Marigold Restoration Pipeline - Base Implementation
# Thesis Project: Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement

import logging
import math
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    HeunDiscreteScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Union

from .util.batchsize import find_batch_size
from .util.image_util import (
    chw2hwc,
    get_tv_resample_method,
    resize_max_res,
)


def resize_to_square(
    img: torch.Tensor,
    target_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to exact square dimensions (ignoring aspect ratio).
    
    This matches the training preprocessing where images are resized to 768x768
    regardless of original aspect ratio. This is critical for inference quality
    because the model was trained on square images only.
    
    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        target_resolution (`int`):
            Target size for both dimensions (e.g., 768 for 768x768).
        resample_method (`InterpolationMode`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image with shape [B, C, target_resolution, target_resolution].
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"
    return resize(img, (target_resolution, target_resolution), resample_method, antialias=True)


def pad_to_square(
    img: torch.Tensor,
    padding_mode: str = "reflect",
    min_size: int = 768,
) -> tuple:
    """
    Pad image to make it square, preserving original content without distortion.
    
    Uses reflection padding to avoid edge artifacts. The image is centered
    within the square output. Enforces a minimum size to match training resolution.
    
    Args:
        img (`torch.Tensor`):
            Image tensor to be padded. Expected shape: [B, C, H, W]
        padding_mode (`str`):
            Padding mode: 'reflect', 'replicate', or 'constant'. Default: 'reflect'
        min_size (`int`):
            Minimum target size. If max(h, w) < min_size, pad to min_size×min_size.
            Default: 768 (matches training resolution)
    
    Returns:
        `tuple`: (padded_image, padding_info) where padding_info is 
                 (pad_left, pad_right, pad_top, pad_bottom)
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"
    
    _, _, h, w = img.shape
    
    # Target size: max of (height, width, min_size) to ensure square and minimum size
    target_size = max(h, w, min_size)
    
    # Round up to nearest multiple of 8 (VAE downscaling factor).
    # Without this, ControlNet's internal conditioning encoder can produce spatial
    # dimensions that don't match the latent noise tensor (e.g., 1036 -> ceil(1036/8)=130
    # vs floor(1036/8)=129), causing a RuntimeError in sample + controlnet_cond.
    target_size = int(math.ceil(target_size / 8.0)) * 8
    
    # Already at target size - no padding needed
    if h == target_size and w == target_size:
        return img, (0, 0, 0, 0)
    
    # Calculate padding to center the image
    pad_h = target_size - h
    pad_w = target_size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # torch.nn.functional.pad uses order: (left, right, top, bottom)
    padded = torch.nn.functional.pad(
        img, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode
    )
    
    return padded, (pad_left, pad_right, pad_top, pad_bottom)


def unpad_from_square(
    img: torch.Tensor,
    padding_info: tuple,
) -> torch.Tensor:
    """
    Remove padding added by pad_to_square.
    
    Args:
        img (`torch.Tensor`):
            Padded image tensor. Expected shape: [B, C, H, W]
        padding_info (`tuple`):
            Tuple (pad_left, pad_right, pad_top, pad_bottom) from pad_to_square
    
    Returns:
        `torch.Tensor`: Unpadded image tensor with original dimensions.
    """
    pad_left, pad_right, pad_top, pad_bottom = padding_info
    
    # No padding was applied
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        return img
    
    _, _, h, w = img.shape
    
    # Calculate crop boundaries
    top = pad_top
    bottom = h - pad_bottom if pad_bottom > 0 else h
    left = pad_left
    right = w - pad_right if pad_right > 0 else w
    
    return img[:, :, top:bottom, left:right]


class MarigoldRestorationOutput(BaseOutput):
    """
    Output class for Marigold Image Restoration pipeline.

    Args:
        restored_np (`np.ndarray`):
            Restored image array, with shape [3, H, W] and values in the range of [0, 1].
        restored_img (`PIL.Image.Image`):
            Restored image, with the shape of [H, W, 3] and values in [0, 255].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    restored_np: np.ndarray
    restored_img: Image.Image
    uncertainty: Union[None, np.ndarray]



class MarigoldRestorationPipelineBase(DiffusionPipeline):
    """
    Base Pipeline for Marigold Image Restoration: Blind image restoration via diffusion-based quality-aware reconstruction.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the restoration latent, conditioned on degraded image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and restorations
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a restoration of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None
        self._text_embed_cache = {}  # Cache for text embeddings
        
        # Timestep-aware conditioning scaling (must match training settings)
        # These defaults match the trainer's default values
        self.cond_timestep_scaling = False
        self.cond_scale_min = 0.1
        self.cond_scale_max = 0.9
        self.scheduler_timesteps = 1000  # Default, will be updated from scheduler config
        
        # Latent normalization (must match training settings)
        # If enabled during training, must be enabled during inference
        self.normalize_latents = False
        
        # Maximum ensemble batch size for memory management
        # Default to None (process all ensemble predictions at once)
        self._max_ensemble_batch_size = None
        
        # ARNIQA quality-aware conditioning (optional)
        # When set, replaces empty_text_embed with quality features from degraded image
        # See: thesis-docs/notes/arniqa-quality-conditioning.md
        self.arniqa_conditioner = None
    
    def encode_text_prompt(self, prompt: str) -> torch.Tensor:
        """
        Encode a text prompt into embeddings.
        
        Args:
            prompt (`str`): Text prompt to encode. Empty string for unconditional.
            
        Returns:
            `torch.Tensor`: Text embeddings.
        """
        # Check cache first
        if prompt in self._text_embed_cache:
            return self._text_embed_cache[prompt]
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
        
        # Cache the embedding
        self._text_embed_cache[prompt] = text_embed
        
        return text_embed
    
    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode RGB latent into RGB image.

        Args:
            rgb_latent (`torch.Tensor`):
                RGB latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded RGB image.
        """
        # scale latent
        rgb_latent = rgb_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(rgb_latent)
        rgb_image = self.vae.decoder(z)
        return rgb_image

    def _get_conditioning_scale(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Compute timestep-aware conditioning scale.
        Higher scale at high noise levels (structure guidance needed),
        lower scale at low noise levels (fine details).
        
        This must match the training behavior exactly.
        
        Args:
            timestep: Single timestep value (scalar tensor)
        
        Returns:
            scale: Scalar tensor for broadcasting
        """
        # Linear interpolation: scale_min at t=0, scale_max at t=max_timestep
        t_normalized = timestep.float() / self.scheduler_timesteps
        scale = self.cond_scale_min + (self.cond_scale_max - self.cond_scale_min) * t_normalized
        return scale

    def set_conditioning_scale(
        self,
        enabled: bool = True,
        scale_min: float = 0.1,
        scale_max: float = 0.9,
        scheduler_timesteps: int = 1000,
    ):
        """
        Configure timestep-aware conditioning scaling.
        
        This should match the training configuration for optimal results.
        
        Args:
            enabled: Whether to apply timestep-aware scaling
            scale_min: Minimum scale (at t=0, fine details phase)
            scale_max: Maximum scale (at t=max, structure phase)
            scheduler_timesteps: Total training timesteps (usually 1000)
        """
        self.cond_timestep_scaling = enabled
        self.cond_scale_min = scale_min
        self.cond_scale_max = scale_max
        self.scheduler_timesteps = scheduler_timesteps

    def set_normalize_latents(self, enabled: bool = True):
        """
        Enable/disable latent normalization (zero mean + unit variance).
        
        This must match the training configuration. If the model was trained
        with normalize_latents=True, inference must also use it.
        
        Args:
            enabled: Whether to apply latent normalization
        """
        self.normalize_latents = enabled

    @property
    def max_ensemble_batch_size(self) -> Optional[int]:
        """Get the maximum ensemble batch size."""
        return self._max_ensemble_batch_size
    
    def set_max_ensemble_batch_size(self, max_batch_size: Optional[int]):
        """
        Set the maximum number of ensemble predictions to process simultaneously.
        
        This controls GPU memory usage when ensemble_size is large. Instead of
        processing all ensemble predictions at once, they are split into chunks
        of at most max_batch_size predictions.
        
        Args:
            max_batch_size: Maximum ensemble predictions per batch. 
                           Set to None to process all at once (default behavior).
                           Set to a positive integer to limit memory usage.
        
        Example:
            # Limit to 5 ensemble predictions at a time
            pipeline.set_max_ensemble_batch_size(5)
            
            # Process one at a time (minimum memory)
            pipeline.set_max_ensemble_batch_size(1)
            
            # Restore default behavior (no limit)
            pipeline.set_max_ensemble_batch_size(None)
        """
        if max_batch_size is not None:
            assert max_batch_size > 0, "max_ensemble_batch_size must be positive"
        self._max_ensemble_batch_size = max_batch_size

    def set_arniqa_conditioner(self, conditioner):
        """
        Set the ARNIQA conditioner for quality-aware conditioning.
        
        When set, ARNIQA features from the degraded image replace empty_text_embed
        in the U-Net cross-attention, providing quality-aware guidance.
        
        Args:
            conditioner: ArniqaConditioner instance, or None to disable
        """
        self.arniqa_conditioner = conditioner

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        prompt: str = "",
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform a single restoration prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input degraded RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoising steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
            prompt (`str`, *optional*, defaults to `""`):
                Text prompt for conditioning. Empty string for unconditional generation.
            guidance_scale (`float`, *optional*, defaults to `1.0`):
                Classifier-Free Guidance scale. Values > 1.0 apply CFG.
        Returns:
            `torch.Tensor`: Predicted restored RGB image.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode degraded image
        rgb_latent = self.encode_rgb(rgb_in)  # [B, 4, h, w]

        # Latent normalization (if enabled)
        # Save degraded mean and std for restoration at the end (photometric consistency)
        # Normalize conditioning latent once
        if self.normalize_latents:
            degraded_latent_mean = rgb_latent.mean(dim=(2, 3), keepdim=True)
            degraded_latent_std = rgb_latent.std(dim=(2, 3), keepdim=True)
            rgb_latent = (rgb_latent - degraded_latent_mean) / (degraded_latent_std + 1e-8)

        # Noisy latent for restored image
        target_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Conditioning: ARNIQA quality features or text embedding
        if self.arniqa_conditioner is not None:
            # Use ARNIQA quality-aware conditioning
            # Extract features from degraded image (computed once, before denoising loop)
            # Output: [B, 1, 1024] - single quality token per sample
            batch_text_embed = self.arniqa_conditioner(rgb_in, apply_dropout=False)
            batch_text_embed = batch_text_embed.to(device)
        else:
            # Fallback: text embedding (empty or custom prompt)
            if prompt == "":
                if self.empty_text_embed is None:
                    self.encode_empty_text()
                text_embed = self.empty_text_embed
            else:
                text_embed = self.encode_text_prompt(prompt)
            
            batch_text_embed = text_embed.repeat(
                (rgb_latent.shape[0], 1, 1)
            ).to(device)  # [B, seq_len, hidden_dim]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            # Apply timestep-aware conditioning scaling (must match training)
            if self.cond_timestep_scaling:
                cond_scale = self._get_conditioning_scale(t)
                rgb_latent_scaled = rgb_latent * cond_scale
            else:
                rgb_latent_scaled = rgb_latent
            
            # Latent normalization: normalize noisy latent for UNet input only
            # Scheduler updates original target_latent to preserve its internal logic
            if self.normalize_latents:
                target_latent_mean = target_latent.mean(dim=(2, 3), keepdim=True)
                target_latent_std = target_latent.std(dim=(2, 3), keepdim=True)
                target_latent_input = (target_latent - target_latent_mean) / (target_latent_std + 1e-8)
            else:
                target_latent_input = target_latent
            
            # Classifier-Free Guidance (CFG)
            if guidance_scale > 1.0:
                # Unconditional prediction: use zeros as conditioning
                # TODO:: use batch inference instead of double invocation 
                uncond_input = torch.cat(
                    [torch.zeros_like(rgb_latent_scaled), target_latent_input], dim=1
                )
                noise_pred_uncond = self.unet(
                    uncond_input, t, encoder_hidden_states=batch_text_embed
                ).sample  # [B, 4, h, w]
                
                # Conditional prediction: use degraded image as conditioning
                cond_input = torch.cat(
                    [rgb_latent_scaled, target_latent_input], dim=1
                )
                noise_pred_cond = self.unet(
                    cond_input, t, encoder_hidden_states=batch_text_embed
                ).sample  # [B, 4, h, w]
                
                # CFG combination: uncond + guidance_scale * (cond - uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Standard inference (no CFG)
                unet_input = torch.cat(
                    [rgb_latent_scaled, target_latent_input], dim=1
                )  # this order is important

                # predict the noise residual
                noise_pred = self.unet(
                    unet_input, t, encoder_hidden_states=batch_text_embed
                ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            # Note: scheduler operates on target_latent_input (centered) but we store result in target_latent
            # HeunDiscreteScheduler.step() doesn't accept generator kwarg
            if isinstance(self.scheduler, HeunDiscreteScheduler):
                step_output = self.scheduler.step(
                    noise_pred, t, target_latent
                )
            else:
                step_output = self.scheduler.step(
                    noise_pred, t, target_latent, generator=generator
                )
            target_latent = step_output.prev_sample
            
        # Restore degraded_latent_mean and scale back variance for photometric consistency
        # This ensures output has same exposure/brightness and contrast as input
        if self.normalize_latents:
            target_latent = (target_latent - target_latent.mean(dim=(2, 3), keepdim=True)) / (target_latent.std(dim=(2, 3), keepdim=True) + 1e-8)
            target_latent = target_latent * degraded_latent_std + degraded_latent_mean

        restored_rgb = self.decode_rgb(target_latent)  # [B, 3, H, W]

        # clip prediction to valid RGB range
        restored_rgb = torch.clip(restored_rgb, -1.0, 1.0)

        return restored_rgb

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
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input degraded RGB image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection.
            ensemble_size (`int`, *optional*, defaults to `1`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize the prediction to match the input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or
                `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
            prompt (`str`, *optional*, defaults to `""`):
                Text prompt for conditioning the restoration. Empty string for unconditional generation
                (default behavior). Examples: "high quality, sharp, detailed" or "clean, noise-free".
            guidance_scale (`float`, *optional*, defaults to `1.0`):
                Classifier-Free Guidance scale. Values > 1.0 increase the influence of the conditioning
                (degraded image). Requires model trained with conditioning dropout (cfg.conditioning_dropout_prob > 0).
                Typical values: 1.0 (no CFG), 1.5-3.0 (moderate guidance), 5.0+ (strong guidance).
            max_ensemble_batch_size (`int`, *optional*, defaults to `None`):
                Maximum number of ensemble predictions to process simultaneously. Limits GPU memory usage
                when ensemble_size is large. If None, uses the value set via set_max_ensemble_batch_size()
                or processes all predictions at once. Set to a smaller value (e.g., 5) to avoid OOM errors.
        Returns:
            `MarigoldRestorationOutput`: Output class for Marigold image restoration pipeline, including:
            - **restored_np** (`np.ndarray`) Restored image array with values in the range of [0, 1]
            - **restored_img** (`PIL.Image.Image`) Restored image, with the shape of [H, W, 3] and values in [0, 255]
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image to exact square dimensions (matching training preprocessing)
        # Training uses 768x768 with aspect ratio distortion, so inference must match
        padding_info = None  # Track padding for later removal
        if processing_res > 0:
            # Resize to exact square (with aspect ratio distortion)
            rgb = resize_to_square(
                rgb,
                target_resolution=processing_res,
                resample_method=resample_method,
            )
        else:
            # processing_res == 0: Process at original resolution but pad to square
            # This avoids aspect ratio distortion while ensuring square input for the model
            # min_size=768 ensures we match the training resolution
            rgb, padding_info = pad_to_square(rgb, padding_mode="reflect", min_size=768)

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting restoration -----------------
        # Determine effective max_ensemble_batch_size
        effective_max_batch = max_ensemble_batch_size if max_ensemble_batch_size is not None else self._max_ensemble_batch_size
        
        # If no limit set, process all at once (original behavior)
        if effective_max_batch is None:
            effective_max_batch = ensemble_size
        else:
            effective_max_batch = min(effective_max_batch, ensemble_size)
        
        # Process ensemble in chunks to manage memory
        target_preds_list = []
        
        if show_progress_bar and effective_max_batch < ensemble_size:
            chunk_iterable = tqdm(
                range(0, ensemble_size, effective_max_batch),
                desc=" " * 2 + "Ensemble chunks",
                leave=False
            )
        else:
            chunk_iterable = range(0, ensemble_size, effective_max_batch)
        
        for chunk_start in chunk_iterable:
            chunk_end = min(chunk_start + effective_max_batch, ensemble_size)
            chunk_size = chunk_end - chunk_start
            
            # Batch repeated input image for this chunk
            duplicated_rgb = rgb_norm.expand(chunk_size, -1, -1, -1)
            single_rgb_dataset = TensorDataset(duplicated_rgb)
            
            if batch_size > 0:
                _bs = batch_size
            else:
                _bs = find_batch_size(
                    ensemble_size=chunk_size,
                    input_res=max(rgb_norm.shape[1:]),
                    dtype=self.dtype,
                )

            single_rgb_loader = DataLoader(
                single_rgb_dataset, batch_size=_bs, shuffle=False
            )

            # Predict restored images (batched)
            chunk_pred_ls = []
            if show_progress_bar:
                iterable = tqdm(
                    single_rgb_loader, desc=" " * 4 + "Inference batches", leave=False
                )
            else:
                iterable = single_rgb_loader
            
            for batch in iterable:
                (batched_img,) = batch
                target_pred_raw = self.single_infer(
                    rgb_in=batched_img,
                    num_inference_steps=denoising_steps,
                    show_pbar=show_progress_bar,
                    generator=generator,
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                )
                chunk_pred_ls.append(target_pred_raw.detach())
            
            chunk_preds = torch.concat(chunk_pred_ls, dim=0)
            target_preds_list.append(chunk_preds)
            
            # Clear memory after each chunk
            del chunk_pred_ls, chunk_preds
            torch.cuda.empty_cache()
        
        # Concatenate all chunks
        target_preds = torch.concat(target_preds_list, dim=0)
        del target_preds_list
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            # For restoration, we use simple averaging (no scale/shift invariance needed)
            final_pred = torch.mean(target_preds, dim=0, keepdim=True)
            # Calculate uncertainty as standard deviation
            pred_uncert = torch.std(target_preds, dim=0, keepdim=True)
        else:
            final_pred = target_preds
            pred_uncert = None

        # ----------------- Post-processing -----------------
        # Remove padding or resize back to original resolution
        if padding_info is not None:
            # processing_res == 0: Remove padding to restore original dimensions
            final_pred = unpad_from_square(final_pred, padding_info)
            if pred_uncert is not None:
                pred_uncert = unpad_from_square(pred_uncert, padding_info)
        elif match_input_res:
            # processing_res > 0: Resize back to original resolution
            final_pred = resize(
                final_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )
            if pred_uncert is not None:
                pred_uncert = resize(
                    pred_uncert,
                    input_size[-2:],
                    interpolation=resample_method,
                    antialias=True,
                )

        # Convert to numpy
        final_pred = final_pred.squeeze()
        final_pred = final_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Convert from [-1, 1] to [0, 1]
        final_pred = (final_pred + 1.0) / 2.0
        final_pred = final_pred.clip(0, 1)

        # Convert to PIL Image
        if final_pred.ndim == 3:  # [C, H, W]
            restored_hwc = chw2hwc(final_pred)
        else:  # [H, W] - shouldn't happen for RGB but just in case
            restored_hwc = final_pred
        
        restored_img_array = (restored_hwc * 255).astype(np.uint8)
        restored_img = Image.fromarray(restored_img_array)

        return MarigoldRestorationOutput(
            restored_np=final_pred,
            restored_img=restored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"Unexpected timestep_spacing: {self.scheduler.config.timestep_spacing}, expected: 'trailing'"
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if (
                "trailing" != self.scheduler.config.timestep_spacing
                and "linspace" != self.scheduler.config.timestep_spacing
            ):
                logging.warning(
                    f"Unexpected timestep_spacing: {self.scheduler.config.timestep_spacing}, expected: 'trailing' or 'linspace'"
                )
        elif isinstance(self.scheduler, HeunDiscreteScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"Unexpected timestep_spacing: {self.scheduler.config.timestep_spacing}, expected: 'trailing'"
                )
        else:
            logging.warning(f"Unexpected scheduler type: {type(self.scheduler)}")
