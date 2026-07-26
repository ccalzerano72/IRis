# --------------------------------------------------------------------------
# Hybrid-Wavelet: Joint UNet + Wavelet-Conditioned ControlNet Pipeline
# Thesis Project: Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction
#
# This pipeline extends the hybrid-003 ControlNet pipeline with:
# 1. Latent-space conditioning: 8-channel UNet input [degraded_latent, noisy_latent]
# 2. Wavelet-space conditioning: ControlNet receives DWT(degraded_rgb) instead of raw RGB
# 3. Optional ARNIQA quality-aware conditioning
#
# Both UNet and ControlNet are trainable (joint training with differential LR).
# ARNIQA encoder is frozen; only adapters are trained.
#
# Derived from: marigold/marigold_hybrid_controlnet_arniqa_003_pipeline.py
# Key change: controlnet_cond is wavelet-decomposed before passing to ControlNet
# --------------------------------------------------------------------------

import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Union

from .marigold_restoration_pipeline_base import (
    MarigoldRestorationOutput,
    resize_to_square,
    pad_to_square,
    unpad_from_square,
)
from .util.batchsize import find_batch_size
from .util.image_util import (
    chw2hwc,
    get_tv_resample_method,
)


class MarigoldHybridWaveletControlNetPipeline(DiffusionPipeline):
    """
    Pipeline for hybrid-wavelet: joint UNet + wavelet-conditioned ControlNet.

    Combines an 8-channel UNet (trainable, from SD2 weights expanded to 8ch)
    with a trainable ControlNet that receives wavelet-decomposed degraded images
    instead of raw RGB. Optional ARNIQA quality-aware conditioning.

    When wavelet conditioning is enabled, the ControlNet input is:
      DWT(degraded_rgb) → [B, N, H, W] where N depends on subband selection
    When disabled, falls back to standard RGB input (same as hybrid-003).

    Based on MarigoldHybridControlNetArniqa003Pipeline.

    Args:
        unet (`UNet2DConditionModel`):
            8-channel UNet. Receives concatenated [degraded_latent(4ch), noisy_latent(4ch)].
        controlnet (`ControlNetModel`):
            ControlNet module. Receives wavelet-decomposed conditioning.
        vae (`AutoencoderKL`):
            VAE for encoding/decoding images to/from latent space.
        scheduler (`DDIMScheduler` or `LCMScheduler`):
            Scheduler for the denoising process.
        text_encoder (`CLIPTextModel`):
            Text encoder for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        default_denoising_steps (`int`, *optional*):
            Default number of denoising steps.
        default_processing_resolution (`int`, *optional*):
            Default processing resolution.
    """

    latent_scale_factor = 0.18215

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
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            controlnet=controlnet,
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
        # ARNIQA conditioner (set by trainer via set_arniqa_conditioner())
        self.arniqa_conditioner = None

        # Wavelet conditioning config (set by trainer via set_wavelet_config())
        # Defaults to disabled (raw RGB input, same as hybrid-003)
        self.use_wavelet = False
        self.wavelet_subbands = 'all'
        self.wavelet_upsample = True
        self.haar_filters = None

    def set_wavelet_config(self, enabled, subbands='all', upsample=True):
        """Configure wavelet conditioning for inference.

        Called by the trainer after pipeline initialization to pass wavelet
        settings that match the training configuration.

        Args:
            enabled: Whether to apply wavelet decomposition to controlnet_cond
            subbands: Which subbands to use ('all', 'll_only', 'hf_only')
            upsample: Whether to upsample wavelet coefficients back to input resolution
        """
        self.use_wavelet = enabled
        self.wavelet_subbands = subbands
        self.wavelet_upsample = upsample

        if enabled:
            self._build_haar_filters()
            logging.info(
                f"Pipeline wavelet conditioning configured: "
                f"subbands={subbands}, upsample={upsample}"
            )
        else:
            logging.info("Pipeline wavelet conditioning disabled (raw RGB)")

    def _build_haar_filters(self):
        """Build Haar wavelet decomposition filters.

        Same implementation as trainer._build_haar_filters().
        Creates [12, 1, 2, 2] depthwise conv filters for 3-channel RGB input.
        """
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5

        filters_1ch = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        filters_3ch = filters_1ch.repeat(3, 1, 1, 1)  # [12, 1, 2, 2]

        self.haar_filters = filters_3ch

    def _apply_wavelet_decomposition(self, rgb_input):
        """Apply 1-level Haar DWT to an RGB image tensor.

        Same implementation as trainer._apply_wavelet_decomposition().

        Args:
            rgb_input: [B, 3, H, W] tensor in [-1, 1]

        Returns:
            Wavelet-decomposed tensor with shape depending on config.
        """
        device = rgb_input.device
        B, C, H, W = rgb_input.shape

        haar_filters = self.haar_filters.to(device=device, dtype=rgb_input.dtype)

        # Depthwise conv: [B, 3, H, W] → [B, 12, H/2, W/2]
        wavelet_coeffs = F.conv2d(rgb_input, haar_filters, stride=2, groups=3)

        # Rearrange: group by subband instead of by channel
        wavelet_coeffs = wavelet_coeffs.view(B, 3, 4, H // 2, W // 2)
        wavelet_coeffs = wavelet_coeffs.permute(0, 2, 1, 3, 4)
        wavelet_coeffs = wavelet_coeffs.reshape(B, 12, H // 2, W // 2)
        # channels 0-2 = LL, 3-5 = LH, 6-8 = HL, 9-11 = HH

        # Subband selection
        if self.wavelet_subbands == 'all':
            result = wavelet_coeffs
        elif self.wavelet_subbands == 'll_only':
            result = wavelet_coeffs[:, 0:3, :, :]
        elif self.wavelet_subbands == 'hf_only':
            result = wavelet_coeffs[:, 3:12, :, :]
        else:
            raise ValueError(f"Unknown subbands: {self.wavelet_subbands}")

        # Optional upsample back to input resolution
        if self.wavelet_upsample:
            result = F.interpolate(
                result, size=(H, W), mode='bilinear', align_corners=False
            )

        return result

    def encode_empty_text(self):
        """Encode text embedding for empty prompt.
        Copied from 003 pipeline.encode_empty_text() (line 115-128)."""
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

    def set_arniqa_conditioner(self, conditioner):
        """Set the ARNIQA conditioner for quality-aware conditioning.
        Copied from 003 pipeline.set_arniqa_conditioner() (line 131-142)."""
        self.arniqa_conditioner = conditioner

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """Encode RGB image into latent.
        Copied from 003 pipeline.encode_rgb() (line 144-153)."""
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """Decode RGB latent into RGB image.
        Copied from 003 pipeline.decode_rgb() (line 155-163)."""
        rgb_latent = rgb_latent / self.latent_scale_factor
        z = self.vae.post_quant_conv(rgb_latent)
        rgb_image = self.vae.decoder(z)
        return rgb_image

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform a single hybrid-wavelet restoration inference pass.

        Based on 003 pipeline.single_infer() (lines 166-330).
        KEY DIFFERENCE: applies wavelet decomposition to controlnet_cond
        before the denoising loop when wavelet conditioning is enabled.

        Args:
            rgb_in: Input degraded RGB image [B, 3, H, W] in [-1, 1].
            num_inference_steps: Number of DDIM denoising steps.
            generator: Random generator for noise.
            show_pbar: Display progress bar.
            guidance_scale: CFG scale. Values > 1.0 apply CFG.

        Returns:
            Predicted restored RGB image [B, 3, H, W] in [-1, 1].
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Encode empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()

        # Conditioning: ARNIQA quality features or empty text embedding
        if self.arniqa_conditioner is not None:
            batch_text_embed = self.arniqa_conditioner(rgb_in, apply_dropout=False)
            batch_text_embed = batch_text_embed.to(device)
        else:
            batch_text_embed = self.empty_text_embed.repeat(
                (rgb_in.shape[0], 1, 1)
            ).to(device)

        # Empty text embed for CFG unconditional path
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_in.shape[0], 1, 1)
        ).to(device)

        # Encode degraded RGB -> degraded_latent
        rgb_latent = self.encode_rgb(rgb_in)  # [B, 4, h, w]

        # ControlNet conditioning: raw RGB or wavelet-decomposed
        controlnet_cond = rgb_in  # [B, 3, H, W]

        # NEW: Apply wavelet decomposition to controlnet_cond
        if self.use_wavelet:
            controlnet_cond = self._apply_wavelet_decomposition(controlnet_cond)
            # Shape: [B, N, H, W] or [B, N, H/2, W/2] depending on upsample config

        # Initial noisy latent
        target_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )

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
            if guidance_scale > 1.0:
                # --- Classifier-Free Guidance ---
                down_block_res, mid_block_res = self.controlnet(
                    target_latent,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                cond_input = torch.cat(
                    [rgb_latent, target_latent], dim=1
                )

                noise_pred_cond = self.unet(
                    cond_input,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    down_block_additional_residuals=down_block_res,
                    mid_block_additional_residual=mid_block_res,
                ).sample

                # Unconditional: zeroed ControlNet residuals + empty text embedding
                zero_down_block_res = [torch.zeros_like(r) for r in down_block_res]
                zero_mid_block_res = torch.zeros_like(mid_block_res)

                uncond_input = torch.cat(
                    [rgb_latent, target_latent], dim=1
                )

                noise_pred_uncond = self.unet(
                    uncond_input,
                    t,
                    encoder_hidden_states=batch_empty_text_embed,
                    down_block_additional_residuals=zero_down_block_res,
                    mid_block_additional_residual=zero_mid_block_res,
                ).sample

                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                # --- Standard inference (no CFG) ---
                down_block_res, mid_block_res = self.controlnet(
                    target_latent,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                unet_input = torch.cat(
                    [rgb_latent, target_latent], dim=1
                )

                noise_pred = self.unet(
                    unet_input,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    down_block_additional_residuals=down_block_res,
                    mid_block_additional_residual=mid_block_res,
                ).sample

            # Scheduler step
            step_output = self.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            )
            target_latent = step_output.prev_sample

        # Decode latent to RGB
        restored_rgb = self.decode_rgb(target_latent)
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
        guidance_scale: float = 1.0,
    ) -> MarigoldRestorationOutput:
        """
        Invoke the hybrid-wavelet restoration pipeline.
        Copied from 003 pipeline.__call__() (lines 334-488).
        """
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # Image Preprocess
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        padding_info = None
        if processing_res > 0:
            rgb = resize_to_square(
                rgb,
                target_resolution=processing_res,
                resample_method=resample_method,
            )
        else:
            rgb, padding_info = pad_to_square(rgb, padding_mode="reflect", min_size=768)

        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # Predicting restoration
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        target_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
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
                guidance_scale=guidance_scale,
            )
            target_pred_ls.append(target_pred_raw.detach())
        target_preds = torch.concat(target_pred_ls, dim=0)
        torch.cuda.empty_cache()

        # Test-time ensembling
        if ensemble_size > 1:
            final_pred = torch.mean(target_preds, dim=0, keepdim=True)
            pred_uncert = torch.std(target_preds, dim=0, keepdim=True)
        else:
            final_pred = target_preds
            pred_uncert = None

        # Post-processing
        if padding_info is not None:
            final_pred = unpad_from_square(final_pred, padding_info)
            if pred_uncert is not None:
                pred_uncert = unpad_from_square(pred_uncert, padding_info)
        elif match_input_res:
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

        final_pred = final_pred.squeeze()
        final_pred = final_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        final_pred = (final_pred + 1.0) / 2.0
        final_pred = final_pred.clip(0, 1)

        if final_pred.ndim == 3:
            restored_hwc = chw2hwc(final_pred)
        else:
            restored_hwc = final_pred

        restored_img_array = (restored_hwc * 255).astype(np.uint8)
        restored_img = Image.fromarray(restored_img_array)

        return MarigoldRestorationOutput(
            restored_np=final_pred,
            restored_img=restored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """Check if denoising step is reasonable.
        Copied from 003 pipeline._check_inference_step() (lines 489-510)."""
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"Unexpected timestep_spacing: {self.scheduler.config.timestep_spacing}, "
                    f"expected: 'trailing'"
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if (
                "trailing" != self.scheduler.config.timestep_spacing
                and "linspace" != self.scheduler.config.timestep_spacing
            ):
                logging.warning(
                    f"Unexpected timestep_spacing: {self.scheduler.config.timestep_spacing}, "
                    f"expected: 'trailing' or 'linspace'"
                )
        else:
            logging.warning(f"Unexpected scheduler type: {type(self.scheduler)}")
