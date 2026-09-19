# Hybrid ControlNet + Degraded Latent Concatenation Image Restoration Pipeline
# Thesis Project: Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# This pipeline combines two conditioning mechanisms:
# 1. Latent-space conditioning: 8-channel UNet input [degraded_latent, noisy_latent]
# 2. Pixel-space conditioning: ControlNet receives degraded RGB and injects residuals
#
# The frozen 8ch UNet (from a pre-trained base checkpoint) provides structural
# restoration, while the trainable ControlNet adds texture/color refinement.

import logging
import numpy as np
import torch
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


class MarigoldHybridControlNetRestorationPipeline(DiffusionPipeline):
    """
    Pipeline for hybrid ControlNet + degraded latent concatenation image restoration.

    Combines an 8-channel UNet (frozen, from a pre-trained base restoration checkpoint)
    with a trainable ControlNet module. The UNet receives [degraded_latent, noisy_latent]
    as 8-channel input, while the ControlNet receives the degraded RGB image in pixel
    space and produces residuals injected into the UNet's intermediate features.

    Args:
        unet (`UNet2DConditionModel`):
            8-channel UNet from base restoration checkpoint. Receives concatenated
            [degraded_latent(4ch), noisy_latent(4ch)] as input.
        controlnet (`ControlNetModel`):
            Trainable ControlNet module for pixel-space conditioning. Receives
            noisy_latent (4ch) as sample and degraded RGB (3ch) as conditioning.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder for encoding/decoding images to/from latent space.
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

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt.
        Pattern copied from MarigoldControlNetRestorationPipeline.encode_empty_text().
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
        Pattern copied from MarigoldControlNetRestorationPipeline.encode_rgb().

        Args:
            rgb_in (`torch.Tensor`): Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode RGB latent into RGB image.
        Pattern copied from MarigoldControlNetRestorationPipeline.decode_rgb().

        Args:
            rgb_latent (`torch.Tensor`): RGB latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded RGB image.
        """
        rgb_latent = rgb_latent / self.latent_scale_factor
        z = self.vae.post_quant_conv(rgb_latent)
        rgb_image = self.vae.decoder(z)
        return rgb_image

    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform a single hybrid restoration inference pass.

        Combines latent-space conditioning (8ch UNet with degraded_latent concat)
        and pixel-space conditioning (ControlNet residuals from degraded RGB).

        At each denoising step:
          1. ControlNet receives (noisy_latent, t, text_embed, degraded_rgb) → residuals
          2. UNet receives cat([degraded_latent, noisy_latent]) as 8ch input + residuals

        For CFG (guidance_scale > 1.0):
          - Conditional: full ControlNet residuals + degraded_latent concat
          - Unconditional: zeroed ControlNet residuals + degraded_latent concat
          - CFG only controls ControlNet contribution; UNet always sees degraded_latent

        Args:
            rgb_in (`torch.Tensor`):
                Input degraded RGB image, shape [B, 3, H, W], values in [-1, 1].
            num_inference_steps (`int`):
                Number of diffusion denoising steps (DDIM).
            generator (`torch.Generator` or None):
                Random generator for initial noise generation.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            guidance_scale (`float`, *optional*, defaults to `1.0`):
                Classifier-Free Guidance scale. Values > 1.0 apply CFG.

        Returns:
            `torch.Tensor`: Predicted restored RGB image, shape [B, 3, H, W],
                values clipped to [-1, 1].
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode empty text embedding (computed once)
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_text_embed = self.empty_text_embed.repeat(
            (rgb_in.shape[0], 1, 1)
        ).to(device)  # [B, seq_len, hidden_dim]

        # Encode degraded RGB → degraded_latent (from base restoration pipeline pattern)
        rgb_latent = self.encode_rgb(rgb_in)  # [B, 4, h, w]

        # ControlNet conditioning: degraded RGB in pixel space (from ControlNet pipeline pattern)
        controlnet_cond = rgb_in  # [B, 3, H, W]

        # Initial noisy latent (random Gaussian noise)
        target_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

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
                # Conditional prediction: ControlNet residuals + degraded_latent concat
                down_block_res, mid_block_res = self.controlnet(
                    target_latent,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                cond_input = torch.cat(
                    [rgb_latent, target_latent], dim=1
                )  # [B, 8, h, w]

                noise_pred_cond = self.unet(
                    cond_input,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    down_block_additional_residuals=down_block_res,
                    mid_block_additional_residual=mid_block_res,
                ).sample  # [B, 4, h, w]

                # Unconditional prediction: zeroed ControlNet residuals + degraded_latent concat
                # UNet still sees degraded_latent — CFG only controls ControlNet contribution
                zero_down_block_res = [torch.zeros_like(r) for r in down_block_res]
                zero_mid_block_res = torch.zeros_like(mid_block_res)

                uncond_input = torch.cat(
                    [rgb_latent, target_latent], dim=1
                )  # [B, 8, h, w]

                noise_pred_uncond = self.unet(
                    uncond_input,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    down_block_additional_residuals=zero_down_block_res,
                    mid_block_additional_residual=zero_mid_block_res,
                ).sample  # [B, 4, h, w]

                # CFG combination: uncond + guidance_scale * (cond - uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                # --- Standard inference (no CFG) ---
                # ControlNet forward: get residuals from degraded RGB
                down_block_res, mid_block_res = self.controlnet(
                    target_latent,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                # 8ch UNet input: cat([degraded_latent, noisy_latent])
                unet_input = torch.cat(
                    [rgb_latent, target_latent], dim=1
                )  # [B, 8, h, w]

                noise_pred = self.unet(
                    unet_input,
                    t,
                    encoder_hidden_states=batch_text_embed,
                    down_block_additional_residuals=down_block_res,
                    mid_block_additional_residual=mid_block_res,
                ).sample  # [B, 4, h, w]

            # Compute the previous noisy sample x_t -> x_t-1
            step_output = self.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            )
            target_latent = step_output.prev_sample

        # Decode latent to RGB
        restored_rgb = self.decode_rgb(target_latent)  # [B, 3, H, W]

        # Clip to valid RGB range
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
        Invoke the hybrid ControlNet + degraded latent concatenation restoration pipeline.

        Pattern follows MarigoldControlNetRestorationPipeline.__call__() with
        hybrid single_infer().

        Args:
            input_image (`Image` or `torch.Tensor`):
                Input degraded RGB image.
            denoising_steps (`int`, *optional*):
                Number of denoising diffusion steps. None uses default.
            ensemble_size (`int`, *optional*, defaults to `1`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*):
                Processing resolution. 0 = original resolution with padding.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize prediction to match input resolution.
            resample_method (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method for resizing.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size. 0 = auto.
            generator (`torch.Generator`, *optional*):
                Random generator for noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display progress bar.
            ensemble_kwargs (`dict`, *optional*):
                Arguments for ensembling settings.
            guidance_scale (`float`, *optional*, defaults to `1.0`):
                Classifier-Free Guidance scale.

        Returns:
            `MarigoldRestorationOutput`: restored_np, restored_img, uncertainty.
        """
        # Default values from model config
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
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, 3, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize to square (matching training preprocessing)
        padding_info = None
        if processing_res > 0:
            rgb = resize_to_square(
                rgb,
                target_resolution=processing_res,
                resample_method=resample_method,
            )
        else:
            # Process at original resolution but pad to square
            rgb, padding_info = pad_to_square(rgb, padding_mode="reflect", min_size=768)

        # Normalize rgb values: [0, 255] -> [-1, 1]
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting restoration -----------------
        # Batch repeated input image
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

        # Predict restored images (batched)
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
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            # Simple averaging for restoration (no scale/shift invariance needed)
            final_pred = torch.mean(target_preds, dim=0, keepdim=True)
            pred_uncert = torch.std(target_preds, dim=0, keepdim=True)
        else:
            final_pred = target_preds
            pred_uncert = None

        # ----------------- Post-processing -----------------
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
        """
        Check if denoising step is reasonable.
        Pattern copied from MarigoldControlNetRestorationPipeline._check_inference_step().
        """
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
