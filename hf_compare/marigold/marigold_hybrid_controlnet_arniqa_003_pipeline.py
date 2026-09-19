# --------------------------------------------------------------------------
# Hybrid-003: Joint UNet + ControlNet + ARNIQA Conditioning Pipeline
# Thesis Project: Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction
#
# This pipeline extends the hybrid ControlNet pipeline with:
# 1. Latent-space conditioning: 8-channel UNet input [degraded_latent, noisy_latent]
# 2. Pixel-space conditioning: ControlNet receives degraded RGB and injects residuals
# 3. ARNIQA quality-aware conditioning: replaces empty text embedding with quality features
#
# Both UNet and ControlNet are trainable (joint training with differential LR).
# ARNIQA encoder is frozen; only adapters are trained.
#
# Based on marigold/marigold_hybrid_controlnet_restoration_pipeline.py
# --------------------------------------------------------------------------

import logging
import numpy as np
import torch
from torch.amp import autocast
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


class MarigoldHybridControlNetArniqa003Pipeline(DiffusionPipeline):
    """
    Pipeline for hybrid-003: joint UNet + ControlNet + ARNIQA conditioning.

    Combines an 8-channel UNet (trainable, from SD2 weights expanded to 8ch)
    with a trainable ControlNet module and optional ARNIQA quality-aware conditioning.

    When ARNIQA is enabled, quality features from the degraded image replace the
    empty text embedding in cross-attention, providing quality-aware guidance.
    When disabled, uses empty text embedding (same as hybrid-001/002).

    Based on MarigoldHybridControlNetRestorationPipeline.

    Args:
        unet (`UNet2DConditionModel`):
            8-channel UNet. Receives concatenated [degraded_latent(4ch), noisy_latent(4ch)].
        controlnet (`ControlNetModel`):
            ControlNet module for pixel-space conditioning.
        vae (`AutoencoderKL`):
            VAE for encoding/decoding images to/from latent space.
        scheduler (`DDIMScheduler` or `LCMScheduler`):
            Scheduler for the denoising process.
        text_encoder (`CLIPTextModel`):
            Text encoder for empty text embedding (used when ARNIQA is disabled).
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
        # Pattern from marigold_restoration_pipeline_base.py line 244
        self.arniqa_conditioner = None

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt.
        Pattern copied from MarigoldHybridControlNetRestorationPipeline.encode_empty_text().
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

    def set_arniqa_conditioner(self, conditioner):
        """
        Set the ARNIQA conditioner for quality-aware conditioning.
        Pattern from marigold_restoration_pipeline_base.py line 413-424.

        When set, ARNIQA features from the degraded image replace empty_text_embed
        in the UNet cross-attention, providing quality-aware guidance.

        Args:
            conditioner: ArniqaConditioner instance, or None to disable
        """
        self.arniqa_conditioner = conditioner

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        Pattern copied from MarigoldHybridControlNetRestorationPipeline.encode_rgb().
        """
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode RGB latent into RGB image.
        Pattern copied from MarigoldHybridControlNetRestorationPipeline.decode_rgb().
        """
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
        recycle_start_step: Optional[int] = None,
        recycle_interval: int = 1,
    ) -> torch.Tensor:
        """
        Perform a single hybrid restoration inference pass.

        Based on MarigoldHybridControlNetRestorationPipeline.single_infer()
        with ARNIQA conditioner support and optional ControlNet Feature Recycling.

        ControlNet Feature Recycling:
          Instead of always conditioning the ControlNet on the original degraded RGB,
          periodically decode the current predicted clean image (x0 estimate) and
          re-condition both the ControlNet (pixel-space) and the UNet 8ch input
          (latent-space) on this partially-restored image. As denoising progresses,
          the ControlNet receives increasingly clean input and can provide finer-
          grained guidance for detail recovery.

          Recycling is controlled by two parameters:
          - recycle_start_step: denoising step index (0-based) at which recycling
            begins. Earlier steps have high noise, so the x0 estimate is unreliable.
            Set to None or 0 to disable recycling entirely.
          - recycle_interval: how often to recycle after recycle_start_step.
            1 = every step, 2 = every other step, etc.

        At each denoising step:
          1. ControlNet receives (noisy_latent, t, text_embed, controlnet_cond) -> residuals
          2. UNet receives cat([rgb_latent, noisy_latent]) as 8ch input + residuals
          Where controlnet_cond and rgb_latent are either the original degraded image
          or the recycled partially-restored image.

        When ARNIQA is enabled, text_embed comes from ARNIQA conditioner instead of
        empty text embedding (pattern from marigold_restoration_pipeline_base.py line 481).

        For CFG (guidance_scale > 1.0):
          - Conditional: full ControlNet residuals + ARNIQA/text conditioning
          - Unconditional: zeroed ControlNet residuals + empty text embedding
          - CFG controls both ControlNet contribution and ARNIQA conditioning

        Args:
            rgb_in: Input degraded RGB image [B, 3, H, W] in [-1, 1].
            num_inference_steps: Number of DDIM denoising steps.
            generator: Random generator for noise.
            show_pbar: Display progress bar.
            guidance_scale: CFG scale. Values > 1.0 apply CFG.
            recycle_start_step: Step index to begin ControlNet recycling.
                None or 0 disables recycling (default behavior).
            recycle_interval: Recycle every N steps after recycle_start_step.
                1 = every step (default), 2 = every other step, etc.

        Returns:
            Predicted restored RGB image [B, 3, H, W] in [-1, 1].
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Determine if recycling is enabled
        recycle_enabled = (
            recycle_start_step is not None
            and recycle_start_step > 0
            and recycle_interval > 0
        )

        # Use autocast on CUDA to match training validation precision behavior.
        # Training validation (trainer line ~1565) wraps single_infer with autocast('cuda'),
        # which keeps model weights in fp32 but casts matmul/conv inputs to fp16,
        # while keeping softmax and layernorm in fp32. Without this, pure fp16 loading
        # causes precision loss in attention softmax and layernorm, producing
        # "painting/watercolor" artifacts with loss of fine texture detail.
        use_autocast = device.type == "cuda"
        autocast_ctx = autocast("cuda") if use_autocast else torch.inference_mode(False)

        with autocast_ctx:
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps  # [T]

            # Encode empty text embedding (always needed for CFG unconditional path)
            if self.empty_text_embed is None:
                self.encode_empty_text()

            # Conditioning: ARNIQA quality features or empty text embedding
            # Pattern from marigold_restoration_pipeline_base.py line 481-487
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

            # ControlNet conditioning: degraded RGB in pixel space
            controlnet_cond = rgb_in  # [B, 3, H, W]

            # Keep originals for potential blending or fallback
            rgb_latent_orig = rgb_latent
            controlnet_cond_orig = controlnet_cond

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
                # --- ControlNet Feature Recycling ---
                # At designated steps, decode the current x0 estimate to pixel space
                # and use it as the new conditioning for both ControlNet and UNet.
                if recycle_enabled and i >= recycle_start_step:
                    if (i - recycle_start_step) % recycle_interval == 0:
                        # Use pred_original_sample (x0 estimate) from the previous
                        # scheduler step. DDIM/LCM schedulers always compute this.
                        if i > 0 and hasattr(step_output, 'pred_original_sample') and step_output.pred_original_sample is not None:
                            pred_x0 = step_output.pred_original_sample

                            # Decode predicted x0 to pixel space
                            recycled_rgb = self.decode_rgb(pred_x0)
                            recycled_rgb = torch.clip(recycled_rgb, -1.0, 1.0)

                            # Update conditioning for this and subsequent steps
                            controlnet_cond = recycled_rgb
                            rgb_latent = self.encode_rgb(recycled_rgb)

                if guidance_scale > 1.0:
                    # --- Classifier-Free Guidance ---
                    # Conditional: ControlNet residuals + ARNIQA/text conditioning
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

                    # Unconditional: zeroed ControlNet residuals + empty text embedding
                    zero_down_block_res = [torch.zeros_like(r) for r in down_block_res]
                    zero_mid_block_res = torch.zeros_like(mid_block_res)

                    uncond_input = torch.cat(
                        [rgb_latent_orig, target_latent], dim=1
                    )  # [B, 8, h, w]

                    noise_pred_uncond = self.unet(
                        uncond_input,
                        t,
                        encoder_hidden_states=batch_empty_text_embed,
                        down_block_additional_residuals=zero_down_block_res,
                        mid_block_additional_residual=zero_mid_block_res,
                    ).sample  # [B, 4, h, w]

                    # CFG combination
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
        recycle_start_step: Optional[int] = None,
        recycle_interval: int = 1,
    ) -> MarigoldRestorationOutput:
        """
        Invoke the hybrid-003 restoration pipeline.

        Copied from MarigoldHybridControlNetRestorationPipeline.__call__()
        (marigold_hybrid_controlnet_restoration_pipeline.py lines 317-500).

        Additional args for ControlNet Feature Recycling:
            recycle_start_step: Step index to begin recycling. None or 0 disables.
            recycle_interval: Recycle every N steps after start. Default 1.
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
                recycle_start_step=recycle_start_step,
                recycle_interval=recycle_interval,
            )
            target_pred_ls.append(target_pred_raw.detach())
        target_preds = torch.concat(target_pred_ls, dim=0)
        torch.cuda.empty_cache()

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            final_pred = torch.mean(target_preds, dim=0, keepdim=True)
            pred_uncert = torch.std(target_preds, dim=0, keepdim=True)
        else:
            final_pred = target_preds
            pred_uncert = None

        # ----------------- Post-processing -----------------
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
        Copied from MarigoldHybridControlNetRestorationPipeline._check_inference_step().
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
