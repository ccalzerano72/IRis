# MarigoldControlNetTrainableUnetTrainer - Thesis Implementation
#
# Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction:
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# 4ch Trainable UNet + ControlNet (Ablation Experiment re_004)
# - Standard 4ch UNet (SD2 weights, TRAINABLE with low LR)
# - ControlNet (from SD2 UNet, TRAINABLE with higher LR)
# - UNet receives only noisy_latents (4ch) — NO degraded latent concatenation
# - ControlNet receives degraded RGB in pixel space (3ch)
#
# Purpose: Isolate the contribution of ControlNet pixel-space conditioning
# WITHOUT the 8-channel latent concatenation used in hybrid architectures.
# See thesis-docs/notes/re-004-controlnet-trainable-unet-rationale.md
#
# Derived from: marigold_hybrid_controlnet_arniqa_003_trainer.py
# Removed: 8ch conv_in expansion, ARNIQA conditioning, CFG dropout,
#          degraded latent concatenation in forward pass
# --------------------------------------------------------------------------

import gc
import json
import logging
import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from datetime import datetime
from diffusers import ControlNetModel, DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from typing import List, Union

from marigold.marigold_controlnet_restoration_pipeline import (
    MarigoldControlNetRestorationPipeline,
    MarigoldRestorationOutput,
)
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential, CosineAnnealingWarmRestarts
from src.util.metric import MetricTracker
from src.util.seeding import generate_seed_sequence


class MarigoldControlNetTrainableUnetTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldControlNetRestorationPipeline,
        train_dataloader: DataLoader,
        device,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: MarigoldControlNetRestorationPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # NO conv_in replacement — UNet stays at native 4 channels.
        # This is the key architectural difference from hybrid trainers.

        # Encode empty text prompt
        # Verified: MarigoldControlNetRestorationPipeline.encode_empty_text()
        # sets self.empty_text_embed (pipeline line 101-115)
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        # XFormers memory-efficient attention for both UNet and ControlNet
        # Verified: both inherit from ModelMixin (line 87-89 of controlnet trainer)
        self.model.unet.enable_xformers_memory_efficient_attention()
        self.model.controlnet.enable_xformers_memory_efficient_attention()

        # Gradient checkpointing
        # Pattern from hybrid-003 trainer __init__ (line 97-102)
        self.gradient_checkpointing = self.cfg.trainer.get('gradient_checkpointing', True)
        if self.gradient_checkpointing:
            self.model.unet.enable_gradient_checkpointing()
            self.model.controlnet.enable_gradient_checkpointing()
            logging.info("Gradient checkpointing ENABLED for both UNet and ControlNet")
        else:
            logging.info("Gradient checkpointing disabled")

        # Trainability — UNet + ControlNet trainable, VAE + text_encoder frozen
        # Pattern from hybrid-003 trainer __init__ (line 107-113)
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)
        self.model.controlnet.requires_grad_(True)

        unet_params = sum(p.numel() for p in self.model.unet.parameters()) / 1e6
        controlnet_params = sum(p.numel() for p in self.model.controlnet.parameters()) / 1e6
        logging.info(
            f"Trainability: UNet=TRAINABLE ({unet_params:.1f}M), "
            f"ControlNet=TRAINABLE ({controlnet_params:.1f}M), "
            f"VAE=frozen, text_encoder=frozen"
        )

        # Reinitialize ControlNet zero convolutions (optional)
        # Pattern from hybrid-003 trainer __init__ (line 119-140)
        zero_conv_init_scale = self.cfg.get('controlnet_zero_conv_init_scale', 0.0)
        if zero_conv_init_scale > 0.0:
            with torch.no_grad():
                for i, block in enumerate(self.model.controlnet.controlnet_down_blocks):
                    torch.nn.init.normal_(block.weight, std=zero_conv_init_scale)
                    torch.nn.init.zeros_(block.bias)
                torch.nn.init.normal_(
                    self.model.controlnet.controlnet_mid_block.weight,
                    std=zero_conv_init_scale,
                )
                torch.nn.init.zeros_(self.model.controlnet.controlnet_mid_block.bias)
            logging.info(
                f"ControlNet zero convolutions reinitialized: "
                f"weights ~ N(0, {zero_conv_init_scale}), biases = 0"
            )
        else:
            logging.info("ControlNet zero convolutions kept at default (exact zero)")

        # Optimizer — 2-way differential LR: UNet (low) + ControlNet (high)
        # Pattern from hybrid-003 trainer __init__ (lines 181-218), without ARNIQA
        lr = self.cfg.lr  # UNet LR
        controlnet_lr = self.cfg.get('controlnet_lr', lr)  # ControlNet LR

        param_groups = [
            {'params': list(self.model.unet.parameters()), 'lr': lr, 'name': 'unet'},
            {'params': list(self.model.controlnet.parameters()), 'lr': controlnet_lr, 'name': 'controlnet'},
        ]
        self.optimizer = Adam(param_groups)
        logging.info(f"Optimizer: Adam, UNet LR={lr}, ControlNet LR={controlnet_lr}")

        # LR scheduler
        # Pattern from hybrid-003 trainer __init__ (lines 221-244)
        scheduler_name = self.cfg.lr_scheduler.name
        scheduler_kwargs = self.cfg.lr_scheduler.kwargs

        if scheduler_name == "IterExponential":
            lr_func = IterExponential(
                total_iter_length=scheduler_kwargs.total_iter,
                final_ratio=scheduler_kwargs.final_ratio,
                warmup_steps=scheduler_kwargs.warmup_steps,
            )
        elif scheduler_name == "CosineAnnealingWarmRestarts":
            lr_func = CosineAnnealingWarmRestarts(
                T_0=scheduler_kwargs.T_0,
                T_mult=scheduler_kwargs.T_mult,
                eta_min_ratio=scheduler_kwargs.eta_min_ratio,
                warmup_steps=scheduler_kwargs.warmup_steps,
                total_iter_length=scheduler_kwargs.total_iter_length,
            )
        else:
            raise ValueError(
                f"Unknown scheduler: {scheduler_name}. "
                f"Supported: IterExponential, CosineAnnealingWarmRestarts"
            )

        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Mixed Precision Training (FP16)
        self.scaler = GradScaler()
        logging.info("Mixed precision training (FP16) enabled via GradScaler")

        # Loss function
        # Pattern from hybrid-003 trainer __init__ (lines 251-255)
        trainer_only_params = {'lpips_weight', 'pixel_loss_weight', 'pixel_loss_type', 'pixel_loss_max_timestep', 'pixel_loss_max_value'}
        loss_kwargs = {k: v for k, v in self.cfg.loss.kwargs.items()
                       if v is not None and k not in trainer_only_params}
        logging.info(f"Loss kwargs: {loss_kwargs}")
        self.loss = get_loss(loss_name=self.cfg.loss.name, **loss_kwargs)

        # Pixel-space LPIPS loss (optional)
        # Pattern from hybrid-003 trainer __init__ (lines 258-273)
        self.lpips_weight = self.cfg.loss.kwargs.get('lpips_weight', 0.0)
        self.pixel_loss_crop_size = self.cfg.trainer.get('pixel_loss_crop_size', 128)
        if self.lpips_weight > 0:
            import lpips
            self.lpips_loss = lpips.LPIPS(net='alex').to(device)
            for param in self.lpips_loss.parameters():
                param.requires_grad = False
            self.lpips_loss.eval()
            logging.info(
                f"Pixel-space LPIPS ENABLED: weight={self.lpips_weight}, "
                f"crop_size={self.pixel_loss_crop_size}px "
                f"({self.pixel_loss_crop_size // 8}x{self.pixel_loss_crop_size // 8} latent)"
            )
        else:
            self.lpips_loss = None
            logging.info("Pixel-space LPIPS disabled (lpips_weight=0)")

        # Pixel-space reconstruction loss (optional, reuses LPIPS crop)
        # Pattern from hybrid-003 trainer __init__ (lines 276-289)
        self.pixel_loss_weight = self.cfg.loss.kwargs.get('pixel_loss_weight', 0.0)
        self.pixel_loss_type = self.cfg.loss.kwargs.get('pixel_loss_type', 'l1')
        if self.pixel_loss_weight > 0:
            if self.pixel_loss_type == 'l1':
                self.pixel_reconstruction_loss = torch.nn.L1Loss(reduction='mean')
            elif self.pixel_loss_type == 'l2':
                self.pixel_reconstruction_loss = torch.nn.MSELoss(reduction='mean')
            else:
                raise ValueError(f"Unknown pixel_loss_type: {self.pixel_loss_type}. Use 'l1' or 'l2'.")
            logging.info(
                f"Pixel-space reconstruction loss ENABLED: type={self.pixel_loss_type}, "
                f"weight={self.pixel_loss_weight} (reuses LPIPS crop)"
            )
        else:
            self.pixel_reconstruction_loss = None
            logging.info("Pixel-space reconstruction loss disabled (pixel_loss_weight=0)")

        # Pixel-space loss timestep threshold
        # Pattern from hybrid-003 trainer __init__ (lines 292-300)
        self.pixel_loss_max_timestep = self.cfg.loss.kwargs.get('pixel_loss_max_timestep', 1000)
        if self.pixel_loss_max_timestep < 1000:
            logging.info(
                f"Pixel-space loss timestep threshold: {self.pixel_loss_max_timestep} "
                f"(losses disabled above this timestep)"
            )
        else:
            logging.info("Pixel-space loss timestep threshold: disabled (no limit)")

        # Pixel-space loss value clamping
        # Pattern from hybrid-003 trainer __init__ (lines 303-310)
        self.pixel_loss_max_value = self.cfg.loss.kwargs.get('pixel_loss_max_value', None)
        if self.pixel_loss_max_value is not None:
            logging.info(
                f"Pixel-space loss value clamping ENABLED: max={self.pixel_loss_max_value}"
            )
        else:
            logging.info("Pixel-space loss value clamping: disabled (no limit)")

        # Training noise scheduler
        # Pattern from hybrid-003 trainer __init__ (lines 313-337)
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(
            self.model.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
        logging.info(
            "DDPM training noise scheduler config: "
            f"rescale_betas_zero_snr = {self.training_noise_scheduler.config.rescale_betas_zero_snr}, "
            f"timestep_spacing = {self.training_noise_scheduler.config.timestep_spacing}"
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Inference DDIM scheduler (used for validation)
        self.model.scheduler = DDIMScheduler.from_config(
            self.training_noise_scheduler.config,
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        # Train and validation metrics
        train_metric_keys = ["loss"]
        if self.lpips_weight > 0:
            train_metric_keys.append("pixel_lpips")
        if self.pixel_loss_weight > 0:
            train_metric_keys.append("pixel_reconstruction")
        self.train_metrics = MetricTracker(*train_metric_keys)

        val_metric_keys = [m.__name__ for m in self.metric_funcs] + ["val_loss"]
        if self.lpips_weight > 0:
            val_metric_keys.append("val_pixel_lpips")
        self.val_metrics = MetricTracker(*val_metric_keys)

        # Main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.degraded_rgb_type = getattr(self.cfg, 'degraded_rgb_type', 'degraded_rgb_norm')
        self.clean_rgb_type = getattr(self.cfg, 'clean_rgb_type', 'clean_rgb_norm')
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Offset noise
        offset_noise_cfg = self.cfg.get('offset_noise', {})
        self.offset_noise_strength = offset_noise_cfg.get('strength', 0.0)
        if self.offset_noise_strength > 0.0:
            logging.info(f"Offset noise ENABLED - strength: {self.offset_noise_strength}")
        else:
            logging.info("Offset noise disabled (strength = 0.0)")

        # Input noise augmentation (pixel space for ControlNet conditioning)
        input_noise_cfg = self.cfg.get('input_noise_augmentation', {})
        self.input_noise_prob = input_noise_cfg.get('probability', 0.0)
        self.input_noise_max_strength = input_noise_cfg.get('max_strength', 0.1)
        if self.input_noise_prob > 0.0:
            logging.info(
                f"Input noise augmentation ENABLED (pixel space) - "
                f"probability: {self.input_noise_prob:.1%}, "
                f"max_strength: {self.input_noise_max_strength}"
            )
        else:
            logging.info("Input noise augmentation disabled (probability = 0.0)")

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0
        self.effective_iter = 0
        self.in_evaluation = False
        self.global_seed_sequence: List = []

        # Checkpoint strategy
        checkpoint_strategy = getattr(cfg.trainer, 'checkpoint_strategy', None)
        self.checkpoint_mode = checkpoint_strategy.get('mode', 'marigold') if checkpoint_strategy else 'marigold'
        if self.checkpoint_mode not in ['marigold', 'decoupled']:
            logging.warning(f"Unknown checkpoint mode '{self.checkpoint_mode}', using 'marigold'")
            self.checkpoint_mode = 'marigold'
        logging.info(f"Checkpoint strategy mode: {self.checkpoint_mode}")

        self.save_trainer_state = self.cfg.trainer.get('save_trainer_state', True)
        logging.info(f"Save trainer state in checkpoints: {self.save_trainer_state}")

        self.checkpoint_test_config = getattr(cfg, 'checkpoint_test', None)
        if self.checkpoint_test_config:
            logging.info("Space-optimized checkpoint settings enabled")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def encode_rgb(self, image_in):
        """Encode RGB image to latent space.
        Copied from controlnet trainer.encode_rgb (line 281).
        Cast to VAE dtype because VAE is float16 (frozen)."""
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        image_in = image_in.to(self.model.vae.dtype)
        latent = self.model.encode_rgb(image_in)
        return latent

    def decode_rgb(self, latent_in):
        """Decode latent to RGB image [0, 1].
        Copied from controlnet trainer.decode_rgb (line 291)."""
        assert len(latent_in.shape) == 4 and latent_in.shape[1] == 4
        latent_in = latent_in.to(self.model.vae.dtype)
        rgb = self.model.decode_rgb(latent_in)
        rgb = (rgb + 1.0) / 2.0
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb

    def _get_next_seed(self):
        """Get next seed from global sequence.
        Copied from controlnet trainer._get_next_seed (line 303)."""
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, t_end=None):
        """Training loop for 4ch trainable UNet + ControlNet.

        Based on hybrid-003 trainer.train() with key differences:
        - NO degraded latent encoding (UNet stays 4ch)
        - NO degraded latent concatenation (UNet receives noisy_latents only)
        - NO ARNIQA conditioning (empty text embedding only)
        - NO CFG dropout
        - Forward pass: ControlNet(noisy_latents, degraded_rgb) → residuals →
          UNet(noisy_latents, residuals)
        """
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            if self.checkpoint_mode == 'marigold':
                logging.info(
                    "Last evaluation was not finished, will do evaluation before continue training."
                )
                self.validate()
            else:
                logging.info(
                    "Last evaluation was not finished, but in decoupled mode we skip it on resume."
                )
                self.in_evaluation = False

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Update epoch in dataset for deterministic-variable degradations
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(self.epoch)
            elif hasattr(self.train_loader.dataset, 'datasets'):
                for dataset in self.train_loader.dataset.datasets:
                    if hasattr(dataset, 'set_epoch'):
                        dataset.set_epoch(self.epoch)

            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                # Both UNet and ControlNet in train mode
                self.model.unet.train()
                self.model.controlnet.train()

                # Globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # Get data
                degraded_rgb = batch[self.degraded_rgb_type].to(device)  # [B, 3, H, W] in [-1, 1]
                clean_rgb = batch[self.clean_rgb_type].to(device)  # [B, 3, H, W] in [-1, 1]
                batch_size = degraded_rgb.shape[0]

                with torch.no_grad():
                    # Encode clean RGB to latent (target for loss)
                    clean_latent = self.encode_rgb(clean_rgb)  # [B, 4, h, w]
                    # NO degraded latent encoding — 4ch UNet doesn't need it

                # Sample random timestep
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()

                # Sample noise
                noise = torch.randn(
                    clean_latent.shape,
                    device=device,
                    generator=rand_num_generator,
                )

                # Offset noise
                if self.offset_noise_strength > 0.0:
                    offset = torch.randn(
                        batch_size, clean_latent.shape[1], 1, 1,
                        device=device,
                        generator=rand_num_generator,
                    )
                    noise = noise + self.offset_noise_strength * offset

                # Add noise to clean latents
                noisy_latents = self.training_noise_scheduler.add_noise(
                    clean_latent, noise, timesteps
                )

                # Text embedding (empty text — no ARNIQA)
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )

                # Input noise augmentation on degraded RGB (pixel space)
                controlnet_cond = degraded_rgb
                if self.input_noise_prob > 0.0:
                    augment_mask = (
                        torch.rand(batch_size, device=device, generator=rand_num_generator)
                        < self.input_noise_prob
                    )
                    if augment_mask.any():
                        input_noise_strength = (
                            torch.rand(batch_size, device=device, generator=rand_num_generator)
                            * self.input_noise_max_strength
                        )
                        input_noise_strength = input_noise_strength * augment_mask.float()
                        input_noise_strength = input_noise_strength.view(batch_size, 1, 1, 1)
                        pixel_noise = torch.randn(
                            controlnet_cond.shape,
                            device=device,
                            generator=rand_num_generator,
                        )
                        controlnet_cond = controlnet_cond + input_noise_strength * pixel_noise
                        controlnet_cond = torch.clamp(controlnet_cond, -1.0, 1.0)

                # Forward pass with FP16 mixed precision
                with autocast('cuda'):
                    # ControlNet forward: degraded RGB → residuals
                    down_block_res, mid_block_res = self.model.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embed,
                        controlnet_cond=controlnet_cond,
                        return_dict=False,
                    )

                    # 4ch UNet forward: noisy_latents only (NO degraded latent concat)
                    model_pred = self.model.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embed,
                        down_block_additional_residuals=down_block_res,
                        mid_block_additional_residual=mid_block_res,
                    ).sample  # [B, 4, h, w]

                    # Get target
                    if "sample" == self.prediction_type:
                        target = clean_latent
                    elif "epsilon" == self.prediction_type:
                        target = noise
                    elif "v_prediction" == self.prediction_type:
                        target = self.training_noise_scheduler.get_velocity(
                            clean_latent, noise, timesteps
                        )
                    else:
                        raise ValueError(f"Unknown prediction type {self.prediction_type}")

                # Loss computation in FP32
                latent_loss = self.loss(model_pred.float(), target.float())
                loss = latent_loss.mean()

                # NaN guard
                if torch.isnan(loss).any() or torch.isnan(model_pred).any():
                    logging.warning(
                        f"NaN detected at iter {self.effective_iter + 1}, "
                        f"skipping backward pass."
                    )
                    self.optimizer.zero_grad()
                    accumulated_step = 0
                    self.n_batch_in_epoch += 1
                    continue

                self.train_metrics.update("loss", loss.item())

                # Optional pixel-space losses (LPIPS and/or L1/L2)
                # Pattern from hybrid-003 trainer (lines 780-860)
                _use_lpips = self.lpips_loss is not None and self.lpips_weight > 0
                _use_pixel_recon = self.pixel_reconstruction_loss is not None and self.pixel_loss_weight > 0

                if _use_lpips or _use_pixel_recon:
                    _t_mask = timesteps <= self.pixel_loss_max_timestep
                    _n_valid = _t_mask.sum().item()

                    if _n_valid > 0:
                        _valid_noisy = noisy_latents[_t_mask]
                        _valid_pred = model_pred[_t_mask]
                        _valid_clean_lat = clean_latent[_t_mask]
                        _valid_t = timesteps[_t_mask]

                        # Recover predicted clean latent
                        if "epsilon" == self.prediction_type:
                            alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[_valid_t]
                            beta_prod_t = 1 - alpha_prod_t
                            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                            beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                            pred_original_sample = (
                                _valid_noisy - beta_prod_t.sqrt() * _valid_pred
                            ) / alpha_prod_t.sqrt()
                        elif "sample" == self.prediction_type:
                            pred_original_sample = _valid_pred
                        elif "v_prediction" == self.prediction_type:
                            alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[_valid_t]
                            beta_prod_t = 1 - alpha_prod_t
                            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                            beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                            pred_original_sample = (
                                alpha_prod_t.sqrt() * _valid_noisy
                                - beta_prod_t.sqrt() * _valid_pred
                            )
                        else:
                            raise ValueError(f"Unknown prediction type {self.prediction_type}")

                        pred_original_sample = pred_original_sample.clamp(-6.0, 6.0)

                        # Random crop in latent space
                        latent_crop = self.pixel_loss_crop_size // 8
                        _, _, lh, lw = pred_original_sample.shape

                        if latent_crop < lh and latent_crop < lw:
                            top = torch.randint(
                                0, lh - latent_crop, (1,),
                                generator=rand_num_generator, device=device,
                            ).item()
                            left = torch.randint(
                                0, lw - latent_crop, (1,),
                                generator=rand_num_generator, device=device,
                            ).item()
                        else:
                            top, left = 0, 0

                        pred_crop = pred_original_sample[
                            :, :, top:top + latent_crop, left:left + latent_crop
                        ]
                        clean_crop = _valid_clean_lat[
                            :, :, top:top + latent_crop, left:left + latent_crop
                        ]

                        # VAE decode crops to pixel space
                        pred_rgb_crop = self.decode_rgb(pred_crop.float())
                        clean_rgb_crop = self.decode_rgb(clean_crop.float())

                        # LPIPS
                        if _use_lpips:
                            pred_lpips_in = pred_rgb_crop * 2.0 - 1.0
                            clean_lpips_in = clean_rgb_crop * 2.0 - 1.0
                            pixel_lpips = self.lpips_loss(pred_lpips_in, clean_lpips_in).mean()
                            if self.pixel_loss_max_value is not None:
                                pixel_lpips = pixel_lpips.clamp(max=self.pixel_loss_max_value)
                            if torch.isfinite(pixel_lpips):
                                loss = loss + self.lpips_weight * pixel_lpips
                                self.train_metrics.update("pixel_lpips", pixel_lpips.item())

                        # Pixel reconstruction loss
                        if _use_pixel_recon:
                            pixel_recon = self.pixel_reconstruction_loss(pred_rgb_crop, clean_rgb_crop)
                            if self.pixel_loss_max_value is not None:
                                pixel_recon = pixel_recon.clamp(max=self.pixel_loss_max_value)
                            if torch.isfinite(pixel_recon):
                                loss = loss + self.pixel_loss_weight * pixel_recon
                                self.train_metrics.update("pixel_reconstruction", pixel_recon.item())

                loss = loss / self.gradient_accumulation_steps

                # Loss clipping (same threshold as hybrid-003)
                max_loss = 1.0
                if loss.item() > max_loss:
                    logging.warning(
                        f"Loss clipped from {loss.item():.4f} to {max_loss} "
                        f"at iter {self.effective_iter + 1}"
                    )
                    loss = loss.clamp(max=max_loss)

                # Backward pass
                self.scaler.scale(loss).backward()

                accumulated_step += 1
                self.n_batch_in_epoch += 1

                # Optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.unet.parameters(), max_norm=1.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.controlnet.parameters(), max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dict(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    self._train_step_callback()

                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

            self.n_batch_in_epoch = 0

    # ------------------------------------------------------------------
    # Callbacks (copied from hybrid-003 trainer)
    # ------------------------------------------------------------------

    def _train_step_callback(self):
        if self.checkpoint_mode == 'marigold':
            self._train_step_callback_marigold()
        else:
            self._train_step_callback_decoupled()

    def _train_step_callback_marigold(self):
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )
        _is_latest_saved = False
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def _train_step_callback_decoupled(self):
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )
        if self.save_period > 0 and 0 == self.effective_iter % self.save_period:
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.validate_decoupled()
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, ckpt_name, save_train_state):
        """Save checkpoint: UNet + ControlNet (both trainable).
        Pattern from hybrid-003 trainer.save_checkpoint, without ARNIQA."""
        if self.checkpoint_test_config and self.checkpoint_test_config.get('auto_cleanup', False):
            self._check_disk_space_and_cleanup()

        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")

        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            if self.checkpoint_test_config and self.checkpoint_test_config.get('keep_only_latest_best', False):
                logging.info(f"Removing old checkpoint: {ckpt_dir}")
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            else:
                temp_ckpt_dir = os.path.join(
                    os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
                )
                if os.path.exists(temp_ckpt_dir):
                    shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
                os.rename(ckpt_dir, temp_ckpt_dir)
                logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet (trainable)
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=True)
        logging.info(f"UNet is saved to: {unet_path}")

        # Save ControlNet (trainable)
        controlnet_path = os.path.join(ckpt_dir, "controlnet")
        self.model.controlnet.save_pretrained(controlnet_path, safe_serialization=True)
        logging.info(f"ControlNet is saved to: {controlnet_path}")

        # Save scheduler
        if not (self.checkpoint_test_config and self.checkpoint_test_config.get('save_unet_only', False)):
            scheduler_path = os.path.join(ckpt_dir, "scheduler")
            self.model.scheduler.save_pretrained(scheduler_path)
            logging.info(f"Scheduler is saved to: {scheduler_path}")

        # Save architecture config for reproducibility
        arch_config = {
            "architecture": "controlnet_trainable_unet_4ch",
            "unet_channels": 4,
            "unet_trainable": True,
            "controlnet_trainable": True,
            "latent_concatenation": False,
        }
        config_path = os.path.join(ckpt_dir, "architecture_config.json")
        with open(config_path, "w") as f:
            json.dump(arch_config, f, indent=2)
        logging.info(f"Architecture config saved to: {config_path}")

        if save_train_state and self.save_trainer_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
                "scaler": self.scaler.state_dict(),
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            logging.info(f"Trainer state is saved to: {train_state_path}")

        if save_train_state:
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

        if self.checkpoint_test_config:
            self._log_checkpoint_info(ckpt_dir)
        if self.checkpoint_test_config and self.checkpoint_test_config.get('keep_only_latest_best', False):
            self._cleanup_old_checkpoints(ckpt_name)

        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True):
        """Load checkpoint: UNet + ControlNet + trainer state.
        Pattern from hybrid-003 trainer.load_checkpoint, without ARNIQA."""
        logging.info(f"Loading checkpoint from: {ckpt_path}")

        # Load UNet weights
        _unet_path = os.path.join(
            ckpt_path, "unet", "diffusion_pytorch_model.safetensors"
        )
        if os.path.isfile(_unet_path):
            from safetensors.torch import load_file
            unet_state_dict = load_file(_unet_path)
            self.model.unet.load_state_dict(unet_state_dict)
            self.model.unet.to(self.device)
            logging.info(f"UNet parameters loaded from {_unet_path}")
        else:
            logging.warning(f"UNet weights not found at {_unet_path}, skipping UNet load")

        # Load ControlNet weights
        _controlnet_path = os.path.join(
            ckpt_path, "controlnet", "diffusion_pytorch_model.safetensors"
        )
        if os.path.isfile(_controlnet_path):
            from safetensors.torch import load_file
            controlnet_state_dict = load_file(_controlnet_path)
            self.model.controlnet.load_state_dict(controlnet_state_dict)
            self.model.controlnet.to(self.device)
            logging.info(f"ControlNet parameters loaded from {_controlnet_path}")
        else:
            logging.warning(f"ControlNet weights not found at {_controlnet_path}, skipping ControlNet load")

        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]
            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

            if "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
                logging.info(f"Mixed precision scaler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. "
            f"Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )

    def _check_disk_space_and_cleanup(self):
        """Copied from hybrid-003 trainer."""
        if not self.checkpoint_test_config:
            return
        try:
            import shutil as disk_util
            total, used, free = disk_util.disk_usage(self.out_dir_ckpt)
            free_gb = free / (1024**3)
            min_free_gb = self.checkpoint_test_config.get('min_free_space_gb', 5.0)
            if free_gb < min_free_gb:
                logging.warning(f"Low disk space: {free_gb:.1f} GB free (minimum: {min_free_gb} GB)")
                if self.checkpoint_test_config.get('auto_cleanup', False):
                    self._cleanup_old_checkpoints()
                    total, used, free = disk_util.disk_usage(self.out_dir_ckpt)
                    free_gb = free / (1024**3)
                    if free_gb < min_free_gb:
                        logging.error(f"Still low on space after cleanup: {free_gb:.1f} GB")
                    else:
                        logging.info(f"Cleanup successful: {free_gb:.1f} GB free")
        except Exception as e:
            logging.warning(f"Could not check disk space: {e}")

    def _log_checkpoint_info(self, ckpt_dir):
        """Copied from hybrid-003 trainer."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(ckpt_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            size_gb = total_size / (1024**3)
            import shutil as disk_util
            total, used, free = disk_util.disk_usage(self.out_dir_ckpt)
            free_gb = free / (1024**3)
            logging.info(f"Checkpoint saved: {os.path.basename(ckpt_dir)}")
            logging.info(f"  - Size: {size_gb:.2f} GB")
            logging.info(f"  - Free space: {free_gb:.1f} GB")
        except Exception as e:
            logging.warning(f"Could not log checkpoint info: {e}")

    def _cleanup_old_checkpoints(self, current_ckpt_name=None):
        """Copied from hybrid-003 trainer."""
        if not self.checkpoint_test_config or not self.checkpoint_test_config.get('keep_only_latest_best', False):
            return
        try:
            max_checkpoints = self.checkpoint_test_config.get('max_checkpoints', 2)
            ckpt_dirs = []
            for item in os.listdir(self.out_dir_ckpt):
                item_path = os.path.join(self.out_dir_ckpt, item)
                if os.path.isdir(item_path) and item.startswith('iter_'):
                    ckpt_dirs.append(item)
            ckpt_dirs.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
            keep_dirs = set()
            if current_ckpt_name:
                keep_dirs.add(current_ckpt_name)
            elif ckpt_dirs:
                keep_dirs.add(ckpt_dirs[0])
            if hasattr(self, 'best_iter') and self.best_iter:
                best_ckpt_name = f"iter_{self.best_iter:06d}"
                keep_dirs.add(best_ckpt_name)
            removed_count = 0
            for ckpt_dir in ckpt_dirs:
                if ckpt_dir not in keep_dirs and len(keep_dirs) + removed_count < max_checkpoints:
                    ckpt_path = os.path.join(self.out_dir_ckpt, ckpt_dir)
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(ckpt_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total_size += os.path.getsize(filepath)
                    size_gb = total_size / (1024**3)
                    shutil.rmtree(ckpt_path, ignore_errors=True)
                    logging.info(f"Removed old checkpoint: {ckpt_dir} ({size_gb:.2f} GB)")
                    removed_count += 1
            if removed_count > 0:
                logging.info(f"Cleanup completed: removed {removed_count} old checkpoints")
        except Exception as e:
            logging.warning(f"Could not cleanup old checkpoints: {e}")

    # ------------------------------------------------------------------
    # Validation and visualization
    # ------------------------------------------------------------------

    def validate(self):
        """Validation with checkpoint saving (Marigold mode).
        Copied from hybrid-003 trainer.validate."""
        log_images = getattr(self.cfg.validation, 'log_images_during_validation', True)
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dict = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                log_images_to_wandb=log_images,
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
            )
            tb_logger.log_dict(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                global_step=self.effective_iter,
            )
            eval_text = eval_dict_to_text(
                val_metrics=val_metric_dict,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)
            if 0 == i:
                main_eval_metric = val_metric_dict[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def validate_decoupled(self):
        """Validation without checkpoint saving (Decoupled mode).
        Copied from hybrid-003 trainer.validate_decoupled."""
        log_images = getattr(self.cfg.validation, 'log_images_during_validation', True)
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dict = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                log_images_to_wandb=log_images,
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
            )
            tb_logger.log_dict(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                global_step=self.effective_iter,
            )
            eval_text = eval_dict_to_text(
                val_metrics=val_metric_dict,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)
            if 0 == i:
                main_eval_metric = val_metric_dict[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
        log_images_to_wandb: bool = False,
    ):
        """Validate on a single dataset.
        Based on hybrid-003 trainer.validate_single_dataset.
        Uses MarigoldControlNetRestorationPipeline.single_infer() which handles
        ControlNet conditioning internally (4ch UNet, no degraded latent concat).
        """
        self.model.to(self.device)
        metric_tracker.reset()

        gc.collect()
        torch.cuda.empty_cache()

        val_init_seed = self.cfg.validation.init_seed
        total_images = len(data_loader.dataset)
        val_seed_ls = generate_seed_sequence(val_init_seed, total_images)

        val_loss_dicts = []
        max_images_to_log = getattr(self.cfg.validation, 'max_images_to_log', 8)
        wandb_images = []
        wandb_image_metrics = []

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            batch_size = batch["degraded_rgb_int"].shape[0]
            degraded_rgb_int = batch["degraded_rgb_int"]
            clean_rgb_int = batch["clean_rgb_int"]

            batch_seeds = [val_seed_ls.pop() for _ in range(batch_size)]

            for b_idx in range(batch_size):
                batch_single = {
                    k: v[b_idx:b_idx+1] if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v
                    for k, v in batch.items()
                }
                seed = batch_seeds[b_idx]
                generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
                val_loss_dict = self._calculate_validation_loss(batch_single, generator)
                val_loss_dicts.append(val_loss_dict)

            degraded_rgb_norm = degraded_rgb_int.float() / 255.0 * 2.0 - 1.0
            degraded_rgb_norm = degraded_rgb_norm.to(self.device)

            self.model.scheduler.set_timesteps(
                self.cfg.validation.denoising_steps, device=self.device
            )

            # single_infer handles 4ch ControlNet forward pass internally
            # Verified: MarigoldControlNetRestorationPipeline.single_infer (line 151-292)
            with autocast('cuda'):
                restored_rgb_batch_ts = self.model.single_infer(
                    rgb_in=degraded_rgb_norm,
                    num_inference_steps=self.cfg.validation.denoising_steps,
                    generator=None,
                    show_pbar=False,
                )

            restored_rgb_batch_ts = (restored_rgb_batch_ts + 1.0) / 2.0
            restored_rgb_batch_ts = torch.clip(restored_rgb_batch_ts, 0.0, 1.0)
            restored_rgb_batch = restored_rgb_batch_ts.detach().cpu().numpy()

            for b_idx in range(batch_size):
                restored_rgb_ts = torch.from_numpy(restored_rgb_batch[b_idx]).to(self.device)
                clean_single_ts = clean_rgb_int[b_idx].to(self.device).float() / 255.0

                sample_metric_dict = {}
                has_nan = torch.isnan(restored_rgb_ts).any()
                if has_nan:
                    logging.warning(
                        f"NaN detected in restored image (batch {i}, idx {b_idx}), "
                        f"skipping metrics."
                    )
                    restored_rgb_ts = torch.nan_to_num(restored_rgb_ts, nan=0.0)
                    restored_rgb_batch[b_idx] = restored_rgb_ts.cpu().numpy()

                if not has_nan:
                    for met_func in self.metric_funcs:
                        _metric_name = met_func.__name__
                        _metric = met_func(restored_rgb_ts, clean_single_ts)
                        sample_metric_dict[_metric_name] = float(_metric)
                        metric_tracker.update(_metric_name, _metric)

                if self.lpips_loss is not None and self.lpips_weight > 0:
                    pred_lpips_in = restored_rgb_ts.unsqueeze(0) * 2.0 - 1.0
                    clean_lpips_in = clean_single_ts.unsqueeze(0) * 2.0 - 1.0
                    with torch.no_grad():
                        pixel_lpips_val = self.lpips_loss(pred_lpips_in, clean_lpips_in).mean()
                    metric_tracker.update("val_pixel_lpips", pixel_lpips_val.item())

                if save_to_dir is not None:
                    img_name = batch["rgb_relative_path"][b_idx].replace("/", "_")
                    png_save_path = os.path.join(save_to_dir, f"{img_name}_restored.png")
                    restored_pil = Image.fromarray(
                        (restored_rgb_batch[b_idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    restored_pil.save(png_save_path)

                if log_images_to_wandb and len(wandb_images) < max_images_to_log:
                    clean_np = clean_single_ts.cpu().numpy().transpose(1, 2, 0)
                    degraded_np = degraded_rgb_int[b_idx].cpu().numpy().transpose(1, 2, 0) / 255.0
                    restored_np = restored_rgb_batch[b_idx].transpose(1, 2, 0)

                    degraded_single_ts = degraded_rgb_int[b_idx].to(self.device).float() / 255.0
                    degraded_metrics = {}
                    for met_func in self.metric_funcs:
                        _metric_name = met_func.__name__
                        _metric_deg = met_func(degraded_single_ts, clean_single_ts)
                        degraded_metrics[_metric_name] = float(_metric_deg)

                    comparison = self._create_comparison_image(clean_np, degraded_np, restored_np)
                    img_name = batch["rgb_relative_path"][b_idx].replace("/", "_")

                    caption = f"{img_name}\n"
                    caption += "Degraded -> Restored:\n"
                    for metric_name in sample_metric_dict.keys():
                        deg_val = degraded_metrics[metric_name]
                        res_val = sample_metric_dict[metric_name]
                        improvement = res_val - deg_val
                        if 'lpips' in metric_name.lower():
                            improvement = -improvement
                        caption += f"{metric_name}: {deg_val:.3f} -> {res_val:.3f} ({improvement:+.3f}) | "
                    caption = caption.rstrip(" | ")

                    wandb_images.append(wandb.Image(comparison, caption=caption))
                    wandb_image_metrics.append({
                        'image_name': img_name,
                        **{f'restored_{k}': v for k, v in sample_metric_dict.items()},
                        **{f'degraded_{k}': v for k, v in degraded_metrics.items()},
                    })

        if val_loss_dicts:
            loss_keys = val_loss_dicts[0].keys()
            avg_losses = {}
            for key in loss_keys:
                values = [d[key] for d in val_loss_dicts]
                avg_losses[key] = sum(values) / len(values)
            metric_tracker.update('val_loss', avg_losses['total'])

        results = metric_tracker.result()

        if log_images_to_wandb and wandb_images:
            dataset_name = data_loader.dataset.disp_name
            logging.info(f"Logging {len(wandb_images)} images to W&B at iteration {self.effective_iter}")
            try:
                wandb.log({
                    f"visualization/{dataset_name}": wandb_images,
                }, step=self.effective_iter, commit=False)
                if wandb_image_metrics:
                    all_keys = list(wandb_image_metrics[0].keys())
                    all_keys.remove('image_name')
                    columns = ['image_name'] + sorted(all_keys)
                    data = [[m['image_name']] + [m.get(key, 0.0) for key in sorted(all_keys)]
                            for m in wandb_image_metrics]
                    table = wandb.Table(columns=columns, data=data)
                    wandb.log({
                        f"metrics_table/{dataset_name}": table
                    }, step=self.effective_iter, commit=True)
                logging.info(f"Successfully logged {len(wandb_images)} images to W&B")
            except Exception as e:
                logging.warning(f"Failed to log images to W&B: {e}")
                import traceback
                traceback.print_exc()

        del wandb_images, wandb_image_metrics, val_loss_dicts
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def visualize(self):
        """Copied from hybrid-003 trainer.visualize."""
        logging.info(f"Starting visualization at iteration {self.effective_iter}")
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            logging.info(f"Visualizing dataset: {vis_dataset_name} (W&B logging enabled)")
            vis_metrics = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
                log_images_to_wandb=True,
            )
            if vis_metrics is not None:
                vis_metrics_file = os.path.join(vis_out_dir, "metrics.txt")
                eval_text = eval_dict_to_text(
                    vis_metrics,
                    dataset_name=vis_dataset_name,
                    sample_list_path=val_loader.dataset.filename_ls_path,
                )
                with open(vis_metrics_file, "w+") as f:
                    f.write(eval_text)
                logging.info(f"Visualization metrics saved to: {vis_metrics_file}")

    def _calculate_validation_loss(self, batch, generator):
        """Calculate validation loss — 4ch forward pass (no degraded latent concat).
        Based on hybrid-003 trainer._calculate_validation_loss with
        degraded_latent encoding and concatenation removed."""
        device = self.device

        degraded_rgb = batch[self.degraded_rgb_type].to(device)
        clean_rgb = batch[self.clean_rgb_type].to(device)
        batch_size = degraded_rgb.shape[0]

        with torch.no_grad():
            clean_latent = self.encode_rgb(clean_rgb)
            # NO degraded latent encoding — 4ch UNet

        timesteps = torch.randint(
            0, self.scheduler_timesteps, (batch_size,),
            device=device, generator=generator,
        ).long()

        noise = torch.randn(
            clean_latent.shape, device=device, generator=generator,
        )

        noisy_latents = self.training_noise_scheduler.add_noise(
            clean_latent, noise, timesteps
        )

        text_embed = self.empty_text_embed.to(device).repeat(
            (batch_size, 1, 1)
        )

        controlnet_cond = degraded_rgb

        with autocast('cuda'):
            down_block_res, mid_block_res = self.model.controlnet(
                noisy_latents, timesteps,
                encoder_hidden_states=text_embed,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )

            # 4ch UNet: noisy_latents only (NO degraded latent concat)
            model_pred = self.model.unet(
                noisy_latents, timesteps,
                encoder_hidden_states=text_embed,
                down_block_additional_residuals=down_block_res,
                mid_block_additional_residual=mid_block_res,
            ).sample

            if "sample" == self.prediction_type:
                target = clean_latent
            elif "epsilon" == self.prediction_type:
                target = noise
            elif "v_prediction" == self.prediction_type:
                target = self.training_noise_scheduler.get_velocity(
                    clean_latent, noise, timesteps
                )
            else:
                raise ValueError(f"Unknown prediction type {self.prediction_type}")

        latent_loss = self.loss(model_pred.float(), target.float())
        loss = latent_loss.mean()

        return {'total': loss.item()}

    def _create_comparison_image(self, clean_np, degraded_np, restored_np):
        """Copied from hybrid-003 trainer._create_comparison_image."""
        clean_uint8 = (clean_np * 255).astype(np.uint8)
        degraded_uint8 = (degraded_np * 255).astype(np.uint8)
        restored_uint8 = (restored_np * 255).astype(np.uint8)

        h, w = clean_uint8.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = clean_uint8
        comparison[:, w:2*w] = degraded_uint8
        comparison[:, 2*w:] = restored_uint8

        comparison_img = Image.fromarray(comparison)

        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(comparison_img)
            font_size = max(12, h // 40)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            labels = ["Clean (GT)", "Degraded (Input)", "Restored (Output)"]
            for idx, label in enumerate(labels):
                x_pos = idx * w + 10
                y_pos = 10
                bbox = draw.textbbox((x_pos, y_pos), label, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 128))
                draw.text((x_pos, y_pos), label, fill=(255, 255, 255), font=font)
        except Exception as e:
            logging.debug(f"Could not add text labels to comparison image: {e}")

        return comparison_img
