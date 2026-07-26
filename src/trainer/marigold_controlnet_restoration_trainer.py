# MarigoldControlNetRestorationTrainer - Thesis Implementation
#
# Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction:
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# ControlNet-based approach: frozen SD2 UNet + trainable ControlNet module.
# The ControlNet receives degraded RGB (3ch, pixel space) and injects
# conditioning via zero-convolution residuals into the frozen UNet.
# Only ControlNet parameters are trained (~360M vs ~860M for full UNet).
# --------------------------------------------------------------------------

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


class MarigoldControlNetRestorationTrainer:
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
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # ControlNet approach: UNet stays at native 4 channels, no conv_in replacement needed.
        # The ControlNet receives degraded RGB (3ch) via its built-in conditioning encoder.

        # Encode empty text prompt
        # Verified: MarigoldControlNetRestorationPipeline.encode_empty_text() sets self.empty_text_embed
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        # XFormers memory-efficient attention for both UNet and ControlNet
        # Verified: both inherit from ModelMixin which provides enable_xformers_memory_efficient_attention()
        self.model.unet.enable_xformers_memory_efficient_attention()
        self.model.controlnet.enable_xformers_memory_efficient_attention()

        # Gradient checkpointing (saves VRAM by recomputing activations during backward)
        # Verified: ModelMixin provides enable_gradient_checkpointing()
        self.gradient_checkpointing = self.cfg.trainer.get('gradient_checkpointing', True)
        if self.gradient_checkpointing:
            # Enable for both: UNet needs it for memory during frozen forward pass,
            # ControlNet needs it for memory during trainable forward+backward pass
            self.model.unet.enable_gradient_checkpointing()
            self.model.controlnet.enable_gradient_checkpointing()
            logging.info("Gradient checkpointing ENABLED for both UNet and ControlNet")
        else:
            logging.info("Gradient checkpointing disabled")

        # Trainability: freeze everything except ControlNet
        # UNet is frozen AND set to eval mode (no dropout, no batchnorm updates)
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(False)
        self.model.unet.eval()
        # ControlNet is trainable (requires_grad=True is the default, but be explicit)
        self.model.controlnet.requires_grad_(True)

        logging.info(
            f"Trainability: UNet=frozen+eval, ControlNet=trainable, VAE=frozen, text_encoder=frozen"
        )
        logging.info(
            f"ControlNet params: {sum(p.numel() for p in self.model.controlnet.parameters()) / 1e6:.1f}M"
        )

        # Optimizer: only ControlNet parameters
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.controlnet.parameters(), lr=lr)
        logging.info(f"Optimizer: Adam, LR={lr}, params=ControlNet only")

        # LR scheduler
        # Verified: IterExponential and CosineAnnealingWarmRestarts from src/util/lr_scheduler.py
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

        # Mixed Precision Training (FP16) — always enabled for ControlNet trainer
        self.scaler = GradScaler()
        logging.info("Mixed precision training (FP16) enabled via GradScaler")

        # Loss — simple MSE or L1 on noise/velocity prediction
        # No CombinedRestorationLoss, no gradient loss, no pixel LPIPS
        # Verified: get_loss(loss_name, **kwargs) from src/util/loss.py
        loss_kwargs = {k: v for k, v in self.cfg.loss.kwargs.items() if v is not None}
        logging.info(f"Loss kwargs: {loss_kwargs}")
        self.loss = get_loss(loss_name=self.cfg.loss.name, **loss_kwargs)

        # Training noise scheduler
        # Verified: DDPMScheduler.from_config() with rescale_betas_zero_snr and trailing spacing
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

        # Eval metrics for restoration (PSNR, SSIM, LPIPS)
        # Verified: metric module has psnr, ssim, lpips_alex as functions
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        # Initialize train metrics — simple loss only (no combined loss components)
        self.train_metrics = MetricTracker("loss")

        # Initialize validation metrics
        val_metric_keys = [m.__name__ for m in self.metric_funcs] + ["val_loss"]
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
        # Task-specific data keys (for restoration: degraded and clean RGB)
        self.degraded_rgb_type = getattr(self.cfg, 'degraded_rgb_type', 'degraded_rgb_norm')
        self.clean_rgb_type = getattr(self.cfg, 'clean_rgb_type', 'clean_rgb_norm')
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Offset noise for preventing latent drift
        # Adds a global (spatially constant) offset to the noise to help the model
        # learn to handle images with extreme brightness/darkness values.
        offset_noise_cfg = self.cfg.get('offset_noise', {})
        self.offset_noise_strength = offset_noise_cfg.get('strength', 0.0)
        if self.offset_noise_strength > 0.0:
            logging.info(f"Offset noise ENABLED - strength: {self.offset_noise_strength}")
        else:
            logging.info("Offset noise disabled (strength = 0.0)")

        # Input noise augmentation for conditioning robustness
        # For ControlNet: adds small random noise to degraded RGB in PIXEL SPACE
        # (not latent space, since ControlNet receives pixel-space conditioning)
        # Applied to a percentage of samples with variable strength [0, max_strength]
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

        # No CFG dropout in training.
        # With a frozen UNet, zeroing ControlNet residuals means no trainable parameter
        # receives gradients for dropped samples — pure wasted compute.
        # CFG at inference is still supported via guidance_scale > 1.0 in the pipeline.

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence

        # Checkpoint strategy
        checkpoint_strategy = getattr(cfg.trainer, 'checkpoint_strategy', None)
        self.checkpoint_mode = checkpoint_strategy.get('mode', 'marigold') if checkpoint_strategy else 'marigold'
        if self.checkpoint_mode not in ['marigold', 'decoupled']:
            logging.warning(f"Unknown checkpoint mode '{self.checkpoint_mode}', using 'marigold'")
            self.checkpoint_mode = 'marigold'
        logging.info(f"Checkpoint strategy mode: {self.checkpoint_mode}")
        if self.checkpoint_mode == 'marigold':
            logging.info("  - Original Marigold pattern: save before+after validation, save best during validation")
        else:
            logging.info("  - Decoupled pattern: independent save/val/vis intervals")

        # Whether to save trainer state (optimizer, lr_scheduler, etc.) in checkpoints
        self.save_trainer_state = self.cfg.trainer.get('save_trainer_state', True)
        logging.info(f"Save trainer state in checkpoints: {self.save_trainer_state}")

        # Checkpoint test configuration for disk space management
        self.checkpoint_test_config = getattr(cfg, 'checkpoint_test', None)
        if self.checkpoint_test_config:
            logging.info("Space-optimized checkpoint settings enabled:")
            logging.info(f"  - Save ControlNet only: {self.checkpoint_test_config.get('save_unet_only', False)}")
            logging.info(f"  - Keep only latest+best: {self.checkpoint_test_config.get('keep_only_latest_best', False)}")
            logging.info(f"  - Auto cleanup: {self.checkpoint_test_config.get('auto_cleanup', False)}")
            logging.info(f"  - Max checkpoints: {self.checkpoint_test_config.get('max_checkpoints', 2)}")
            logging.info(f"  - Min free space: {self.checkpoint_test_config.get('min_free_space_gb', 5.0)} GB")

    def encode_rgb(self, image_in):
        """Encode RGB image to latent space.
        Verified: MarigoldControlNetRestorationPipeline.encode_rgb() at line 117.
        Cast to VAE dtype because VAE is loaded in float16 (frozen) but dataset
        tensors arrive as float32."""
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        image_in = image_in.to(self.model.vae.dtype)
        latent = self.model.encode_rgb(image_in)
        return latent

    def decode_rgb(self, latent_in):
        """Decode latent to RGB image [0, 1].
        Verified: MarigoldControlNetRestorationPipeline.decode_rgb() at line 134.
        Cast to VAE dtype because VAE is loaded in float16 (frozen)."""
        assert len(latent_in.shape) == 4 and latent_in.shape[1] == 4
        latent_in = latent_in.to(self.model.vae.dtype)
        rgb = self.model.decode_rgb(latent_in)
        # Convert from [-1, 1] to [0, 1] range
        rgb = (rgb + 1.0) / 2.0
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb

    def _get_next_seed(self):
        """Get next seed from global sequence.
        Copied from MarigoldRestorationTrainer._get_next_seed() at line 1834."""
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
        """Copied from MarigoldRestorationTrainer._get_backup_ckpt_name() at line 2035."""
        return f"iter_{self.effective_iter:06d}"

    def train(self, t_end=None):
        """Training loop for ControlNet restoration.
        
        Key differences from MarigoldRestorationTrainer.train():
        - ControlNet forward pass: degraded RGB (pixel space) → ControlNet → residuals → frozen UNet
        - UNet stays in eval mode, ControlNet in train mode
        - FP16 only (single code path)
        - Simple MSE loss on noise/velocity prediction
        - No ARNIQA, no combined loss, no gradient loss, no pixel LPIPS
        - Input noise augmentation in pixel space (not latent space)
        - No CFG dropout (useless with frozen UNet)
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
                # Handle ConcatDataset (mixed datasets)
                for dataset in self.train_loader.dataset.datasets:
                    if hasattr(dataset, 'set_epoch'):
                        dataset.set_epoch(self.epoch)

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                # ControlNet in train mode, UNet stays in eval mode
                self.model.controlnet.train()

                # Globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data — degraded and clean RGB in [-1, 1]
                degraded_rgb = batch[self.degraded_rgb_type].to(device)  # [B, 3, H, W] in [-1, 1]
                clean_rgb = batch[self.clean_rgb_type].to(device)  # [B, 3, H, W] in [-1, 1]

                batch_size = degraded_rgb.shape[0]

                with torch.no_grad():
                    # Encode clean RGB to latent (target for loss)
                    clean_latent = self.encode_rgb(clean_rgb)  # [B, 4, h, w]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                noise = torch.randn(
                    clean_latent.shape,
                    device=device,
                    generator=rand_num_generator,
                )  # [B, 4, h, w]

                # Apply offset noise to prevent latent drift toward mean values
                if self.offset_noise_strength > 0.0:
                    offset = torch.randn(
                        batch_size, clean_latent.shape[1], 1, 1,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, 1, 1] — same offset across spatial dimensions
                    noise = noise + self.offset_noise_strength * offset

                # Add noise to the clean latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    clean_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Text embedding (empty text — no ARNIQA)
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # Input noise augmentation: add small noise to degraded RGB in PIXEL SPACE
                # ControlNet receives pixel-space conditioning, so augmentation is in pixel space
                controlnet_cond = degraded_rgb  # [B, 3, H, W] in [-1, 1]
                if self.input_noise_prob > 0.0:
                    augment_mask = (
                        torch.rand(batch_size, device=device, generator=rand_num_generator)
                        < self.input_noise_prob
                    )
                    if augment_mask.any():
                        # Variable strength per sample: uniform [0, max_strength]
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
                        # Clamp to valid range after adding noise
                        controlnet_cond = torch.clamp(controlnet_cond, -1.0, 1.0)

                # Forward pass with FP16 mixed precision
                with autocast('cuda'):
                    # ControlNet forward: produces residuals from degraded RGB conditioning
                    # Verified: ControlNetModel.__call__ signature accepts
                    # (sample, timestep, encoder_hidden_states, controlnet_cond, return_dict)
                    down_block_res, mid_block_res = self.model.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embed,
                        controlnet_cond=controlnet_cond,
                        return_dict=False,
                    )

                    # Frozen UNet forward: inject ControlNet residuals
                    # Verified: UNet2DConditionModel.__call__ accepts
                    # down_block_additional_residuals and mid_block_additional_residual
                    model_pred = self.model.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embed,
                        down_block_additional_residuals=down_block_res,
                        mid_block_additional_residual=mid_block_res,
                    ).sample  # [B, 4, h, w]

                    # Get the target for loss depending on the prediction type
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

                # Loss computation in FP32 for numerical stability
                latent_loss = self.loss(model_pred.float(), target.float())
                loss = latent_loss.mean()

                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                self.train_metrics.update("loss", loss.item())

                loss = loss / self.gradient_accumulation_steps

                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()

                accumulated_step += 1
                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping on ControlNet params only
                    torch.nn.utils.clip_grad_norm_(
                        self.model.controlnet.parameters(), max_norm=1.0
                    )
                    # Mixed precision optimizer step
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

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def _train_step_callback(self):
        """Executed after every iteration.
        Copied from MarigoldRestorationTrainer._train_step_callback() at line 1168."""
        if self.checkpoint_mode == 'marigold':
            self._train_step_callback_marigold()
        else:
            self._train_step_callback_decoupled()

    def _train_step_callback_marigold(self):
        """Original Marigold checkpoint pattern.
        Copied from MarigoldRestorationTrainer._train_step_callback_marigold() at line 1175."""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def _train_step_callback_decoupled(self):
        """Decoupled checkpoint pattern: independent save/val/vis intervals.
        Copied from MarigoldRestorationTrainer._train_step_callback_decoupled() at line 1205."""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        # Save training checkpoint (independent from validation)
        if self.save_period > 0 and 0 == self.effective_iter % self.save_period:
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Validation (no checkpoint saving, only metrics logging)
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.validate_decoupled()

        # Visualization (independent from save)
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def save_checkpoint(self, ckpt_name, save_train_state):
        """Save checkpoint with ControlNet weights.
        
        Key difference from existing trainer: saves controlnet/ instead of unet/.
        Pattern copied from MarigoldRestorationTrainer.save_checkpoint() at line 1845.
        """
        # Check if space-optimized mode is enabled
        if self.checkpoint_test_config and self.checkpoint_test_config.get('auto_cleanup', False):
            self._check_disk_space_and_cleanup()

        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")

        # Backup previous checkpoint
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

        # Save ControlNet (instead of UNet)
        controlnet_path = os.path.join(ckpt_dir, "controlnet")
        self.model.controlnet.save_pretrained(controlnet_path, safe_serialization=True)
        logging.info(f"ControlNet is saved to: {controlnet_path}")

        # Save scheduler (unless in save_unet_only mode)
        if not (self.checkpoint_test_config and self.checkpoint_test_config.get('save_unet_only', False)):
            scheduler_path = os.path.join(ckpt_dir, "scheduler")
            self.model.scheduler.save_pretrained(scheduler_path)
            logging.info(f"Scheduler is saved to: {scheduler_path}")
        else:
            logging.info("Skipping scheduler save (ControlNet-only mode)")

        # Save restoration config (for forward compatibility)
        restoration_config = {}
        restoration_config_path = os.path.join(ckpt_dir, "restoration_config.json")
        with open(restoration_config_path, "w") as f:
            json.dump(restoration_config, f, indent=2)
        logging.info(f"Restoration config saved to: {restoration_config_path}")

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

        # Iteration indicator (always saved, regardless of trainer state)
        if save_train_state:
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

        # Log checkpoint size and disk usage
        if self.checkpoint_test_config:
            self._log_checkpoint_info(ckpt_dir)

        # Cleanup old checkpoints if in space-optimized mode
        if self.checkpoint_test_config and self.checkpoint_test_config.get('keep_only_latest_best', False):
            self._cleanup_old_checkpoints(ckpt_name)

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        """Load checkpoint with ControlNet weights.
        
        Key difference from existing trainer: loads controlnet/ instead of unet/.
        Pattern copied from MarigoldRestorationTrainer.load_checkpoint() at line 1950.
        """
        logging.info(f"Loading checkpoint from: {ckpt_path}")

        # Load ControlNet weights into existing ControlNet (not create new one!)
        # This keeps the optimizer connected to the same ControlNet object
        _model_path = os.path.join(
            ckpt_path, "controlnet", "diffusion_pytorch_model.safetensors"
        )

        from safetensors.torch import load_file
        state_dict = load_file(_model_path)

        # Update existing ControlNet weights (optimizer stays connected!)
        self.model.controlnet.load_state_dict(state_dict)
        self.model.controlnet.to(self.device)
        logging.info(f"ControlNet parameters loaded from {_model_path}")

        # Load training states
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

            # Load scaler state
            if "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
                logging.info(f"Mixed precision scaler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. "
            f"Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _check_disk_space_and_cleanup(self):
        """Check available disk space and cleanup if necessary.
        Copied from MarigoldRestorationTrainer._check_disk_space_and_cleanup() at line 2094."""
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
        """Log checkpoint size and disk usage information.
        Copied from MarigoldRestorationTrainer._log_checkpoint_info() at line 2119."""
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
        """Remove old checkpoints to save space.
        Copied from MarigoldRestorationTrainer._cleanup_old_checkpoints() at line 2143."""
        if not self.checkpoint_test_config or not self.checkpoint_test_config.get('keep_only_latest_best', False):
            return

        try:
            max_checkpoints = self.checkpoint_test_config.get('max_checkpoints', 2)

            # Get all checkpoint directories
            ckpt_dirs = []
            for item in os.listdir(self.out_dir_ckpt):
                item_path = os.path.join(self.out_dir_ckpt, item)
                if os.path.isdir(item_path) and item.startswith('iter_'):
                    ckpt_dirs.append(item)

            # Sort by iteration number (newest first)
            ckpt_dirs.sort(key=lambda x: int(x.split('_')[1]), reverse=True)

            # Keep only the latest and best checkpoints
            keep_dirs = set()

            if current_ckpt_name:
                keep_dirs.add(current_ckpt_name)
            elif ckpt_dirs:
                keep_dirs.add(ckpt_dirs[0])

            if hasattr(self, 'best_iter') and self.best_iter:
                best_ckpt_name = f"iter_{self.best_iter:06d}"
                keep_dirs.add(best_ckpt_name)

            # Remove excess checkpoints
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

    def validate(self):
        """Validation with checkpoint saving (Marigold mode).
        Copied from MarigoldRestorationTrainer.validate() at line 1225."""
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
            # Save to file
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

            # Update main eval metric
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
        Copied from MarigoldRestorationTrainer.validate_decoupled() at line 1275."""
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

            # Update main eval metric (but don't save checkpoint)
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

    def visualize(self):
        """Visualization: run inference and save images.
        Copied from MarigoldRestorationTrainer.visualize() at line 1322."""
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

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
        log_images_to_wandb: bool = False,
    ):
        """Validate on a single dataset.
        
        Pattern copied from MarigoldRestorationTrainer.validate_single_dataset() at line 1354.
        Adapted for ControlNet: uses model.single_infer() which handles ControlNet conditioning.
        Simplified: no gradient loss visualization, no ARNIQA, no combined loss components.
        """
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        total_images = len(data_loader.dataset)
        val_seed_ls = generate_seed_sequence(val_init_seed, total_images)

        # Track validation loss
        val_loss_dicts = []

        # W&B image logging configuration
        max_images_to_log = getattr(self.cfg.validation, 'max_images_to_log', 8)
        wandb_images = []
        wandb_image_metrics = []

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            batch_size = batch["degraded_rgb_int"].shape[0]

            # Read input images — pipeline expects [0, 255] format
            degraded_rgb_int = batch["degraded_rgb_int"]  # [B, 3, H, W] in [0, 255]
            clean_rgb_int = batch["clean_rgb_int"]  # [B, 3, H, W] in [0, 255]

            # Generate seeds for this batch
            batch_seeds = [val_seed_ls.pop() for _ in range(batch_size)]

            # Calculate validation loss for each image in batch
            for b_idx in range(batch_size):
                batch_single = {
                    k: v[b_idx:b_idx+1] if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v
                    for k, v in batch.items()
                }
                seed = batch_seeds[b_idx]
                generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
                val_loss_dict = self._calculate_validation_loss(batch_single, generator)
                val_loss_dicts.append(val_loss_dict)

            # Predict restored images using single_infer
            # Normalize degraded images to [-1, 1] range
            degraded_rgb_norm = degraded_rgb_int.float() / 255.0 * 2.0 - 1.0  # [B, 3, H, W] in [-1, 1]
            degraded_rgb_norm = degraded_rgb_norm.to(self.device)

            # Set up scheduler for inference
            self.model.scheduler.set_timesteps(
                self.cfg.validation.denoising_steps, device=self.device
            )

            # Process batch through pipeline (single_infer handles ControlNet conditioning)
            # autocast needed: ControlNet is float32 (trainable), UNet+VAE are float16 (frozen)
            with autocast('cuda'):
                restored_rgb_batch_ts = self.model.single_infer(
                    rgb_in=degraded_rgb_norm,
                    num_inference_steps=self.cfg.validation.denoising_steps,
                    generator=None,
                    show_pbar=False,
                )  # Returns [B, 3, H, W] in [-1, 1]

            # Convert to [0, 1] range and numpy
            restored_rgb_batch_ts = (restored_rgb_batch_ts + 1.0) / 2.0
            restored_rgb_batch_ts = torch.clip(restored_rgb_batch_ts, 0.0, 1.0)
            restored_rgb_batch = restored_rgb_batch_ts.cpu().numpy()  # [B, 3, H, W] in [0, 1]

            # Process each image in batch for metrics and logging
            for b_idx in range(batch_size):
                restored_rgb_ts = torch.from_numpy(restored_rgb_batch[b_idx]).to(self.device)  # [3, H, W]
                clean_single_ts = clean_rgb_int[b_idx].to(self.device).float() / 255.0  # [3, H, W]

                # Evaluate restoration metrics (on GPU for speed)
                sample_metric_dict = {}
                for met_func in self.metric_funcs:
                    _metric_name = met_func.__name__
                    _metric = met_func(restored_rgb_ts, clean_single_ts)
                    sample_metric_dict[_metric_name] = float(_metric)
                    metric_tracker.update(_metric_name, _metric)

                # Save restored image
                if save_to_dir is not None:
                    img_name = batch["rgb_relative_path"][b_idx].replace("/", "_")
                    png_save_path = os.path.join(save_to_dir, f"{img_name}_restored.png")
                    restored_pil = Image.fromarray(
                        (restored_rgb_batch[b_idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    restored_pil.save(png_save_path)

                # Log images to W&B (only first N images)
                if log_images_to_wandb and len(wandb_images) < max_images_to_log:
                    clean_np = clean_single_ts.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
                    degraded_np = degraded_rgb_int[b_idx].cpu().numpy().transpose(1, 2, 0) / 255.0
                    restored_np = restored_rgb_batch[b_idx].transpose(1, 2, 0)  # [H, W, 3]

                    # Calculate degraded image metrics for comparison
                    degraded_single_ts = degraded_rgb_int[b_idx].to(self.device).float() / 255.0
                    degraded_metrics = {}
                    for met_func in self.metric_funcs:
                        _metric_name = met_func.__name__
                        _metric_deg = met_func(degraded_single_ts, clean_single_ts)
                        degraded_metrics[_metric_name] = float(_metric_deg)

                    # Create side-by-side comparison
                    comparison = self._create_comparison_image(clean_np, degraded_np, restored_np)

                    img_name = batch["rgb_relative_path"][b_idx].replace("/", "_")

                    # Create caption with metrics
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

        # Calculate average validation loss
        # Simplified: only 'total' key (no CombinedRestorationLoss components)
        if val_loss_dicts:
            loss_keys = val_loss_dicts[0].keys()
            avg_losses = {}
            for key in loss_keys:
                values = [d[key] for d in val_loss_dicts]
                avg_losses[key] = sum(values) / len(values)

            metric_tracker.update('val_loss', avg_losses['total'])

        # Get results from metric tracker
        results = metric_tracker.result()

        # Log images to W&B and TensorBoard
        # Pattern copied from MarigoldRestorationTrainer.validate_single_dataset() at line 1585
        if log_images_to_wandb and wandb_images:
            dataset_name = data_loader.dataset.disp_name

            logging.info(f"Logging {len(wandb_images)} images to W&B at iteration {self.effective_iter}")
            try:
                wandb.log({
                    f"visualization/{dataset_name}": wandb_images,
                }, step=self.effective_iter, commit=False)

                # Log individual image metrics as a table
                if wandb_image_metrics:
                    all_keys = list(wandb_image_metrics[0].keys())
                    all_keys.remove('image_name')

                    columns = ['image_name'] + sorted(all_keys)
                    data = [[m['image_name']] + [m.get(key, 0.0) for key in sorted(all_keys)]
                            for m in wandb_image_metrics]

                    table = wandb.Table(columns=columns, data=data)

                    logging.info(f"Logging metrics table to W&B with {len(data)} rows and {len(columns)} columns")
                    wandb.log({
                        f"metrics_table/{dataset_name}": table
                    }, step=self.effective_iter, commit=True)

                logging.info(f"Successfully logged {len(wandb_images)} images to W&B")

            except Exception as e:
                logging.warning(f"Failed to log images to W&B: {e}")
                import traceback
                traceback.print_exc()

        return results

    def _calculate_validation_loss(self, batch, generator):
        """Calculate validation loss using the same process as training loss.

        Adapted for ControlNet: uses ControlNet + frozen UNet forward pass.
        Simplified: no CombinedRestorationLoss, no gradient loss, no latent normalization,
        no multi-res noise, no ARNIQA, no timestep-aware conditioning.

        Pattern from MarigoldRestorationTrainer._calculate_validation_loss() at line 1636.

        Returns:
            dict: Dictionary with 'total' loss value
        """
        device = self.device

        # Get data — same as training
        degraded_rgb = batch[self.degraded_rgb_type].to(device)  # [B, 3, H, W] in [-1, 1]
        clean_rgb = batch[self.clean_rgb_type].to(device)  # [B, 3, H, W] in [-1, 1]
        batch_size = degraded_rgb.shape[0]

        # Encode clean RGB to latent — same as training
        clean_latent = self.encode_rgb(clean_rgb)  # [B, 4, h, w]

        # Sample random timestep — same as training
        timesteps = torch.randint(
            0,
            self.scheduler_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        ).long()  # [B]

        # Sample noise — same as training (no multi-res noise)
        noise = torch.randn(
            clean_latent.shape,
            device=device,
            generator=generator,
        )  # [B, 4, h, w]

        # Add noise — same as training (no offset noise during validation)
        noisy_latents = self.training_noise_scheduler.add_noise(
            clean_latent, noise, timesteps
        )  # [B, 4, h, w]

        # Text embedding (empty text — no ARNIQA)
        text_embed = self.empty_text_embed.to(device).repeat(
            (batch_size, 1, 1)
        )  # [B, 77, 1024]

        # ControlNet conditioning: degraded RGB in pixel space (no input noise during validation)
        controlnet_cond = degraded_rgb  # [B, 3, H, W] in [-1, 1]

        # Forward pass with FP16 mixed precision (matching training loop pattern)
        with autocast('cuda'):
            # ControlNet forward: produces residuals from degraded RGB conditioning
            # Verified: ControlNetModel.__call__ at diffusers source
            down_block_res, mid_block_res = self.model.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embed,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )

            # Frozen UNet forward: inject ControlNet residuals
            # Verified: UNet2DConditionModel.__call__ accepts down_block_additional_residuals
            model_pred = self.model.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embed,
                down_block_additional_residuals=down_block_res,
                mid_block_additional_residual=mid_block_res,
            ).sample  # [B, 4, h, w]

            # Get target — same as training
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

        # Simple loss (MSE or L1 on noise/velocity prediction) — FP32 for stability
        latent_loss = self.loss(model_pred.float(), target.float())
        loss = latent_loss.mean()

        loss_dict = {'total': loss.item()}
        return loss_dict

    def _create_comparison_image(self, clean_np, degraded_np, restored_np):
        """Create a side-by-side comparison image: Clean | Degraded | Restored.

        Copied from MarigoldRestorationTrainer._create_comparison_image() at line 2038.

        Args:
            clean_np: Clean image [H, W, 3] in [0, 1]
            degraded_np: Degraded image [H, W, 3] in [0, 1]
            restored_np: Restored image [H, W, 3] in [0, 1]

        Returns:
            PIL Image with side-by-side comparison
        """
        # Convert to uint8
        clean_uint8 = (clean_np * 255).astype(np.uint8)
        degraded_uint8 = (degraded_np * 255).astype(np.uint8)
        restored_uint8 = (restored_np * 255).astype(np.uint8)

        # Get dimensions
        h, w = clean_uint8.shape[:2]

        # Create comparison image (3 images side by side)
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = clean_uint8
        comparison[:, w:2*w] = degraded_uint8
        comparison[:, 2*w:] = restored_uint8

        # Convert to PIL Image
        comparison_img = Image.fromarray(comparison)

        # Add text labels
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
