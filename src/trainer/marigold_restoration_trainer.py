# MarigoldRestorationTrainer - Thesis Implementation
# 
# Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction:
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# This implementation extends the Marigold framework for blind image restoration.
# Based on the Marigold depth trainer architecture but adapted for RGB-to-RGB restoration.
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
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from typing import List, Union

from marigold.marigold_restoration_pipeline_base import MarigoldRestorationPipelineBase as MarigoldRestorationPipeline, MarigoldRestorationOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss, LatentGradientLoss
from src.util.lr_scheduler import IterExponential, CosineAnnealingWarmRestarts
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence

# ARNIQA quality-aware conditioning (optional)
from src.ARNIQA import ArniqaConditioner


class MarigoldRestorationTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldRestorationPipeline,
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
        self.model: MarigoldRestorationPipeline = model
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

        # Adapt input layers
        if 8 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        self.model.unet.enable_xformers_memory_efficient_attention()

        # Gradient checkpointing (saves VRAM by recomputing activations during backward)
        self.gradient_checkpointing = self.cfg.trainer.get('gradient_checkpointing', True)
        if self.gradient_checkpointing:
            self.model.unet.enable_gradient_checkpointing()
            logging.info("Gradient checkpointing ENABLED for U-Net (saves VRAM, ~20-30% slower)")
        else:
            logging.info("Gradient checkpointing disabled")

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)

        # ARNIQA quality-aware conditioning (optional)
        # When enabled, replaces empty_text_embed with quality features from degraded image
        # Stage 1: Global-only conditioning (1 token)
        # Stage 2: Global + Spatial conditioning (1 + spatial_size² tokens)
        arniqa_cfg = self.cfg.get('arniqa', {})
        self.use_arniqa = arniqa_cfg.get('enabled', False)
        self.arniqa_conditioner = None
        self.arniqa_stage = 1  # Default to Stage 1
        
        if self.use_arniqa:
            arniqa_dropout = arniqa_cfg.get('conditioning_dropout', 0.1)
            arniqa_num_tokens = arniqa_cfg.get('num_tokens', 1)
            self.arniqa_stage = arniqa_cfg.get('stage', 1)
            arniqa_spatial_size = arniqa_cfg.get('spatial_size', 24)
            
            self.arniqa_conditioner = ArniqaConditioner(
                output_dim=1024,
                conditioning_dropout=arniqa_dropout,
                stage=self.arniqa_stage,
                spatial_size=arniqa_spatial_size,
                num_tokens=arniqa_num_tokens,
            ).to(device)
            self.model.set_arniqa_conditioner(self.arniqa_conditioner)
            
            if self.arniqa_stage >= 2:
                total_tokens = 1 + (arniqa_spatial_size * arniqa_spatial_size)
                logging.info(
                    f"ARNIQA Stage {self.arniqa_stage} ENABLED: "
                    f"dropout={arniqa_dropout:.1%}, "
                    f"spatial_size={arniqa_spatial_size}, "
                    f"total_tokens={total_tokens}"
                )
            else:
                logging.info(
                    f"ARNIQA Stage {self.arniqa_stage} ENABLED: "
                    f"dropout={arniqa_dropout:.1%}, "
                    f"num_tokens={arniqa_num_tokens}"
                )
        else:
            logging.info("ARNIQA conditioning disabled (using empty text embedding)")

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        if self.use_arniqa:
            # Separate parameter groups: U-Net (lower LR) + ARNIQA adapters (higher LR)
            arniqa_global_lr = arniqa_cfg.get('adapter_lr', 1e-4)
            arniqa_spatial_lr = arniqa_cfg.get('spatial_adapter_lr', arniqa_global_lr)  # Default to same as global
            
            # Get parameter groups from conditioner (handles Stage 1 vs Stage 2)
            arniqa_param_groups = self.arniqa_conditioner.get_parameter_groups(
                global_adapter_lr=arniqa_global_lr,
                spatial_adapter_lr=arniqa_spatial_lr,
            )
            
            param_groups = [
                {'params': self.model.unet.parameters(), 'lr': lr, 'name': 'unet'},
            ] + arniqa_param_groups
            
            self.optimizer = Adam(param_groups)
            
            if self.arniqa_stage >= 2:
                logging.info(
                    f"Optimizer: U-Net LR={lr}, "
                    f"ARNIQA global_adapter LR={arniqa_global_lr}, "
                    f"ARNIQA spatial_adapter LR={arniqa_spatial_lr}"
                )
            else:
                logging.info(f"Optimizer: U-Net LR={lr}, ARNIQA adapter LR={arniqa_global_lr}")
        else:
            self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        # LR scheduler
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
            raise ValueError(f"Unknown scheduler: {scheduler_name}. Supported: IterExponential, CosineAnnealingWarmRestarts")
        
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Mixed Precision Training (FP16)
        self.mixed_precision = self.cfg.trainer.get('mixed_precision', True)
        if self.mixed_precision:
            self.scaler = GradScaler()
            logging.info("Mixed precision training (FP16) enabled - expect 2-3× speedup")
        else:
            self.scaler = None
            logging.info("Training in FP32 (full precision)")

        # Noise loss type for latent space (mse or l1) - trainer-only parameter
        self.noise_loss_type = self.cfg.loss.kwargs.get('noise_loss_type', 'mse')
        logging.info(f"Noise loss type (latent space): {self.noise_loss_type}")

        # Loss - Filter out None values and trainer-only parameters from kwargs
        # noise_weight, image_weight are used by trainer to combine losses, not by loss class
        trainer_only_params = {'noise_loss_type', 'noise_weight', 'image_weight', 'gradient_weight', 'lpips_weight', 'dino_weight', 'dino_model_size', 'dino_version', 'dino_target_size', 'dino_layers'}
        loss_kwargs = {k: v for k, v in self.cfg.loss.kwargs.items() 
                       if v is not None and k not in trainer_only_params}
        logging.info(f"Loss kwargs after filtering: {loss_kwargs}")
        self.loss = get_loss(loss_name=self.cfg.loss.name, **loss_kwargs)

        # Latent gradient loss (optional) - penalizes loss of edges/sharpness in latent space
        # Uses fixed Sobel kernels, zero trainable parameters, negligible compute
        self.gradient_weight = self.cfg.loss.kwargs.get('gradient_weight', 0.0)
        if self.gradient_weight > 0:
            self.gradient_loss = LatentGradientLoss().to(device)
            logging.info(f"Latent gradient loss ENABLED: weight={self.gradient_weight}")
        else:
            self.gradient_loss = None
            logging.info("Latent gradient loss disabled (gradient_weight=0)")

        # Pixel-space LPIPS loss (optional) - perceptual quality via VAE decode on random crop
        # Decodes only a small random crop of the predicted latent to pixel space,
        # then computes LPIPS against the corresponding clean crop.
        # Crop position is seeded from per-iteration RNG for reproducibility.
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
                f"({self.pixel_loss_crop_size // 8}×{self.pixel_loss_crop_size // 8} latent)"
            )
        else:
            self.lpips_loss = None
            logging.info("Pixel-space LPIPS disabled (lpips_weight=0)")

        # Pixel-space DINO perceptual loss (optional) - modern alternative to LPIPS
        # Uses frozen DINOv2/v3 features on the same random crop as LPIPS.
        # Reference: https://na-vae.github.io/dino_perceptual/
        self.dino_weight = self.cfg.loss.kwargs.get('dino_weight', 0.0)
        if self.dino_weight > 0:
            from src.util.loss import DINOPerceptualLoss
            dino_model_size = self.cfg.loss.kwargs.get('dino_model_size', 'B')
            dino_version = self.cfg.loss.kwargs.get('dino_version', 'v3')
            dino_target_size = self.cfg.loss.kwargs.get('dino_target_size', 256)
            dino_layers = self.cfg.loss.kwargs.get('dino_layers', 'all')
            self.dino_loss = DINOPerceptualLoss(
                model_size=dino_model_size,
                version=dino_version,
                target_size=dino_target_size,
                layers=dino_layers,
                device=device,
            )
            logging.info(
                f"Pixel-space DINO ENABLED: weight={self.dino_weight}, "
                f"crop_size={self.pixel_loss_crop_size}px, "
                f"model=DINOv{dino_version[-1]}-{dino_model_size}, "
                f"target_size={dino_target_size}"
            )
        else:
            self.dino_loss = None
            logging.info("Pixel-space DINO disabled (dino_weight=0)")

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(
            self.model.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )

        logging.info(
            "DDPM training noise scheduler config is updated: "
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
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        # Initialize train metrics with loss components
        train_metric_keys = ["loss"]
        # Add loss component keys if using combined loss
        if hasattr(self.loss, '__class__') and 'CombinedRestorationLoss' in self.loss.__class__.__name__:
            train_metric_keys.extend(["noise_loss", "elatentlpips", "image_loss", "image_mse", "image_l1", "image_perceptual", "image_lpips"])
        # Add gradient loss key if enabled
        if self.gradient_weight > 0:
            train_metric_keys.append("gradient_loss")
        # Add pixel-space LPIPS key if enabled
        if self.lpips_weight > 0:
            train_metric_keys.append("pixel_lpips")
        # Add pixel-space DINO key if enabled
        if self.dino_weight > 0:
            train_metric_keys.append("pixel_dino")
        self.train_metrics = MetricTracker(*train_metric_keys)
        
        # Initialize validation metrics with loss components
        val_metric_keys = [m.__name__ for m in self.metric_funcs] + ["val_loss"]
        if hasattr(self.loss, '__class__') and 'CombinedRestorationLoss' in self.loss.__class__.__name__:
            val_metric_keys.extend(["val_noise_loss", "val_elatentlpips", "val_image_loss", "val_image_mse", "val_image_l1", "val_image_perceptual", "val_image_lpips"])
        # Add gradient loss key if enabled
        if self.gradient_weight > 0:
            val_metric_keys.append("val_gradient_loss")
        # Add pixel-space LPIPS key if enabled
        if self.lpips_weight > 0:
            val_metric_keys.append("val_pixel_lpips")
        # Add pixel-space DINO key if enabled
        if self.dino_weight > 0:
            val_metric_keys.append("val_pixel_dino")
        self.val_metrics = MetricTracker(*val_metric_keys)

        # main metric for best checkpoint saving
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

        # Latent normalization (zero mean + unit variance)
        # Normalizes clean_latent and degraded_latent BEFORE noise addition
        # This makes the velocity target naturally zero-mean and unit variance
        self.normalize_latents = self.cfg.trainer.get('normalize_latents', False)
        if self.normalize_latents:
            logging.info("Latent normalization ENABLED - latents will be normalized (zero mean + unit variance) before noise addition")
        else:
            logging.info("Latent normalization disabled (default)")

        # Multi-resolution noise
        self.apply_multi_res_noise = False # self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Offset noise for preventing latent drift
        # Adds a global (spatially constant) offset to the noise to help the model
        # learn to handle images with extreme brightness/darkness values.
        # Reference: https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_cfg = self.cfg.get('offset_noise', {})
        self.offset_noise_strength = offset_noise_cfg.get('strength', 0.0)
        if self.offset_noise_strength > 0.0:
            logging.info(f"Offset noise ENABLED - strength: {self.offset_noise_strength}")
        else:
            logging.info("Offset noise disabled (strength = 0.0)")

        # Input noise augmentation for conditioning robustness
        # Adds small random noise to the degraded latent (conditioning) during training
        # Applied to a percentage of samples with variable strength [0, max_strength]
        input_noise_cfg = self.cfg.get('input_noise_augmentation', {})
        self.input_noise_prob = input_noise_cfg.get('probability', 0.0)
        self.input_noise_max_strength = input_noise_cfg.get('max_strength', 0.1)
        if self.input_noise_prob > 0.0:
            logging.info(f"Input noise augmentation ENABLED - probability: {self.input_noise_prob:.1%}, max_strength: {self.input_noise_max_strength}")
        else:
            logging.info("Input noise augmentation disabled (probability = 0.0)")

        # Timestep-aware conditioning scaling
        cond_cfg = self.cfg.get('conditioning', {})
        self.cond_timestep_scaling = cond_cfg.get('timestep_scaling', False)
        self.cond_scale_min = cond_cfg.get('scale_min', 0.1)
        self.cond_scale_max = cond_cfg.get('scale_max', 0.9)
        if self.cond_timestep_scaling:
            logging.info(f"Timestep-aware conditioning scaling enabled: [{self.cond_scale_min}, {self.cond_scale_max}]")
        else:
            logging.info("Timestep-aware conditioning scaling disabled")

        # Classifier-Free Guidance (CFG) training
        # Randomly drop conditioning to enable guidance_scale > 1.0 during inference
        cfg_config = self.cfg.get('cfg', {})
        self.conditioning_dropout_prob = cfg_config.get('conditioning_dropout_prob', 0.0)
        if self.conditioning_dropout_prob > 0.0:
            logging.info(f"CFG training ENABLED - conditioning dropout probability: {self.conditioning_dropout_prob:.1%}")
        else:
            logging.info("CFG training disabled (conditioning_dropout_prob = 0.0)")

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

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
        # When False, only model weights are saved (inference-only, cannot resume training)
        self.save_trainer_state = self.cfg.trainer.get('save_trainer_state', True)
        logging.info(f"Save trainer state in checkpoints: {self.save_trainer_state}")

        # Test configuration support for 20GB constraint
        self.checkpoint_test_config = getattr(cfg, 'checkpoint_test', None)
        if self.checkpoint_test_config:
            logging.info("Space-optimized checkpoint settings enabled:")
            logging.info(f"  - Save U-Net only: {self.checkpoint_test_config.get('save_unet_only', False)}")
            logging.info(f"  - Keep only latest+best: {self.checkpoint_test_config.get('keep_only_latest_best', False)}")
            logging.info(f"  - Auto cleanup: {self.checkpoint_test_config.get('auto_cleanup', False)}")
            logging.info(f"  - Max checkpoints: {self.checkpoint_test_config.get('max_checkpoints', 2)}")
            logging.info(f"  - Min free space: {self.checkpoint_test_config.get('min_free_space_gb', 5.0)} GB")

    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]
        
        # Zero-initialize conditioning channels (degraded image), keep pretrained weights for noisy channels
        # This follows ControlNet's approach: the network starts as a standard denoiser,
        # then gradually learns to incorporate the conditioning signal
        # Channel layout: [0:4] = degraded (condition), [4:8] = noisy (target to denoise)
        # _weight_cond = torch.zeros_like(_weight)  # Conditioning channels start from zero
        # _weight_noisy = _weight.clone()  # Noisy channels keep pretrained weights
        # _weight = torch.cat([_weight_cond, _weight_noisy], dim=1)  # [320, 8, 3, 3]

        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.5

        # new conv_in channel
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.model.unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return

    def _get_conditioning_scale(self, timesteps):
        """
        Compute timestep-aware conditioning scale.
        Higher scale at high noise levels (structure guidance needed),
        lower scale at low noise levels (fine details).
        
        Args:
            timesteps: Tensor of timesteps [B]
        
        Returns:
            scale: Tensor of shape [B, 1, 1, 1] for broadcasting
        """
        # Linear interpolation: scale_min at t=0, scale_max at t=max_timestep
        t_normalized = timesteps.float() / self.scheduler_timesteps
        scale = self.cond_scale_min + (self.cond_scale_max - self.cond_scale_min) * t_normalized
        return scale.view(-1, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting

    def train(self, t_end=None):
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
                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data - for restoration: degraded and clean RGB
                degraded_rgb = batch[self.degraded_rgb_type].to(device)  # [-1, 1]
                clean_rgb = batch[self.clean_rgb_type].to(device)  # [-1, 1]

                batch_size = degraded_rgb.shape[0]

                with torch.no_grad():
                    # Encode degraded RGB to latent
                    degraded_latent = self.encode_rgb(degraded_rgb)  # [B, 4, h, w]
                    # Encode clean RGB to latent (target)
                    clean_latent = self.encode_rgb(clean_rgb)  # [B, 4, h, w]
                    
                    # Latent normalization (if enabled)
                    # Normalize BEFORE noise addition so velocity target is naturally zero-mean and unit variance
                    # Uses spatial mean and std per channel: mean/std(dim=(2, 3), keepdim=True)
                    if self.normalize_latents:
                        # Clean latent normalization
                        clean_latent_mean = clean_latent.mean(dim=(2, 3), keepdim=True)
                        clean_latent_std = clean_latent.std(dim=(2, 3), keepdim=True)
                        clean_latent = (clean_latent - clean_latent_mean) / (clean_latent_std + 1e-8)
                        
                        # Degraded latent normalization
                        degraded_latent_mean = degraded_latent.mean(dim=(2, 3), keepdim=True)
                        degraded_latent_std = degraded_latent.std(dim=(2, 3), keepdim=True)
                        degraded_latent = (degraded_latent - degraded_latent_mean) / (degraded_latent_std + 1e-8)

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # DISABLED INVERSE annealing: more multi-res noise at LOW timesteps (refinement phase)
                        # This differs from original Marigold (depth/normals/iid) which uses direct annealing.
                        # Rationale: Multi-resolution noise is most beneficial when the model refines details (low t),
                        # not when the signal is already dominated by noise (high t).
                        # See: thesis-docs/multi-res-noise-annealing-analysis.md
                        # strength = strength * (1.0 - timesteps / self.scheduler_timesteps)
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        clean_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        clean_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Apply offset noise to prevent latent drift toward mean values
                # Adds a spatially constant offset per sample to help model handle extreme brightness
                # Reference: https://www.crosslabs.org/blog/diffusion-with-offset-noise
                if self.offset_noise_strength > 0.0:
                    offset = torch.randn(
                        batch_size, clean_latent.shape[1], 1, 1,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, 1, 1] - same offset across spatial dimensions
                    noise = noise + self.offset_noise_strength * offset

                # Add noise to the clean latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    clean_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Conditioning: ARNIQA quality features or empty text embedding
                if self.use_arniqa:
                    # Extract quality features from degraded image
                    # Output: [B, 1, 1024] - single quality token per sample
                    text_embed = self.arniqa_conditioner(degraded_rgb, apply_dropout=True)
                    # Capture ARNIQA output statistics for logging
                    with torch.no_grad():
                        self._arniqa_output_stats = {
                            'output_norm': text_embed.norm().item(),
                            'output_mean': text_embed.mean().item(),
                            'output_std': text_embed.std().item(),
                        }
                else:
                    # Fallback: empty text embedding
                    text_embed = self.empty_text_embed.to(device).repeat(
                        (batch_size, 1, 1)
                    )  # [B, 77, 1024]

                # Input noise augmentation: add small noise to conditioning for robustness
                # Applied to a percentage of samples with variable strength
                if self.input_noise_prob > 0.0:
                    # Determine which samples get augmented
                    augment_mask = (torch.rand(batch_size, device=device, generator=rand_num_generator) < self.input_noise_prob)
                    if augment_mask.any():
                        # Variable strength per sample: uniform [0, max_strength]
                        input_noise_strength = torch.rand(batch_size, device=device, generator=rand_num_generator) * self.input_noise_max_strength
                        input_noise_strength = input_noise_strength * augment_mask.float()  # Zero out non-augmented samples
                        input_noise_strength = input_noise_strength.view(batch_size, 1, 1, 1)
                        
                        # Add noise to degraded latent
                        input_noise = torch.randn(
                            degraded_latent.shape,
                            device=device,
                            generator=rand_num_generator,
                        )
                        degraded_latent = degraded_latent + input_noise_strength * input_noise

                # Apply timestep-aware conditioning scaling (if enabled)
                if self.cond_timestep_scaling:
                    cond_scale = self._get_conditioning_scale(timesteps)
                    degraded_latent_scaled = degraded_latent * cond_scale
                else:
                    degraded_latent_scaled = degraded_latent

                # Classifier-Free Guidance (CFG) training: randomly drop conditioning
                # Replace degraded_latent with zeros for unconditional training
                if self.conditioning_dropout_prob > 0.0:
                    # Mask: 1.0 to keep, 0.0 to drop
                    keep_mask = (torch.rand(batch_size, device=device, generator=rand_num_generator) >= self.conditioning_dropout_prob).view(batch_size, 1, 1, 1).to(degraded_latent_scaled.dtype)
                    degraded_latent_scaled = degraded_latent_scaled * keep_mask

                # Concat degraded and noisy clean latents
                cat_latents = torch.cat(
                    [degraded_latent_scaled, noisy_latents], dim=1
                )  # [B, 8, h, w]
                cat_latents = cat_latents.float()

                # Forward pass with mixed precision (if enabled)
                if self.mixed_precision:
                    with autocast('cuda'):
                        # Predict the noise residual in FP16
                        model_pred = self.model.unet(
                            cat_latents, timesteps, text_embed
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
                    
                    # Loss computation (outside autocast for FP32 stability)
                    if hasattr(self.loss, '__class__') and 'CombinedRestorationLoss' in self.loss.__class__.__name__:
                        # Combined loss: noise loss (latent space) + image loss (RGB space)
                        # 1. Noise loss in FP32 (configurable: mse or l1)
                        noise_loss = self._compute_noise_loss(model_pred, target)
                        
                        # 2. Get the denoised latent from model prediction (same as FP32 path)
                        if "epsilon" == self.prediction_type:
                            # Use scheduler to get predicted original sample
                            alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                            beta_prod_t = 1 - alpha_prod_t
                            
                            # Reshape for broadcasting
                            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                            beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                            
                            # Predict original sample (denoised latent)
                            pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                        elif "sample" == self.prediction_type:
                            pred_original_sample = model_pred
                        elif "v_prediction" == self.prediction_type:
                            # For v_prediction: x0 = alpha_t * noisy_latents - sigma_t * model_pred
                            alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                            beta_prod_t = 1 - alpha_prod_t
                            
                            # Reshape for broadcasting
                            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                            beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                            
                            # Predict original sample using v_prediction formula
                            pred_original_sample = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
                        else:
                            raise ValueError(f"Unknown prediction type {self.prediction_type}")
                        
                        # Get loss weights
                        noise_weight = getattr(self.cfg.loss.kwargs, 'noise_weight', 1.0)
                        elatentlpips_weight = getattr(self.cfg.loss.kwargs, 'elatentlpips_weight', 0.0)
                        image_weight = getattr(self.cfg.loss.kwargs, 'image_weight', 0.1)
                        
                        # Start with noise loss
                        loss = noise_weight * noise_loss
                        loss_components_dict = {'noise_loss': noise_loss.item()}
                        
                        # E-LatentLPIPS loss (only if weight > 0) - in FP16 for memory efficiency
                        if elatentlpips_weight > 0:
                            with autocast('cuda'):
                                elatentlpips_loss = self.loss.compute_elatentlpips(pred_original_sample, clean_latent)
                            loss = loss + elatentlpips_weight * elatentlpips_loss
                            loss_components_dict['elatentlpips'] = elatentlpips_loss.item()
                        
                        # Image loss (only if weight > 0) - includes VAE decoding
                        if image_weight > 0:
                            restored_rgb = self.decode_rgb(pred_original_sample.float())
                            clean_rgb_original = (clean_rgb + 1.0) / 2.0
                            clean_rgb_original = torch.clamp(clean_rgb_original, 0.0, 1.0)
                            image_loss, loss_components = self.loss(restored_rgb, clean_rgb_original)
                            loss = loss + image_weight * image_loss
                            loss_components_dict['image_loss'] = image_loss.item()
                            loss_components_dict.update({f'image_{k}': v.item() for k, v in loss_components.items() if k != 'total'})
                        
                        self._last_loss_components = loss_components_dict
                    else:
                        # Standard loss
                        latent_loss = self.loss(model_pred.float(), target.float())
                        loss = latent_loss.mean()
                        self._last_loss_components = {}
                        
                        # Compute pred_original_sample if needed by gradient loss, pixel LPIPS, or DINO
                        _need_pred_original = (
                            (self.gradient_loss is not None and self.gradient_weight > 0)
                            or (self.lpips_loss is not None and self.lpips_weight > 0)
                            or (self.dino_loss is not None and self.dino_weight > 0)
                        )
                        if _need_pred_original:
                            if "epsilon" == self.prediction_type:
                                alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                                beta_prod_t = 1 - alpha_prod_t
                                alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                                beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                                pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                            elif "sample" == self.prediction_type:
                                pred_original_sample = model_pred
                                alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                            elif "v_prediction" == self.prediction_type:
                                alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                                beta_prod_t = 1 - alpha_prod_t
                                alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                                beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                                pred_original_sample = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
                            else:
                                raise ValueError(f"Unknown prediction type {self.prediction_type}")
                        
                        # Latent gradient loss (if enabled) - penalizes loss of edges/sharpness
                        if self.gradient_loss is not None and self.gradient_weight > 0:
                            snr_weights = None
                            grad_loss = self.gradient_loss(pred_original_sample.float(), clean_latent.float(), sample_weights=snr_weights)
                            loss = loss + self.gradient_weight * grad_loss
                            self._last_loss_components['gradient_loss'] = grad_loss.item()
                            
                            # Store latents for gradient map visualization (detached, no grad)
                            self._last_pred_original_sample = pred_original_sample.detach()
                            self._last_clean_latent = clean_latent.detach()
                        
                        # Pixel-space LPIPS loss (if enabled) - VAE decode on random crop
                        if self.lpips_loss is not None and self.lpips_weight > 0:
                            latent_crop = self.pixel_loss_crop_size // 8
                            _, _, lh, lw = pred_original_sample.shape
                            
                            if latent_crop < lh and latent_crop < lw:
                                top = torch.randint(0, lh - latent_crop, (1,), generator=rand_num_generator, device=device).item()
                                left = torch.randint(0, lw - latent_crop, (1,), generator=rand_num_generator, device=device).item()
                            else:
                                top, left = 0, 0
                            
                            pred_crop = pred_original_sample[:, :, top:top+latent_crop, left:left+latent_crop]
                            clean_crop = clean_latent[:, :, top:top+latent_crop, left:left+latent_crop]
                            
                            # VAE decode crops to pixel space
                            # No torch.no_grad(): gradients must flow through VAE decode and LPIPS
                            # back to pred_original_sample → model_pred → UNet weights.
                            # VAE params are frozen (requires_grad=False) so no VAE weight updates.
                            pred_rgb_crop = self.decode_rgb(pred_crop.float())
                            clean_rgb_crop = self.decode_rgb(clean_crop.float())
                            
                            # LPIPS expects [-1, 1]; decode_rgb returns [0, 1]
                            pred_lpips_in = pred_rgb_crop * 2.0 - 1.0
                            clean_lpips_in = clean_rgb_crop * 2.0 - 1.0
                            pixel_lpips = self.lpips_loss(pred_lpips_in, clean_lpips_in).mean()
                            loss = loss + self.lpips_weight * pixel_lpips
                            self._last_loss_components['pixel_lpips'] = pixel_lpips.item()
                            
                            # DINO reuses the same decoded crops (already in [-1, 1])
                            if self.dino_loss is not None and self.dino_weight > 0:
                                pixel_dino = self.dino_loss(pred_lpips_in, clean_lpips_in)
                                loss = loss + self.dino_weight * pixel_dino
                                self._last_loss_components['pixel_dino'] = pixel_dino.item()
                        
                        # Pixel-space DINO loss (standalone, when LPIPS is disabled)
                        elif self.dino_loss is not None and self.dino_weight > 0:
                            latent_crop = self.pixel_loss_crop_size // 8
                            _, _, lh, lw = pred_original_sample.shape
                            
                            if latent_crop < lh and latent_crop < lw:
                                top = torch.randint(0, lh - latent_crop, (1,), generator=rand_num_generator, device=device).item()
                                left = torch.randint(0, lw - latent_crop, (1,), generator=rand_num_generator, device=device).item()
                            else:
                                top, left = 0, 0
                            
                            pred_crop = pred_original_sample[:, :, top:top+latent_crop, left:left+latent_crop]
                            clean_crop = clean_latent[:, :, top:top+latent_crop, left:left+latent_crop]
                            
                            pred_rgb_crop = self.decode_rgb(pred_crop.float())
                            clean_rgb_crop = self.decode_rgb(clean_crop.float())
                            
                            # DINO expects [-1, 1]; decode_rgb returns [0, 1]
                            pred_dino_in = pred_rgb_crop * 2.0 - 1.0
                            clean_dino_in = clean_rgb_crop * 2.0 - 1.0
                            pixel_dino = self.dino_loss(pred_dino_in, clean_dino_in)
                            loss = loss + self.dino_weight * pixel_dino
                            self._last_loss_components['pixel_dino'] = pixel_dino.item()
                else:
                    # FP32 training (original behavior)
                    # Predict the noise residual
                    model_pred = self.model.unet(
                        cat_latents, timesteps, text_embed
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

                    # Check if using combined loss
                    if hasattr(self.loss, '__class__') and 'CombinedRestorationLoss' in self.loss.__class__.__name__:
                        # Combined loss: noise loss (latent space) + image loss (RGB space)
                        noise_loss = self._compute_noise_loss(model_pred, target)
                        
                        # Get the denoised latent from model prediction
                        # For epsilon prediction: x0 = (noisy - sqrt(1-alpha) * pred_noise) / sqrt(alpha)
                        if "epsilon" == self.prediction_type:
                            # Use scheduler to get predicted original sample
                            alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                            beta_prod_t = 1 - alpha_prod_t
                            
                            # Reshape for broadcasting
                            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                            beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                            
                            # Predict original sample (denoised latent)
                            pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                        elif "sample" == self.prediction_type:
                            pred_original_sample = model_pred
                        elif "v_prediction" == self.prediction_type:
                            # For v_prediction: x0 = alpha_t * noisy_latents - sigma_t * model_pred
                            alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                            beta_prod_t = 1 - alpha_prod_t
                            
                            # Reshape for broadcasting
                            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                            beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                            
                            # Predict original sample using v_prediction formula
                            pred_original_sample = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
                        else:
                            raise ValueError(f"Unknown prediction type {self.prediction_type}")
                        
                        # Get loss weights
                        noise_weight = getattr(self.cfg.loss.kwargs, 'noise_weight', 1.0)
                        elatentlpips_weight = getattr(self.cfg.loss.kwargs, 'elatentlpips_weight', 0.0)
                        image_weight = getattr(self.cfg.loss.kwargs, 'image_weight', 0.1)
                        
                        # Start with noise loss
                        loss = noise_weight * noise_loss
                        loss_components_dict = {'noise_loss': noise_loss.item()}
                        
                        # E-LatentLPIPS loss (only if weight > 0)
                        if elatentlpips_weight > 0:
                            elatentlpips_loss = self.loss.compute_elatentlpips(pred_original_sample, clean_latent)
                            loss = loss + elatentlpips_weight * elatentlpips_loss
                            loss_components_dict['elatentlpips'] = elatentlpips_loss.item()
                        
                        # Image loss (only if weight > 0) - includes VAE decoding
                        if image_weight > 0:
                            restored_rgb = self.decode_rgb(pred_original_sample)
                            clean_rgb_original = (clean_rgb + 1.0) / 2.0
                            clean_rgb_original = torch.clamp(clean_rgb_original, 0.0, 1.0)
                            image_loss, loss_components = self.loss(restored_rgb, clean_rgb_original)
                            loss = loss + image_weight * image_loss
                            loss_components_dict['image_loss'] = image_loss.item()
                            loss_components_dict.update({f'image_{k}': v.item() for k, v in loss_components.items() if k != 'total'})
                        
                        self._last_loss_components = loss_components_dict
                    else:
                        # Standard loss
                        latent_loss = self.loss(model_pred.float(), target.float())
                        loss = latent_loss.mean()
                        self._last_loss_components = {}
                        
                        # Compute pred_original_sample if needed by gradient loss, pixel LPIPS, or DINO
                        _need_pred_original = (
                            (self.gradient_loss is not None and self.gradient_weight > 0)
                            or (self.lpips_loss is not None and self.lpips_weight > 0)
                            or (self.dino_loss is not None and self.dino_weight > 0)
                        )
                        if _need_pred_original:
                            if "epsilon" == self.prediction_type:
                                alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                                beta_prod_t = 1 - alpha_prod_t
                                alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                                beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                                pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                            elif "sample" == self.prediction_type:
                                pred_original_sample = model_pred
                                alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                            elif "v_prediction" == self.prediction_type:
                                alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                                beta_prod_t = 1 - alpha_prod_t
                                alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                                beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                                pred_original_sample = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
                            else:
                                raise ValueError(f"Unknown prediction type {self.prediction_type}")
                        
                        # Latent gradient loss (if enabled) - penalizes loss of edges/sharpness
                        if self.gradient_loss is not None and self.gradient_weight > 0:
                            snr_weights = None
                            grad_loss = self.gradient_loss(pred_original_sample.float(), clean_latent.float(), sample_weights=snr_weights)
                            loss = loss + self.gradient_weight * grad_loss
                            self._last_loss_components['gradient_loss'] = grad_loss.item()
                            
                            # Store latents for gradient map visualization (detached, no grad)
                            self._last_pred_original_sample = pred_original_sample.detach()
                            self._last_clean_latent = clean_latent.detach()
                        
                        # Pixel-space LPIPS loss (if enabled) - VAE decode on random crop
                        if self.lpips_loss is not None and self.lpips_weight > 0:
                            latent_crop = self.pixel_loss_crop_size // 8
                            _, _, lh, lw = pred_original_sample.shape
                            
                            if latent_crop < lh and latent_crop < lw:
                                top = torch.randint(0, lh - latent_crop, (1,), generator=rand_num_generator, device=device).item()
                                left = torch.randint(0, lw - latent_crop, (1,), generator=rand_num_generator, device=device).item()
                            else:
                                top, left = 0, 0
                            
                            pred_crop = pred_original_sample[:, :, top:top+latent_crop, left:left+latent_crop]
                            clean_crop = clean_latent[:, :, top:top+latent_crop, left:left+latent_crop]
                            
                            # VAE decode crops to pixel space
                            # No torch.no_grad(): gradients must flow through VAE decode and LPIPS
                            # back to pred_original_sample → model_pred → UNet weights.
                            # VAE params are frozen (requires_grad=False) so no VAE weight updates.
                            pred_rgb_crop = self.decode_rgb(pred_crop.float())
                            clean_rgb_crop = self.decode_rgb(clean_crop.float())
                            
                            # LPIPS expects [-1, 1]; decode_rgb returns [0, 1]
                            pred_lpips_in = pred_rgb_crop * 2.0 - 1.0
                            clean_lpips_in = clean_rgb_crop * 2.0 - 1.0
                            pixel_lpips = self.lpips_loss(pred_lpips_in, clean_lpips_in).mean()
                            loss = loss + self.lpips_weight * pixel_lpips
                            self._last_loss_components['pixel_lpips'] = pixel_lpips.item()
                            
                            # DINO reuses the same decoded crops (already in [-1, 1])
                            if self.dino_loss is not None and self.dino_weight > 0:
                                pixel_dino = self.dino_loss(pred_lpips_in, clean_lpips_in)
                                loss = loss + self.dino_weight * pixel_dino
                                self._last_loss_components['pixel_dino'] = pixel_dino.item()
                        
                        # Pixel-space DINO loss (standalone, when LPIPS is disabled)
                        elif self.dino_loss is not None and self.dino_weight > 0:
                            latent_crop = self.pixel_loss_crop_size // 8
                            _, _, lh, lw = pred_original_sample.shape
                            
                            if latent_crop < lh and latent_crop < lw:
                                top = torch.randint(0, lh - latent_crop, (1,), generator=rand_num_generator, device=device).item()
                                left = torch.randint(0, lw - latent_crop, (1,), generator=rand_num_generator, device=device).item()
                            else:
                                top, left = 0, 0
                            
                            pred_crop = pred_original_sample[:, :, top:top+latent_crop, left:left+latent_crop]
                            clean_crop = clean_latent[:, :, top:top+latent_crop, left:left+latent_crop]
                            
                            pred_rgb_crop = self.decode_rgb(pred_crop.float())
                            clean_rgb_crop = self.decode_rgb(clean_crop.float())
                            
                            # DINO expects [-1, 1]; decode_rgb returns [0, 1]
                            pred_dino_in = pred_rgb_crop * 2.0 - 1.0
                            clean_dino_in = clean_rgb_crop * 2.0 - 1.0
                            pixel_dino = self.dino_loss(pred_dino_in, clean_dino_in)
                            loss = loss + self.dino_weight * pixel_dino
                            self._last_loss_components['pixel_dino'] = pixel_dino.item()

                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                self.train_metrics.update("loss", loss.item())
                
                # Log loss components if available
                if hasattr(self, '_last_loss_components') and self._last_loss_components:
                    for key, value in self._last_loss_components.items():
                        self.train_metrics.update(key, value)

                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision (if enabled)
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Compute ARNIQA adapter gradient and weight norms (for monitoring learning)
                # Stage 2: Log separately for global_adapter and spatial_adapter
                if self.use_arniqa and self.arniqa_conditioner is not None:
                    # Use the conditioner's built-in methods for per-adapter norms
                    scale_factor = self.scaler.get_scale() if self.mixed_precision else 1.0
                    scale_factor = max(scale_factor, 1e-8)  # Guard against zero after NaN/inf
                    
                    # Get gradient norms (unscaled if using mixed precision)
                    grad_norms = self.arniqa_conditioner.get_adapter_grad_norms()
                    if self.mixed_precision:
                        # Unscale gradients
                        grad_norms = {k: v / scale_factor for k, v in grad_norms.items()}
                    
                    # Get weight norms
                    weight_norms = self.arniqa_conditioner.get_adapter_weight_norms()
                    
                    # Store for logging
                    self._arniqa_grad_norms = grad_norms
                    self._arniqa_weight_norms = weight_norms
                    
                    # Also compute total for backward compatibility
                    self._arniqa_grad_norm = sum(v**2 for v in grad_norms.values()) ** 0.5
                    self._arniqa_weight_norm = sum(v**2 for v in weight_norms.values()) ** 0.5
                
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    if self.mixed_precision:
                        # Unscale gradients before clipping (required for mixed precision)
                        self.scaler.unscale_(self.optimizer)
                        # Gradient clipping to prevent NaN from pixel-space loss backprop
                        torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), max_norm=1.0)
                        if self.use_arniqa and self.arniqa_conditioner is not None:
                            torch.nn.utils.clip_grad_norm_(self.arniqa_conditioner.parameters(), max_norm=1.0)
                        # Mixed precision optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Gradient clipping (FP32)
                        torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), max_norm=1.0)
                        if self.use_arniqa and self.arniqa_conditioner is not None:
                            torch.nn.utils.clip_grad_norm_(self.arniqa_conditioner.parameters(), max_norm=1.0)
                        # FP32 optimizer step
                        self.optimizer.step()
                    
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
                    # Log ARNIQA adapter metrics (if enabled)
                    # These are synced to WandB via sync_tensorboard=True
                    if self.use_arniqa:
                        # Total gradient and weight norms (backward compatibility)
                        if hasattr(self, '_arniqa_grad_norm'):
                            tb_logger.writer.add_scalar(
                                "arniqa/grad_norm", self._arniqa_grad_norm, global_step=self.effective_iter
                            )
                        if hasattr(self, '_arniqa_weight_norm'):
                            tb_logger.writer.add_scalar(
                                "arniqa/weight_norm", self._arniqa_weight_norm, global_step=self.effective_iter
                            )
                        
                        # Per-adapter gradient norms (Stage 2: separate global and spatial)
                        if hasattr(self, '_arniqa_grad_norms'):
                            for adapter_name, grad_norm in self._arniqa_grad_norms.items():
                                tb_logger.writer.add_scalar(
                                    f"arniqa/{adapter_name}_grad_norm", grad_norm, global_step=self.effective_iter
                                )
                        
                        # Per-adapter weight norms
                        if hasattr(self, '_arniqa_weight_norms'):
                            for adapter_name, weight_norm in self._arniqa_weight_norms.items():
                                tb_logger.writer.add_scalar(
                                    f"arniqa/{adapter_name}_weight_norm", weight_norm, global_step=self.effective_iter
                                )
                        
                        # Output statistics
                        if hasattr(self, '_arniqa_output_stats'):
                            tb_logger.writer.add_scalar(
                                "arniqa/output_norm", self._arniqa_output_stats['output_norm'], global_step=self.effective_iter
                            )
                            tb_logger.writer.add_scalar(
                                "arniqa/output_mean", self._arniqa_output_stats['output_mean'], global_step=self.effective_iter
                            )
                            tb_logger.writer.add_scalar(
                                "arniqa/output_std", self._arniqa_output_stats['output_std'], global_step=self.effective_iter
                            )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Log gradient maps to W&B (at visualization period)
                    if (self.gradient_loss is not None 
                            and self.gradient_weight > 0
                            and hasattr(self, '_last_pred_original_sample')
                            and hasattr(self, '_last_clean_latent')
                            and self.effective_iter % self.vis_period == 0):
                        try:
                            grad_maps = self.gradient_loss.visualize_gradients(
                                self._last_pred_original_sample, self._last_clean_latent
                            )
                            # Create 3-panel image: Sobel(clean) | Sobel(pred) | |Difference|
                            grad_comparison = np.concatenate([
                                grad_maps['grad_target'],
                                grad_maps['grad_pred'],
                                grad_maps['grad_diff'],
                            ], axis=1)  # [h, 3*w]
                            
                            # Convert to uint8 for W&B
                            grad_comparison_uint8 = (grad_comparison * 255).astype(np.uint8)
                            grad_img = Image.fromarray(grad_comparison_uint8, mode='L')
                            
                            # Add labels
                            try:
                                from PIL import ImageDraw, ImageFont
                                draw = ImageDraw.Draw(grad_img)
                                h, w_total = grad_comparison_uint8.shape
                                w_panel = w_total // 3
                                font_size = max(10, h // 8)
                                try:
                                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                                except Exception:
                                    font = ImageFont.load_default()
                                labels = ["Sobel(clean)", "Sobel(pred)", "|Difference|"]
                                for idx, label in enumerate(labels):
                                    x_pos = idx * w_panel + 2
                                    bbox = draw.textbbox((x_pos, 2), label, font=font)
                                    draw.rectangle(bbox, fill=0)
                                    draw.text((x_pos, 2), label, fill=255, font=font)
                            except Exception:
                                pass
                            
                            wandb.log({
                                "gradient_maps/train": wandb.Image(grad_img, caption=f"Training gradient maps (iter {self.effective_iter})")
                            }, step=self.effective_iter, commit=False)
                            logging.info(f"Logged training gradient maps to W&B at iter {self.effective_iter}")
                        except Exception as e:
                            logging.warning(f"Failed to log training gradient maps: {e}")

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

                    # torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def encode_rgb(self, image_in):
        """Encode RGB image to latent space"""
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        latent = self.model.encode_rgb(image_in)
        return latent
    
    def decode_rgb(self, latent_in):
        """Decode latent to RGB image [0, 1]"""
        assert len(latent_in.shape) == 4 and latent_in.shape[1] == 4
        # Decode using VAE decoder
        rgb = self.model.decode_rgb(latent_in)
        # Convert from [-1, 1] to [0, 1] range
        rgb = (rgb + 1.0) / 2.0
        # Clamp to [0, 1] range
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb

    def _compute_noise_loss(self, model_pred, target):
        """Compute noise loss in latent space based on configured type (mse or l1)"""
        if self.noise_loss_type == 'l1':
            return F.l1_loss(model_pred.float(), target.float())
        else:  # default: mse
            return F.mse_loss(model_pred.float(), target.float())

    def _train_step_callback(self):
        """Executed after every iteration"""
        if self.checkpoint_mode == 'marigold':
            self._train_step_callback_marigold()
        else:
            self._train_step_callback_decoupled()

    def _train_step_callback_marigold(self):
        """Original Marigold checkpoint pattern (same as depth/normals/iid trainers)"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()  # Inside: saves iter_XXXXXX if best metric improves
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
        """Decoupled checkpoint pattern: independent save/val/vis intervals"""
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

    def validate(self):
        """Validation with checkpoint saving (Marigold mode)"""
        # Check if we should log images during validation
        log_images = getattr(self.cfg.validation, 'log_images_during_validation', True)
        
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dict = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                log_images_to_wandb=log_images,  # Log images during validation
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
            )
            tb_logger.log_dict(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                global_step=self.effective_iter,
            )
            # save to file
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
                    # Save a checkpoint (Marigold mode: saves iter_XXXXXX when best improves)
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def validate_decoupled(self):
        """Validation without checkpoint saving (Decoupled mode)"""
        # Check if we should log images during validation
        log_images = getattr(self.cfg.validation, 'log_images_during_validation', True)
        
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dict = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                log_images_to_wandb=log_images,  # Log images during validation
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
            )
            tb_logger.log_dict(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                global_step=self.effective_iter,
            )
            # save to file
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
                    # Decoupled mode: NO checkpoint saving here, only track best metric

    def visualize(self):
        logging.info(f"Starting visualization at iteration {self.effective_iter}")
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            logging.info(f"Visualizing dataset: {vis_dataset_name} (W&B logging enabled)")
            
            # Run visualization and get metrics
            vis_metrics = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
                log_images_to_wandb=True,  # Enable W&B image logging for visualization
            )
            
            # Save visualization metrics to file
            if vis_metrics is not None:
                from src.util.logging_util import eval_dict_to_text
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
        self.model.to(self.device)
        metric_tracker.reset()

        # Configure pipeline to match training settings
        # CRITICAL: Pipeline must use same normalize_latents setting as training
        self.model.set_normalize_latents(self.normalize_latents)

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        total_images = len(data_loader.dataset)
        val_seed_ls = generate_seed_sequence(val_init_seed, total_images)

        # Track validation loss and components
        val_loss_dicts = []
        
        # W&B image logging configuration
        max_images_to_log = getattr(self.cfg.validation, 'max_images_to_log', 8)
        wandb_images = []
        wandb_image_metrics = []
        wandb_gradient_maps = []

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            batch_size = batch["degraded_rgb_int"].shape[0]
            
            # Read input images - pipeline expects [0, 255] format
            degraded_rgb_int = batch["degraded_rgb_int"]  # [B, 3, H, W] in [0, 255]
            clean_rgb_int = batch["clean_rgb_int"]  # [B, 3, H, W] in [0, 255]

            # Generate seeds for this batch
            batch_seeds = [val_seed_ls.pop() for _ in range(batch_size)]
            
            # Calculate validation loss for each image in batch
            for b_idx in range(batch_size):
                batch_single = {k: v[b_idx:b_idx+1] if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v 
                               for k, v in batch.items()}
                seed = batch_seeds[b_idx]
                generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
                val_loss_dict = self._calculate_validation_loss(batch_single, generator)
                val_loss_dicts.append(val_loss_dict)

            # Predict restored images in parallel batch using single_infer
            # Normalize degraded images to [-1, 1] range
            degraded_rgb_norm = degraded_rgb_int.float() / 255.0 * 2.0 - 1.0  # [B, 3, H, W] in [-1, 1]
            degraded_rgb_norm = degraded_rgb_norm.to(self.device)
            
            # Set up scheduler for inference
            self.model.scheduler.set_timesteps(self.cfg.validation.denoising_steps, device=self.device)
            
            # Process batch through pipeline (single_infer handles batches)
            restored_rgb_batch_ts = self.model.single_infer(
                rgb_in=degraded_rgb_norm,
                num_inference_steps=self.cfg.validation.denoising_steps,
                generator=None,  # Deterministic via seed sequence
                show_pbar=False,
            )  # Returns [B, 3, H, W] in [-1, 1]
            
            # Convert to [0, 1] range and numpy
            restored_rgb_batch_ts = (restored_rgb_batch_ts + 1.0) / 2.0
            restored_rgb_batch_ts = torch.clip(restored_rgb_batch_ts, 0.0, 1.0)
            restored_rgb_batch = restored_rgb_batch_ts.cpu().numpy()  # [B, 3, H, W] in [0, 1]
            
            # Process each image in batch for metrics and logging
            for b_idx in range(batch_size):
                # Get single image tensors
                restored_rgb_ts = torch.from_numpy(restored_rgb_batch[b_idx]).to(self.device)  # [3, H, W]
                clean_single_ts = clean_rgb_int[b_idx].to(self.device).float() / 255.0  # [3, H, W]

                # Evaluate restoration metrics (on GPU for speed)
                sample_metric_dict = {}
                for met_func in self.metric_funcs:
                    _metric_name = met_func.__name__
                    _metric = met_func(restored_rgb_ts, clean_single_ts)
                    sample_metric_dict[_metric_name] = float(_metric)
                    metric_tracker.update(_metric_name, _metric)

                # Pixel-space LPIPS on final restored image (if enabled)
                # restored_rgb_ts is [3, H, W] in [0, 1], need [-1, 1] for LPIPS
                if self.lpips_loss is not None and self.lpips_weight > 0:
                    pred_lpips_in = restored_rgb_ts.unsqueeze(0) * 2.0 - 1.0   # [1, 3, H, W] in [-1, 1]
                    clean_lpips_in = clean_single_ts.unsqueeze(0) * 2.0 - 1.0  # [1, 3, H, W] in [-1, 1]
                    with torch.no_grad():
                        pixel_lpips_val = self.lpips_loss(pred_lpips_in, clean_lpips_in).mean()
                    metric_tracker.update("val_pixel_lpips", pixel_lpips_val.item())

                # Pixel-space DINO on final restored image (if enabled)
                if self.dino_loss is not None and self.dino_weight > 0:
                    pred_dino_in = restored_rgb_ts.unsqueeze(0) * 2.0 - 1.0   # [1, 3, H, W] in [-1, 1]
                    clean_dino_in = clean_single_ts.unsqueeze(0) * 2.0 - 1.0  # [1, 3, H, W] in [-1, 1]
                    with torch.no_grad():
                        pixel_dino_val = self.dino_loss(pred_dino_in, clean_dino_in)
                    metric_tracker.update("val_pixel_dino", pixel_dino_val.item())

                # Save restored image
                if save_to_dir is not None:
                    img_name = batch["rgb_relative_path"][b_idx].replace("/", "_")
                    png_save_path = os.path.join(save_to_dir, f"{img_name}_restored.png")
                    # Convert to PIL Image
                    from PIL import Image
                    restored_pil = Image.fromarray((restored_rgb_batch[b_idx].transpose(1, 2, 0) * 255).astype(np.uint8))
                    restored_pil.save(png_save_path)

                # Log images to W&B (only first N images)
                if log_images_to_wandb and len(wandb_images) < max_images_to_log:
                    # Convert tensors to numpy arrays in [H, W, C] format for W&B
                    clean_np = clean_single_ts.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
                    degraded_np = degraded_rgb_int[b_idx].cpu().numpy().transpose(1, 2, 0) / 255.0  # [H, W, 3]
                    restored_np = restored_rgb_batch[b_idx].transpose(1, 2, 0)  # [H, W, 3]
                    
                    # Calculate degraded image metrics for comparison
                    degraded_single_ts = degraded_rgb_int[b_idx].to(self.device).float() / 255.0  # [3, H, W]
                    degraded_metrics = {}
                    for met_func in self.metric_funcs:
                        _metric_name = met_func.__name__
                        _metric_deg = met_func(degraded_single_ts, clean_single_ts)
                        degraded_metrics[_metric_name] = float(_metric_deg)
                    
                    # Create side-by-side comparison
                    comparison = self._create_comparison_image(clean_np, degraded_np, restored_np)
                    
                    # Get image name
                    img_name = batch["rgb_relative_path"][b_idx].replace("/", "_")
                    
                    # Create caption with metrics (degraded vs restored)
                    caption = f"{img_name}\n"
                    caption += "Degraded → Restored:\n"
                    for metric_name in sample_metric_dict.keys():
                        deg_val = degraded_metrics[metric_name]
                        res_val = sample_metric_dict[metric_name]
                        improvement = res_val - deg_val
                        # For LPIPS, lower is better, so improvement is negative
                        if 'lpips' in metric_name.lower():
                            improvement = -improvement
                        caption += f"{metric_name}: {deg_val:.3f} → {res_val:.3f} ({improvement:+.3f}) | "
                    caption = caption.rstrip(" | ")
                    
                    # Store for batch logging
                    wandb_images.append(wandb.Image(comparison, caption=caption))
                    wandb_image_metrics.append({
                        'image_name': img_name,
                        **{f'restored_{k}': v for k, v in sample_metric_dict.items()},
                        **{f'degraded_{k}': v for k, v in degraded_metrics.items()},
                    })
                    
                    # Gradient map visualization (if gradient loss is enabled)
                    if self.gradient_loss is not None and self.gradient_weight > 0:
                        try:
                            with torch.no_grad():
                                # Encode restored and clean images to latent space
                                # encode_rgb expects [-1, 1] input
                                restored_for_encode = restored_rgb_ts.unsqueeze(0) * 2.0 - 1.0  # [1, 3, H, W] in [-1, 1]
                                clean_for_encode = clean_single_ts.unsqueeze(0) * 2.0 - 1.0  # [1, 3, H, W] in [-1, 1]
                                
                                restored_latent = self.encode_rgb(restored_for_encode)  # [1, 4, h, w]
                                clean_latent_vis = self.encode_rgb(clean_for_encode)  # [1, 4, h, w]
                                
                                # Compute gradient maps
                                grad_maps = self.gradient_loss.visualize_gradients(
                                    restored_latent, clean_latent_vis
                                )
                                
                                # Create 3-panel image: Sobel(clean) | Sobel(restored) | |Difference|
                                grad_comparison = np.concatenate([
                                    grad_maps['grad_target'],
                                    grad_maps['grad_pred'],
                                    grad_maps['grad_diff'],
                                ], axis=1)  # [h, 3*w]
                                
                                grad_uint8 = (grad_comparison * 255).astype(np.uint8)
                                grad_img = Image.fromarray(grad_uint8, mode='L')
                                
                                # Add labels
                                try:
                                    from PIL import ImageDraw, ImageFont
                                    draw = ImageDraw.Draw(grad_img)
                                    h_grad, w_total = grad_uint8.shape
                                    w_panel = w_total // 3
                                    font_size = max(10, h_grad // 8)
                                    try:
                                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                                    except Exception:
                                        font = ImageFont.load_default()
                                    labels = ["Sobel(clean)", "Sobel(restored)", "|Difference|"]
                                    for idx, label in enumerate(labels):
                                        x_pos = idx * w_panel + 2
                                        bbox = draw.textbbox((x_pos, 2), label, font=font)
                                        draw.rectangle(bbox, fill=0)
                                        draw.text((x_pos, 2), label, fill=255, font=font)
                                except Exception:
                                    pass
                                
                                wandb_gradient_maps.append(
                                    wandb.Image(grad_img, caption=f"Gradient maps: {img_name}")
                                )
                        except Exception as e:
                            logging.warning(f"Failed to compute visualization gradient maps: {e}")

        # Calculate average validation loss and components
        if val_loss_dicts:
            # Get all keys from first dict
            loss_keys = val_loss_dicts[0].keys()
            
            # Calculate averages for each component
            avg_losses = {}
            for key in loss_keys:
                values = [d[key] for d in val_loss_dicts]
                avg_losses[key] = sum(values) / len(values)
            
            # Update metric tracker with loss components
            metric_tracker.update('val_loss', avg_losses['total'])
            
            # Add component losses if available
            if 'noise_loss' in avg_losses:
                metric_tracker.update('val_noise_loss', avg_losses['noise_loss'])
            if 'elatentlpips' in avg_losses:
                metric_tracker.update('val_elatentlpips', avg_losses['elatentlpips'])
            if 'image_loss' in avg_losses:
                metric_tracker.update('val_image_loss', avg_losses['image_loss'])
            if 'image_mse' in avg_losses:
                metric_tracker.update('val_image_mse', avg_losses['image_mse'])
            if 'image_l1' in avg_losses:
                metric_tracker.update('val_image_l1', avg_losses['image_l1'])
            if 'image_perceptual' in avg_losses:
                metric_tracker.update('val_image_perceptual', avg_losses['image_perceptual'])
            if 'image_lpips' in avg_losses:
                metric_tracker.update('val_image_lpips', avg_losses['image_lpips'])
            if 'gradient_loss' in avg_losses:
                metric_tracker.update('val_gradient_loss', avg_losses['gradient_loss'])
        
        # Get results from metric tracker
        results = metric_tracker.result()
        
        # Log images to W&B and TensorBoard
        if log_images_to_wandb and wandb_images:
            dataset_name = data_loader.dataset.disp_name
            
            # Log images to W&B directly
            logging.info(f"Logging {len(wandb_images)} images to W&B at iteration {self.effective_iter}")
            try:
                wandb.log({
                    f"visualization/{dataset_name}": wandb_images,
                }, step=self.effective_iter, commit=False)
                
                # Log individual image metrics as a table
                if wandb_image_metrics:
                    # Create table with all available columns (degraded + restored metrics)
                    # Get all unique keys from first metric dict
                    all_keys = list(wandb_image_metrics[0].keys())
                    all_keys.remove('image_name')  # Remove image_name, will add as first column
                    
                    columns = ['image_name'] + sorted(all_keys)  # Sort for consistent ordering
                    data = [[m['image_name']] + [m.get(key, 0.0) for key in sorted(all_keys)] 
                            for m in wandb_image_metrics]
                    
                    table = wandb.Table(columns=columns, data=data)
                    
                    logging.info(f"Logging metrics table to W&B with {len(data)} rows and {len(columns)} columns")
                    wandb.log({
                        f"metrics_table/{dataset_name}": table
                    }, step=self.effective_iter, commit=True)
                    
                logging.info(f"✓ Successfully logged {len(wandb_images)} images to W&B")
                
            except Exception as e:
                logging.warning(f"Failed to log images to W&B: {e}")
                import traceback
                traceback.print_exc()
        
            # Log gradient maps to W&B (separate panel)
            if wandb_gradient_maps:
                try:
                    wandb.log({
                        f"gradient_maps/{dataset_name}": wandb_gradient_maps,
                    }, step=self.effective_iter, commit=False)
                    logging.info(f"✓ Logged {len(wandb_gradient_maps)} gradient maps to W&B")
                except Exception as e:
                    logging.warning(f"Failed to log gradient maps to W&B: {e}")
        
        return results

    def _calculate_validation_loss(self, batch, generator):
        """Calculate validation loss using the same process as training loss
        
        Returns:
            dict: Dictionary with 'total' loss and optional component losses
        """
        device = self.device
        
        # Get data - same as training
        degraded_rgb = batch[self.degraded_rgb_type].to(device)  # [-1, 1]
        clean_rgb = batch[self.clean_rgb_type].to(device)  # [-1, 1]
        batch_size = degraded_rgb.shape[0]

        # Encode to latent space - same as training
        degraded_latent = self.encode_rgb(degraded_rgb)  # [B, 4, h, w]
        clean_latent = self.encode_rgb(clean_rgb)  # [B, 4, h, w]
        
        # Latent normalization (if enabled) - same as training
        # Normalize BEFORE noise addition so velocity target is naturally zero-mean and unit variance
        # Uses spatial mean and std per channel: mean/std(dim=(2, 3), keepdim=True)
        if self.normalize_latents:
            # Clean latent normalization
            clean_latent_mean = clean_latent.mean(dim=(2, 3), keepdim=True)
            clean_latent_std = clean_latent.std(dim=(2, 3), keepdim=True)
            clean_latent = (clean_latent - clean_latent_mean) / (clean_latent_std + 1e-8)
            
            # Degraded latent normalization
            degraded_latent_mean = degraded_latent.mean(dim=(2, 3), keepdim=True)
            degraded_latent_std = degraded_latent.std(dim=(2, 3), keepdim=True)
            degraded_latent = (degraded_latent - degraded_latent_mean) / (degraded_latent_std + 1e-8)

        # Sample random timestep - same as training
        timesteps = torch.randint(
            0,
            self.scheduler_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        ).long()  # [B]

        # Sample noise - same as training
        if self.apply_multi_res_noise:
            strength = self.mr_noise_strength
            if self.annealed_mr_noise:
                # DISABLED INVERSE annealing (see thesis-docs/multi-res-noise-annealing-analysis.md)
                # strength = strength * (1.0 - timesteps / self.scheduler_timesteps)
                strength = strength * (timesteps / self.scheduler_timesteps)
            noise = multi_res_noise_like(
                clean_latent,
                strength=strength,
                downscale_strategy=self.mr_noise_downscale_strategy,
                generator=generator,
                device=device,
            )
        else:
            noise = torch.randn(
                clean_latent.shape,
                device=device,
                generator=generator,
            )  # [B, 4, h, w]

        # Add noise - same as training
        noisy_latents = self.training_noise_scheduler.add_noise(
            clean_latent, noise, timesteps
        )  # [B, 4, h, w]

        # Conditioning: ARNIQA quality features or empty text embedding (same as training)
        if self.use_arniqa:
            # Extract quality features from degraded image (no dropout during validation)
            text_embed = self.arniqa_conditioner(degraded_rgb, apply_dropout=False)
        else:
            text_embed = self.empty_text_embed.to(device).repeat(
                (batch_size, 1, 1)
            )  # [B, 77, 1024]

        # Apply timestep-aware conditioning scaling - same as training
        if self.cond_timestep_scaling:
            cond_scale = self._get_conditioning_scale(timesteps)
            degraded_latent_scaled = degraded_latent * cond_scale
        else:
            degraded_latent_scaled = degraded_latent

        # Concat latents - same as training
        cat_latents = torch.cat(
            [degraded_latent_scaled, noisy_latents], dim=1
        )  # [B, 8, h, w]
        cat_latents = cat_latents.float()

        # Predict noise - same as training
        model_pred = self.model.unet(
            cat_latents, timesteps, text_embed
        ).sample  # [B, 4, h, w]

        # Get target - same as training
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

        # Calculate loss with components - same as training
        loss_dict = {}
        
        if hasattr(self.loss, '__class__') and 'CombinedRestorationLoss' in self.loss.__class__.__name__:
            # Combined loss: noise loss (latent space) + image loss (RGB space)
            noise_loss = self._compute_noise_loss(model_pred, target)
            
            # Get the denoised latent from model prediction
            if "epsilon" == self.prediction_type:
                # Use scheduler to get predicted original sample
                alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                beta_prod_t = 1 - alpha_prod_t
                
                # Reshape for broadcasting
                alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                
                # Predict original sample (denoised latent)
                pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
            elif "sample" == self.prediction_type:
                pred_original_sample = model_pred
            elif "v_prediction" == self.prediction_type:
                # For v_prediction: x0 = alpha_t * noisy_latents - sigma_t * model_pred
                alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                beta_prod_t = 1 - alpha_prod_t
                
                # Reshape for broadcasting
                alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                
                # Predict original sample using v_prediction formula
                pred_original_sample = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
            else:
                raise ValueError(f"Unknown prediction type {self.prediction_type}")
            
            # Decode predicted (restored) image and use original clean RGB
            restored_rgb_decoded = self.decode_rgb(pred_original_sample)
            # Use original clean RGB from batch (convert from [-1,1] to [0,1])
            clean_rgb_original = (clean_rgb + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
            clean_rgb_original = torch.clamp(clean_rgb_original, 0.0, 1.0)
            
            # Get loss weights
            elatentlpips_weight = getattr(self.cfg.loss.kwargs, 'elatentlpips_weight', 0.0)
            image_weight = getattr(self.cfg.loss.kwargs, 'image_weight', 0.1)
            
            # Start with noise loss
            total_loss = noise_loss
            loss_dict['noise_loss'] = noise_loss.item()
            
            # E-LatentLPIPS loss (only if weight > 0)
            if elatentlpips_weight > 0:
                elatentlpips_loss = self.loss.compute_elatentlpips(pred_original_sample, clean_latent)
                total_loss = total_loss + elatentlpips_weight * elatentlpips_loss
                loss_dict['elatentlpips'] = elatentlpips_loss.item()
            
            # Image loss (only if weight > 0)
            if image_weight > 0:
                image_loss, loss_components = self.loss(restored_rgb_decoded, clean_rgb_original)
                total_loss = total_loss + image_weight * image_loss
                loss_dict['image_loss'] = image_loss.item()
                # Add individual image loss components
                for k, v in loss_components.items():
                    if k != 'total':
                        loss_dict[f'image_{k}'] = v.item()
            
            loss_dict['total'] = total_loss.item()
        else:
            # Standard loss
            latent_loss = self.loss(model_pred.float(), target.float())
            loss = latent_loss.mean()
            loss_dict['total'] = loss.item()
            
            # Latent gradient loss (if enabled) - penalizes loss of edges/sharpness
            if self.gradient_loss is not None and self.gradient_weight > 0:
                # Compute pred_original_sample (denoised latent) from model prediction
                if "epsilon" == self.prediction_type:
                    alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                    beta_prod_t = 1 - alpha_prod_t
                    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                    beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                    pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                elif "sample" == self.prediction_type:
                    pred_original_sample = model_pred
                    alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                elif "v_prediction" == self.prediction_type:
                    alpha_prod_t = self.training_noise_scheduler.alphas_cumprod[timesteps]
                    beta_prod_t = 1 - alpha_prod_t
                    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                    beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
                    pred_original_sample = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")
                
                # SNR-weighted gradient loss disabled: uniform weighting across all timesteps
                # snr_weights = alpha_prod_t.view(-1)  # [B] — was suppressing gradient loss at high t
                snr_weights = None
                grad_loss = self.gradient_loss(pred_original_sample.float(), clean_latent.float(), sample_weights=snr_weights)
                total_with_grad = loss + self.gradient_weight * grad_loss
                loss_dict['total'] = total_with_grad.item()
                loss_dict['gradient_loss'] = grad_loss.item()

        return loss_dict

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        # Check if space-optimized mode is enabled
        if self.checkpoint_test_config and self.checkpoint_test_config.get('auto_cleanup', False):
            self._check_disk_space_and_cleanup()
        
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        
        # Backup previous checkpoint (unless in space-optimized mode)
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            if self.checkpoint_test_config and self.checkpoint_test_config.get('keep_only_latest_best', False):
                # In space-optimized mode, just remove old checkpoint
                logging.info(f"Removing old checkpoint: {ckpt_dir}")
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            else:
                # Normal mode: backup old checkpoint
                temp_ckpt_dir = os.path.join(
                    os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
                )
                if os.path.exists(temp_ckpt_dir):
                    shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
                os.rename(ckpt_dir, temp_ckpt_dir)
                logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=True)
        logging.info(f"UNet is saved to: {unet_path}")

        # Save scheduler (unless in U-Net only mode)
        if not (self.checkpoint_test_config and self.checkpoint_test_config.get('save_unet_only', False)):
            scheduler_path = os.path.join(ckpt_dir, "scheduler")
            self.model.scheduler.save_pretrained(scheduler_path)
            logging.info(f"Scheduler is saved to: {scheduler_path}")
        else:
            logging.info("Skipping scheduler save (U-Net only mode)")

        # Save restoration config (inference-relevant settings)
        # This allows the pipeline to automatically configure itself when loading
        restoration_config = {
            "normalize_latents": self.normalize_latents,
        }
        restoration_config_path = os.path.join(ckpt_dir, "restoration_config.json")
        with open(restoration_config_path, "w") as f:
            json.dump(restoration_config, f, indent=2)
        logging.info(f"Restoration config saved to: {restoration_config_path}")

        # Save ARNIQA adapter weights (if enabled)
        # Format supports Stage 1 (global-only) and Stage 2 (global+spatial)
        if self.use_arniqa and self.arniqa_conditioner is not None:
            stage_str = f"stage{self.arniqa_stage}"
            arniqa_checkpoint = {
                "stage": stage_str,
                "version": "2.0",  # Version 2.0 supports Stage 2
                "config": {
                    "output_dim": self.arniqa_conditioner.output_dim,
                    "conditioning_dropout": self.arniqa_conditioner.conditioning_dropout,
                    "stage": self.arniqa_stage,
                    "spatial_size": self.arniqa_conditioner.spatial_size if self.arniqa_stage >= 2 else None,
                    "num_tokens": self.arniqa_conditioner.num_tokens,
                },
                "state_dict": self.arniqa_conditioner.state_dict(),
            }
            arniqa_adapter_path = os.path.join(ckpt_dir, "arniqa_adapter.pt")
            torch.save(arniqa_checkpoint, arniqa_adapter_path)
            logging.info(f"ARNIQA adapter ({stage_str}) saved to: {arniqa_adapter_path}")

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
            }
            # Save scaler state if using mixed precision
            if self.mixed_precision and self.scaler is not None:
                state["scaler"] = self.scaler.state_dict()
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
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet weights into existing UNet (not create new one!)
        # This keeps the optimizer connected to the same UNet object
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.safetensors")
        
        # Load from safetensors (since we save with safe_serialization=True)
        from safetensors.torch import load_file
        state_dict = load_file(_model_path)
        
        # Update existing UNet weights (optimizer stays connected!)
        self.model.unet.load_state_dict(state_dict)
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters loaded from {_model_path}")

        # Load ARNIQA adapter weights (if enabled and file exists)
        # Handles stage compatibility: Stage 1 checkpoint can be loaded into Stage 2 model
        # Note: For Stage 2, we always start fresh (joint optimization), so stage mismatch
        # should only happen if resuming a Stage 2 run or loading Stage 1 for comparison
        if self.use_arniqa and self.arniqa_conditioner is not None:
            arniqa_adapter_path = os.path.join(ckpt_path, "arniqa_adapter.pt")
            if os.path.exists(arniqa_adapter_path):
                arniqa_ckpt = torch.load(arniqa_adapter_path, map_location=self.device)
                ckpt_stage = arniqa_ckpt.get("stage", "stage1")
                current_stage = f"stage{self.arniqa_stage}"
                
                if ckpt_stage == current_stage:
                    # Same stage: load everything
                    self.arniqa_conditioner.load_state_dict(arniqa_ckpt["state_dict"])
                    logging.info(f"ARNIQA adapter ({ckpt_stage}) loaded from {arniqa_adapter_path}")
                else:
                    # Different stages: load compatible parts only (global_adapter is common to all stages)
                    logging.warning(f"ARNIQA stage mismatch: checkpoint={ckpt_stage}, current={current_stage}")
                    current_state = self.arniqa_conditioner.state_dict()
                    loaded_count = 0
                    skipped_count = 0
                    for key, value in arniqa_ckpt["state_dict"].items():
                        if key in current_state:
                            if current_state[key].shape == value.shape:
                                current_state[key] = value
                                loaded_count += 1
                            else:
                                logging.debug(f"Shape mismatch for {key}: ckpt={value.shape}, current={current_state[key].shape}")
                                skipped_count += 1
                        else:
                            logging.debug(f"Key {key} not in current model, skipping")
                            skipped_count += 1
                    self.arniqa_conditioner.load_state_dict(current_state)
                    logging.info(
                        f"ARNIQA adapter loaded from {ckpt_stage} checkpoint: "
                        f"{loaded_count} tensors loaded, {skipped_count} skipped (partial load)"
                    )
            else:
                logging.warning(f"ARNIQA adapter file not found at {arniqa_adapter_path}, using initialized weights")

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
            
            # Load scaler state if using mixed precision
            if self.mixed_precision and self.scaler is not None and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
                logging.info(f"Mixed precision scaler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"

    def _create_comparison_image(self, clean_np, degraded_np, restored_np):
        """
        Create a side-by-side comparison image: Clean | Degraded | Restored
        
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
        
        # Add text labels (optional, requires PIL ImageDraw)
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(comparison_img)
            
            # Use default font
            font_size = max(12, h // 40)  # Scale font with image size
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Add labels
            labels = ["Clean (GT)", "Degraded (Input)", "Restored (Output)"]
            for idx, label in enumerate(labels):
                x_pos = idx * w + 10
                y_pos = 10
                # Draw text with background for visibility
                bbox = draw.textbbox((x_pos, y_pos), label, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0, 128))
                draw.text((x_pos, y_pos), label, fill=(255, 255, 255), font=font)
        except Exception as e:
            # If text drawing fails, just return the comparison without labels
            logging.debug(f"Could not add text labels to comparison image: {e}")
        
        return comparison_img

    def _check_disk_space_and_cleanup(self):
        """Check available disk space and cleanup if necessary (test mode only)"""
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
                    # Check again after cleanup
                    total, used, free = disk_util.disk_usage(self.out_dir_ckpt)
                    free_gb = free / (1024**3)
                    if free_gb < min_free_gb:
                        logging.error(f"Still low on space after cleanup: {free_gb:.1f} GB")
                    else:
                        logging.info(f"Cleanup successful: {free_gb:.1f} GB free")
        except Exception as e:
            logging.warning(f"Could not check disk space: {e}")

    def _log_checkpoint_info(self, ckpt_dir):
        """Log checkpoint size and disk usage information (test mode only)"""
        try:
            # Calculate checkpoint size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(ckpt_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            size_gb = total_size / (1024**3)
            
            # Get disk usage
            import shutil as disk_util
            total, used, free = disk_util.disk_usage(self.out_dir_ckpt)
            free_gb = free / (1024**3)
            
            logging.info(f"Checkpoint saved: {os.path.basename(ckpt_dir)}")
            logging.info(f"  - Size: {size_gb:.2f} GB")
            logging.info(f"  - Free space: {free_gb:.1f} GB")
            
        except Exception as e:
            logging.warning(f"Could not log checkpoint info: {e}")

    def _cleanup_old_checkpoints(self, current_ckpt_name=None):
        """Remove old checkpoints to save space (test mode only)"""
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
            
            # Always keep latest (current)
            if current_ckpt_name:
                keep_dirs.add(current_ckpt_name)
            elif ckpt_dirs:
                keep_dirs.add(ckpt_dirs[0])  # Most recent
            
            # Keep best checkpoint (if we know which one it is)
            if hasattr(self, 'best_iter') and self.best_iter:
                best_ckpt_name = f"iter_{self.best_iter:06d}"
                keep_dirs.add(best_ckpt_name)
            
            # Remove excess checkpoints
            removed_count = 0
            for ckpt_dir in ckpt_dirs:
                if ckpt_dir not in keep_dirs and len(keep_dirs) + removed_count < max_checkpoints:
                    # This is an old checkpoint, remove it
                    ckpt_path = os.path.join(self.out_dir_ckpt, ckpt_dir)
                    
                    # Calculate size before removal
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