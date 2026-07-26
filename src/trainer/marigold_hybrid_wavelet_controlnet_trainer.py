# MarigoldHybridWaveletControlNetTrainer - Thesis Implementation
#
# Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction:
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# Hybrid-Wavelet: Joint UNet + Wavelet-Conditioned ControlNet Training
# - 8ch UNet (SD2 weights expanded 4→8ch, TRAINABLE)
# - ControlNet (from SD2 UNet, TRAINABLE) — receives DWT(degraded_rgb)
#   instead of raw RGB as conditioning input
# - Optional ARNIQA Stage 2 conditioning (frozen encoder + trainable adapters)
# - Optional frequency-domain loss (wavelet consistency between pred and GT)
#
# Derived from: marigold_hybrid_controlnet_arniqa_003_trainer.py
# Key changes:
# 1. Wavelet decomposition of controlnet_cond (degraded_rgb → DWT subbands)
# 2. ControlNet conditioning embedding input conv replaced (3ch → 12/3/9ch)
# 3. Frequency-domain loss on VAE-decoded crops
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
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from typing import List, Union

from marigold.marigold_hybrid_wavelet_controlnet_pipeline import (
    MarigoldHybridWaveletControlNetPipeline,
    MarigoldRestorationOutput,
)
from src.ARNIQA.model import ArniqaConditioner
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential, CosineAnnealingWarmRestarts
from src.util.metric import MetricTracker
from src.util.seeding import generate_seed_sequence


class MarigoldHybridWaveletControlNetTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldHybridWaveletControlNetPipeline,
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
        self.model: MarigoldHybridWaveletControlNetPipeline = model
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

        # ---- Step 1: Replace UNet conv_in to accept 8 channels ----
        # Pattern from hybrid-003 trainer __init__ (line 82-83)
        if 8 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        # ---- Step 2: Encode empty text prompt ----
        # Verified: MarigoldHybridWaveletControlNetPipeline.encode_empty_text()
        # (same as 003 pipeline line 115-128)
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        # ---- Step 3: XFormers memory-efficient attention ----
        # Pattern from hybrid-003 trainer __init__ (line 93-94)
        self.model.unet.enable_xformers_memory_efficient_attention()
        self.model.controlnet.enable_xformers_memory_efficient_attention()

        # ---- Step 4: Gradient checkpointing ----
        # Pattern from hybrid-003 trainer __init__ (line 97-102)
        self.gradient_checkpointing = self.cfg.trainer.get('gradient_checkpointing', True)
        if self.gradient_checkpointing:
            self.model.unet.enable_gradient_checkpointing()
            self.model.controlnet.enable_gradient_checkpointing()
            logging.info("Gradient checkpointing ENABLED for both UNet and ControlNet")
        else:
            logging.info("Gradient checkpointing disabled")

        # ---- Step 5: Trainability — UNet + ControlNet trainable, VAE + text_encoder frozen ----
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

        # ---- Step 5b: Wavelet conditioning config ----
        # NEW: Parse wavelet_conditioning section from config
        wavelet_cfg = self.cfg.get('wavelet_conditioning', {})
        self.use_wavelet = wavelet_cfg.get('enabled', False)
        self.wavelet_type = wavelet_cfg.get('wavelet_type', 'haar')
        self.wavelet_levels = wavelet_cfg.get('decomposition_levels', 1)
        self.wavelet_subbands = wavelet_cfg.get('subbands', 'all')
        self.wavelet_upsample = wavelet_cfg.get('upsample_to_input_res', True)

        # Determine number of ControlNet conditioning channels
        if self.use_wavelet:
            if self.wavelet_subbands == 'all':
                self.controlnet_cond_channels = 12  # LL(3) + LH(3) + HL(3) + HH(3)
            elif self.wavelet_subbands == 'll_only':
                self.controlnet_cond_channels = 3   # LL only
            elif self.wavelet_subbands == 'hf_only':
                self.controlnet_cond_channels = 9   # LH(3) + HL(3) + HH(3)
            else:
                raise ValueError(
                    f"Unknown wavelet subbands: {self.wavelet_subbands}. "
                    f"Use 'all', 'll_only', or 'hf_only'."
                )

            # Build Haar wavelet filters (fixed, non-trainable)
            # 1-level Haar DWT: convolve with [1,1]/sqrt(2) and [1,-1]/sqrt(2)
            # Applied per-channel using groups=3
            self._build_haar_filters()

            # Replace ControlNet conditioning embedding input conv
            # Verified: ControlNetConditioningEmbedding.conv_in is Conv2d(3, 16, 3, padding=1)
            # (diffusers/models/controlnets/controlnet.py line 82)
            self._replace_controlnet_cond_conv_in()

            logging.info(
                f"Wavelet conditioning ENABLED: type={self.wavelet_type}, "
                f"levels={self.wavelet_levels}, subbands={self.wavelet_subbands}, "
                f"cond_channels={self.controlnet_cond_channels}, "
                f"upsample={self.wavelet_upsample}"
            )
        else:
            self.controlnet_cond_channels = 3
            logging.info("Wavelet conditioning disabled (using raw RGB)")

        # Pass wavelet config to pipeline so single_infer() applies DWT during inference
        # Verified: MarigoldHybridWaveletControlNetPipeline.set_wavelet_config()
        # (pipeline line 124-143)
        self.model.set_wavelet_config(
            enabled=self.use_wavelet,
            subbands=self.wavelet_subbands if self.use_wavelet else 'all',
            upsample=self.wavelet_upsample if self.use_wavelet else True,
        )

        # ---- Step 5c: Reinitialize ControlNet zero convolutions ----
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

        # ---- Step 6: ARNIQA quality-aware conditioning (optional) ----
        # Pattern from hybrid-003 trainer __init__ (lines 143-178)
        arniqa_cfg = self.cfg.get('arniqa', {})
        self.use_arniqa = arniqa_cfg.get('enabled', False)
        self.arniqa_conditioner = None
        self.arniqa_stage = 1

        if self.use_arniqa:
            arniqa_dropout = arniqa_cfg.get('conditioning_dropout', 0.1)
            arniqa_num_tokens = arniqa_cfg.get('num_tokens', 1)
            self.arniqa_stage = arniqa_cfg.get('stage', 1)
            arniqa_spatial_size = arniqa_cfg.get('spatial_size', 24)

            # Verified: ArniqaConditioner.__init__ signature (model.py line 625-670)
            self.arniqa_conditioner = ArniqaConditioner(
                output_dim=1024,
                conditioning_dropout=arniqa_dropout,
                stage=self.arniqa_stage,
                spatial_size=arniqa_spatial_size,
                num_tokens=arniqa_num_tokens,
            ).to(device)
            # Verified: pipeline.set_arniqa_conditioner() (pipeline line 131-142)
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

        # Gradient explosion protection params (added after train-015 failure)
        if self.use_arniqa:
            self.arniqa_spatial_max_grad_norm = arniqa_cfg.get('spatial_adapter_max_grad_norm', 0.1)
            self.arniqa_output_max_norm = arniqa_cfg.get('output_max_norm', 500.0)
            logging.info(
                f"ARNIQA gradient protection: "
                f"spatial_adapter_max_grad_norm={self.arniqa_spatial_max_grad_norm}, "
                f"output_max_norm={self.arniqa_output_max_norm}"
            )
        else:
            self.arniqa_spatial_max_grad_norm = None
            self.arniqa_output_max_norm = None

        # ---- Step 7: Optimizer — 3-way differential LR ----
        # Pattern from hybrid-003 trainer __init__ (lines 181-218)
        lr = self.cfg.lr  # UNet LR
        controlnet_lr = self.cfg.get('controlnet_lr', lr)  # ControlNet LR

        param_groups = [
            {'params': list(self.model.unet.parameters()), 'lr': lr, 'name': 'unet'},
            {'params': list(self.model.controlnet.parameters()), 'lr': controlnet_lr, 'name': 'controlnet'},
        ]

        if self.use_arniqa and self.arniqa_conditioner is not None:
            arniqa_global_lr = arniqa_cfg.get('adapter_lr', 1e-4)
            arniqa_spatial_lr = arniqa_cfg.get('spatial_adapter_lr', arniqa_global_lr)

            # Verified: ArniqaConditioner.get_parameter_groups() (model.py line 758-788)
            arniqa_param_groups = self.arniqa_conditioner.get_parameter_groups(
                global_adapter_lr=arniqa_global_lr,
                spatial_adapter_lr=arniqa_spatial_lr,
            )
            param_groups += arniqa_param_groups

            if self.arniqa_stage >= 2:
                logging.info(
                    f"Optimizer: UNet LR={lr}, ControlNet LR={controlnet_lr}, "
                    f"ARNIQA global_adapter LR={arniqa_global_lr}, "
                    f"ARNIQA spatial_adapter LR={arniqa_spatial_lr}"
                )
            else:
                logging.info(
                    f"Optimizer: UNet LR={lr}, ControlNet LR={controlnet_lr}, "
                    f"ARNIQA adapter LR={arniqa_global_lr}"
                )
        else:
            logging.info(f"Optimizer: UNet LR={lr}, ControlNet LR={controlnet_lr}")

        self.optimizer = Adam(param_groups)

        # ---- Step 8: LR scheduler ----
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

        # ---- Step 9: Mixed Precision Training (FP16) ----
        # Pattern from hybrid-003 trainer __init__ (line 247-248)
        self.scaler = GradScaler()
        logging.info("Mixed precision training (FP16) enabled via GradScaler")

        # ---- Step 10: Loss function ----
        # Pattern from hybrid-003 trainer __init__ (lines 251-255)
        trainer_only_params = {'lpips_weight', 'pixel_loss_weight', 'pixel_loss_type', 'pixel_loss_max_timestep', 'pixel_loss_max_value'}
        loss_kwargs = {k: v for k, v in self.cfg.loss.kwargs.items()
                       if v is not None and k not in trainer_only_params}
        logging.info(f"Loss kwargs: {loss_kwargs}")
        self.loss = get_loss(loss_name=self.cfg.loss.name, **loss_kwargs)

        # ---- Step 11: Pixel-space LPIPS loss (optional) ----
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

        # ---- Step 11b: Pixel-space reconstruction loss (optional) ----
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

        # ---- Step 11c: Pixel-space loss timestep threshold ----
        # Pattern from hybrid-003 trainer __init__ (lines 292-300)
        self.pixel_loss_max_timestep = self.cfg.loss.kwargs.get('pixel_loss_max_timestep', 1000)
        if self.pixel_loss_max_timestep < 1000:
            logging.info(
                f"Pixel-space loss timestep threshold: {self.pixel_loss_max_timestep} "
                f"(losses disabled above this timestep)"
            )
        else:
            logging.info("Pixel-space loss timestep threshold: disabled (no limit)")

        # ---- Step 11d: Pixel-space loss value clamping ----
        # Pattern from hybrid-003 trainer __init__ (lines 303-310)
        self.pixel_loss_max_value = self.cfg.loss.kwargs.get('pixel_loss_max_value', None)
        if self.pixel_loss_max_value is not None:
            logging.info(
                f"Pixel-space loss value clamping ENABLED: max={self.pixel_loss_max_value}"
            )
        else:
            logging.info("Pixel-space loss value clamping: disabled (no limit)")

        # ---- Step 11e: Frequency-domain loss (NEW) ----
        # Wavelet-domain consistency loss between predicted and ground-truth clean images
        freq_loss_cfg = self.cfg.get('frequency_loss', {})
        self.use_freq_loss = freq_loss_cfg.get('enabled', False)
        self.freq_loss_weight = freq_loss_cfg.get('weight', 0.1)

        if self.use_freq_loss:
            self.freq_loss_wavelet_type = freq_loss_cfg.get('wavelet_type', 'haar')
            subband_weights = freq_loss_cfg.get('subband_weights', {})
            self.freq_subband_weights = {
                'll': subband_weights.get('ll', 1.0),
                'lh': subband_weights.get('lh', 1.0),
                'hl': subband_weights.get('hl', 1.0),
                'hh': subband_weights.get('hh', 1.0),
            }
            # Build frequency loss wavelet filters (may differ from conditioning filters)
            self._build_freq_loss_haar_filters()
            logging.info(
                f"Frequency-domain loss ENABLED: weight={self.freq_loss_weight}, "
                f"wavelet={self.freq_loss_wavelet_type}, "
                f"subband_weights={self.freq_subband_weights}"
            )
        else:
            logging.info("Frequency-domain loss disabled")

        # ---- Step 12: Training noise scheduler ----
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

        # ---- Step 13: Eval metrics ----
        # Pattern from hybrid-003 trainer __init__ (line 341)
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        # ---- Step 14: Train and validation metrics ----
        # Pattern from hybrid-003 trainer __init__ (lines 344-357)
        train_metric_keys = ["loss"]
        if self.lpips_weight > 0:
            train_metric_keys.append("pixel_lpips")
        if self.pixel_loss_weight > 0:
            train_metric_keys.append("pixel_reconstruction")
        if self.use_freq_loss:
            train_metric_keys.append("freq_loss")
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

        # ---- Step 15: Settings ----
        # Pattern from hybrid-003 trainer __init__ (lines 366-376)
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.degraded_rgb_type = getattr(self.cfg, 'degraded_rgb_type', 'degraded_rgb_norm')
        self.clean_rgb_type = getattr(self.cfg, 'clean_rgb_type', 'clean_rgb_norm')
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # ---- Step 16: Offset noise ----
        # Pattern from hybrid-003 trainer __init__ (lines 379-384)
        offset_noise_cfg = self.cfg.get('offset_noise', {})
        self.offset_noise_strength = offset_noise_cfg.get('strength', 0.0)
        if self.offset_noise_strength > 0.0:
            logging.info(f"Offset noise ENABLED - strength: {self.offset_noise_strength}")
        else:
            logging.info("Offset noise disabled (strength = 0.0)")

        # ---- Step 17: Input noise augmentation ----
        # Pattern from hybrid-003 trainer __init__ (lines 387-398)
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

        # ---- Step 18: CFG dropout ----
        # Pattern from hybrid-003 trainer __init__ (lines 401-408)
        cfg_dropout_cfg = self.cfg.get('cfg_dropout', {})
        self.cfg_dropout_prob = cfg_dropout_cfg.get('probability', 0.0)
        if self.cfg_dropout_prob > 0.0:
            logging.info(f"CFG dropout ENABLED - probability: {self.cfg_dropout_prob:.1%}")
        else:
            logging.info("CFG dropout disabled (probability = 0.0)")

        # ---- Step 19: Internal variables ----
        # Pattern from hybrid-003 trainer __init__ (lines 411-415)
        self.epoch = 1
        self.n_batch_in_epoch = 0
        self.effective_iter = 0
        self.in_evaluation = False
        self.global_seed_sequence: List = []

        # ---- Step 20: Checkpoint strategy ----
        # Pattern from hybrid-003 trainer __init__ (lines 418-440)
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

        self.save_trainer_state = self.cfg.trainer.get('save_trainer_state', True)
        logging.info(f"Save trainer state in checkpoints: {self.save_trainer_state}")

        # Checkpoint test configuration for disk space management
        self.checkpoint_test_config = getattr(cfg, 'checkpoint_test', None)
        if self.checkpoint_test_config:
            logging.info("Space-optimized checkpoint settings enabled:")
            logging.info(f"  - Keep only latest+best: {self.checkpoint_test_config.get('keep_only_latest_best', False)}")
            logging.info(f"  - Auto cleanup: {self.checkpoint_test_config.get('auto_cleanup', False)}")
            logging.info(f"  - Max checkpoints: {self.checkpoint_test_config.get('max_checkpoints', 2)}")
            logging.info(f"  - Min free space: {self.checkpoint_test_config.get('min_free_space_gb', 5.0)} GB")

    # ------------------------------------------------------------------
    # Wavelet helper methods (NEW)
    # ------------------------------------------------------------------

    def _build_haar_filters(self):
        """Build Haar wavelet decomposition filters for ControlNet conditioning.

        Creates fixed (non-trainable) convolution filters that compute 1-level
        Haar DWT on a 3-channel RGB image:
          [B, 3, H, W] → [B, 12, H/2, W/2]  (4 subbands × 3 channels)

        Subbands: LL (low-low), LH (low-high), HL (high-low), HH (high-high)
        Each subband has 3 channels (one per RGB channel).

        Uses depthwise convolution (groups=3) to process each channel independently.
        """
        # Haar wavelet filters (unnormalized: [1,1] and [1,-1])
        # Normalization factor: 1/2 for 2D (product of 1/sqrt(2) × 1/sqrt(2))
        inv_sqrt2 = 1.0 / (2.0 ** 0.5)

        # 2D separable Haar filters (applied as 2×2 kernels with stride 2)
        # LL = [+1, +1; +1, +1] / 2  (average)
        # LH = [+1, +1; -1, -1] / 2  (horizontal detail)
        # HL = [+1, -1; +1, -1] / 2  (vertical detail)
        # HH = [+1, -1; -1, +1] / 2  (diagonal detail)
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5

        # Stack into [4, 1, 2, 2] filter bank
        filters_1ch = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]

        # Expand for 3 RGB channels using depthwise convolution (groups=3)
        # Need [12, 1, 2, 2] filters with groups=3 → each filter processes one channel
        filters_3ch = filters_1ch.repeat(3, 1, 1, 1)  # [12, 1, 2, 2]

        # Register as buffer (non-trainable, moves with model to device)
        self.register_buffer_on_device('haar_filters', filters_3ch)
        logging.debug(f"Haar wavelet filters built: shape={filters_3ch.shape}")

    def _build_freq_loss_haar_filters(self):
        """Build Haar wavelet filters for frequency-domain loss computation.

        Same structure as _build_haar_filters but stored separately because
        the frequency loss wavelet type could differ from the conditioning wavelet.
        For now both use Haar, but this keeps the design extensible.
        """
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5

        filters_1ch = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        filters_3ch = filters_1ch.repeat(3, 1, 1, 1)  # [12, 1, 2, 2]

        self.register_buffer_on_device('freq_loss_haar_filters', filters_3ch)
        logging.debug(f"Frequency loss Haar filters built: shape={filters_3ch.shape}")

    def register_buffer_on_device(self, name, tensor):
        """Store a tensor as a non-trainable attribute that moves with the device.

        Unlike nn.Module.register_buffer, this trainer is not an nn.Module,
        so we store the tensor directly and move it in train()/validate().
        """
        setattr(self, name, tensor)

    def _apply_wavelet_decomposition(self, rgb_input):
        """Apply 1-level Haar DWT to an RGB image tensor.

        Args:
            rgb_input: [B, 3, H, W] tensor in [-1, 1]

        Returns:
            Wavelet-decomposed tensor with shape depending on config:
            - subbands='all':     [B, 12, H/2, W/2] or [B, 12, H, W] if upsampled
            - subbands='ll_only': [B, 3, H/2, W/2]  or [B, 3, H, W]  if upsampled
            - subbands='hf_only': [B, 9, H/2, W/2]  or [B, 9, H, W]  if upsampled
        """
        device = rgb_input.device
        B, C, H, W = rgb_input.shape
        assert C == 3, f"Expected 3-channel input, got {C}"

        # Move filters to same device as input
        haar_filters = self.haar_filters.to(device=device, dtype=rgb_input.dtype)

        # Apply depthwise convolution: groups=3 means each RGB channel is
        # convolved independently with the 4 Haar filters
        # Input: [B, 3, H, W], Filters: [12, 1, 2, 2], groups=3
        # Output: [B, 12, H/2, W/2]
        # Channel layout: [R_LL, R_LH, R_HL, R_HH, G_LL, G_LH, G_HL, G_HH, B_LL, B_LH, B_HL, B_HH]
        wavelet_coeffs = F.conv2d(rgb_input, haar_filters, stride=2, groups=3)
        # [B, 12, H/2, W/2]

        # Rearrange to group by subband instead of by channel:
        # [R_LL, G_LL, B_LL, R_LH, G_LH, B_LH, R_HL, G_HL, B_HL, R_HH, G_HH, B_HH]
        # This makes subband selection cleaner
        wavelet_coeffs = wavelet_coeffs.view(B, 3, 4, H // 2, W // 2)  # [B, C=3, S=4, H/2, W/2]
        wavelet_coeffs = wavelet_coeffs.permute(0, 2, 1, 3, 4)  # [B, S=4, C=3, H/2, W/2]
        wavelet_coeffs = wavelet_coeffs.reshape(B, 12, H // 2, W // 2)  # [B, 12, H/2, W/2]
        # Now: channels 0-2 = LL (RGB), 3-5 = LH (RGB), 6-8 = HL (RGB), 9-11 = HH (RGB)

        # Subband selection
        if self.wavelet_subbands == 'all':
            result = wavelet_coeffs  # [B, 12, H/2, W/2]
        elif self.wavelet_subbands == 'll_only':
            result = wavelet_coeffs[:, 0:3, :, :]  # [B, 3, H/2, W/2]
        elif self.wavelet_subbands == 'hf_only':
            result = wavelet_coeffs[:, 3:12, :, :]  # [B, 9, H/2, W/2]
        else:
            raise ValueError(f"Unknown subbands: {self.wavelet_subbands}")

        # Optional upsample back to input resolution
        if self.wavelet_upsample:
            result = F.interpolate(
                result, size=(H, W), mode='bilinear', align_corners=False
            )

        return result

    def _compute_frequency_loss(self, pred_rgb, gt_rgb):
        """Compute wavelet-domain consistency loss between predicted and GT images.

        Applies 1-level Haar DWT to both images and computes weighted MSE
        per subband.

        Args:
            pred_rgb: Predicted clean image [B, 3, H, W] in [0, 1] (from VAE decode)
            gt_rgb: Ground-truth clean image [B, 3, H, W] in [0, 1] (from VAE decode)

        Returns:
            Scalar frequency loss value
        """
        device = pred_rgb.device
        B, C, H, W = pred_rgb.shape

        # Move filters to same device
        freq_filters = self.freq_loss_haar_filters.to(device=device, dtype=pred_rgb.dtype)

        # Apply DWT to both images
        pred_coeffs = F.conv2d(pred_rgb, freq_filters, stride=2, groups=3)  # [B, 12, H/2, W/2]
        gt_coeffs = F.conv2d(gt_rgb, freq_filters, stride=2, groups=3)  # [B, 12, H/2, W/2]

        # Rearrange to group by subband (same as _apply_wavelet_decomposition)
        pred_coeffs = pred_coeffs.view(B, 3, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        gt_coeffs = gt_coeffs.view(B, 3, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        # [B, 4, 3, H/2, W/2] — dim1: LL=0, LH=1, HL=2, HH=3

        # Per-subband MSE with configurable weights
        subband_names = ['ll', 'lh', 'hl', 'hh']
        total_loss = torch.tensor(0.0, device=device, dtype=pred_rgb.dtype)
        total_weight = 0.0

        for i, name in enumerate(subband_names):
            w = self.freq_subband_weights[name]
            if w > 0:
                subband_mse = F.mse_loss(pred_coeffs[:, i], gt_coeffs[:, i])
                total_loss = total_loss + w * subband_mse
                total_weight += w

        # Normalize by total weight to keep loss scale independent of weight sum
        if total_weight > 0:
            total_loss = total_loss / total_weight

        return total_loss

    def _replace_controlnet_cond_conv_in(self):
        """Replace ControlNet conditioning embedding's first conv layer.

        The default ControlNet has controlnet_cond_embedding.conv_in as
        Conv2d(3, 16, kernel_size=3, padding=1).
        We replace it with Conv2d(N, 16, kernel_size=3, padding=1) where N
        depends on wavelet subband selection (12, 3, or 9).

        Verified: ControlNetConditioningEmbedding.conv_in structure at
        diffusers/models/controlnets/controlnet.py line 82:
            self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        where block_out_channels[0] = 16 (default).
        """
        old_conv = self.model.controlnet.controlnet_cond_embedding.conv_in
        out_channels = old_conv.out_channels  # 16
        kernel_size = old_conv.kernel_size  # (3, 3)
        padding = old_conv.padding  # (1, 1)

        new_conv = Conv2d(
            self.controlnet_cond_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Initialize with Kaiming normal (good default for ReLU/SiLU activations)
        torch.nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(new_conv.bias)

        self.model.controlnet.controlnet_cond_embedding.conv_in = new_conv
        logging.info(
            f"ControlNet cond_embedding.conv_in replaced: "
            f"Conv2d(3→{out_channels}) → Conv2d({self.controlnet_cond_channels}→{out_channels})"
        )

    # ------------------------------------------------------------------
    # Helper methods (copied from hybrid-003 trainer)
    # ------------------------------------------------------------------

    def _replace_unet_conv_in(self):
        """Replace the first layer to accept 8 in_channels.
        Exact copy from hybrid-003 trainer._replace_unet_conv_in (line 468-493)."""
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]

        _weight = _weight.repeat((1, 2, 1, 1))  # [320, 8, 3, 3]
        _weight *= 0.5

        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        self.model.unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return

    def encode_rgb(self, image_in):
        """Encode RGB image to latent space.
        Copied from hybrid-003 trainer.encode_rgb (line 495-503)."""
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        image_in = image_in.to(self.model.vae.dtype)
        latent = self.model.encode_rgb(image_in)
        return latent

    def decode_rgb(self, latent_in):
        """Decode latent to RGB image [0, 1].
        Copied from hybrid-003 trainer.decode_rgb (line 506-516)."""
        assert len(latent_in.shape) == 4 and latent_in.shape[1] == 4
        latent_in = latent_in.to(self.model.vae.dtype)
        rgb = self.model.decode_rgb(latent_in)
        rgb = (rgb + 1.0) / 2.0
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb

    def _get_next_seed(self):
        """Get next seed from global sequence.
        Copied from hybrid-003 trainer._get_next_seed (line 519-530)."""
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * 2,
            )
            logging.info(
                f"Global seed sequence generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def _get_backup_ckpt_name(self):
        """Get backup checkpoint name.
        Copied from hybrid-003 trainer._get_backup_ckpt_name (line 533-540)."""
        return f"iter_{self.effective_iter:06d}"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, t_end=None):
        """Main training loop for hybrid-wavelet: joint UNet + Wavelet ControlNet.

        Based on hybrid-003 trainer.train() (lines 543-1045) with key differences:
        - After input noise augmentation, applies wavelet decomposition to controlnet_cond
        - Adds frequency-domain loss computation after pixel-space losses
        - Logs freq_loss to train metrics and tensorboard
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
            # Copied from hybrid-003 trainer (lines 585-590)
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(self.epoch)
            elif hasattr(self.train_loader.dataset, 'datasets'):
                for dataset in self.train_loader.dataset.datasets:
                    if hasattr(dataset, 'set_epoch'):
                        dataset.set_epoch(self.epoch)

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()
                self.model.controlnet.train()

                # Globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # Get data — degraded and clean RGB in [-1, 1]
                degraded_rgb = batch[self.degraded_rgb_type].to(device)  # [B, 3, H, W]
                clean_rgb = batch[self.clean_rgb_type].to(device)  # [B, 3, H, W]

                batch_size = degraded_rgb.shape[0]

                with torch.no_grad():
                    clean_latent = self.encode_rgb(clean_rgb)  # [B, 4, h, w]
                    degraded_latent = self.encode_rgb(degraded_rgb)  # [B, 4, h, w]

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
                # Pattern from hybrid-003 trainer (lines 647-654)
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

                # Conditioning: ARNIQA or empty text embedding
                # Pattern from hybrid-003 trainer (lines 662-672)
                if self.use_arniqa:
                    text_embed = self.arniqa_conditioner(degraded_rgb, apply_dropout=True)

                    # Clamp ARNIQA output norm to prevent destabilizing cross-attention
                    # (added after train-015 gradient explosion at iter ~350)
                    with torch.no_grad():
                        raw_norm = text_embed.norm().item()
                    if self.arniqa_output_max_norm is not None:
                        output_norm = text_embed.norm()
                        if output_norm > self.arniqa_output_max_norm:
                            text_embed = text_embed * (self.arniqa_output_max_norm / output_norm)

                    with torch.no_grad():
                        self._arniqa_output_stats = {
                            'output_norm_raw': raw_norm,
                            'output_norm': text_embed.norm().item(),
                            'output_mean': text_embed.mean().item(),
                            'output_std': text_embed.std().item(),
                            'output_clamped': 1.0 if (self.arniqa_output_max_norm is not None and raw_norm > self.arniqa_output_max_norm) else 0.0,
                        }
                else:
                    text_embed = self.empty_text_embed.to(device).repeat(
                        (batch_size, 1, 1)
                    )

                # Input noise augmentation on degraded RGB (pixel space)
                # Pattern from hybrid-003 trainer (lines 675-696)
                controlnet_cond = degraded_rgb  # [B, 3, H, W] in [-1, 1]
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

                # ============================================================
                # NEW: Wavelet decomposition of controlnet_cond
                # Applied AFTER input noise augmentation so ControlNet sees
                # noisy wavelet coefficients (matching the augmentation intent)
                # ============================================================
                if self.use_wavelet:
                    controlnet_cond = self._apply_wavelet_decomposition(controlnet_cond)
                    # Shape: [B, N, H, W] or [B, N, H/2, W/2] depending on upsample config
                    # where N = 12 (all), 3 (ll_only), or 9 (hf_only)

                # CFG dropout
                # Pattern from hybrid-003 trainer (lines 699-706)
                cfg_dropout_mask = None
                if self.cfg_dropout_prob > 0.0:
                    cfg_dropout_mask = (
                        torch.rand(batch_size, device=device, generator=rand_num_generator)
                        < self.cfg_dropout_prob
                    )

                # Forward pass with FP16 mixed precision
                with autocast('cuda'):
                    # ControlNet forward
                    # Verified: ControlNetModel.__call__ accepts controlnet_cond
                    # (diffusers/models/controlnets/controlnet.py line 715)
                    down_block_res, mid_block_res = self.model.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embed,
                        controlnet_cond=controlnet_cond,
                        return_dict=False,
                    )

                    # Apply CFG dropout
                    # Pattern from hybrid-003 trainer (lines 716-737)
                    if cfg_dropout_mask is not None and cfg_dropout_mask.any():
                        drop_mask_4d = cfg_dropout_mask.float().view(batch_size, 1, 1, 1)
                        keep_mask_4d = 1.0 - drop_mask_4d
                        down_block_res = [r * keep_mask_4d for r in down_block_res]
                        mid_block_res = mid_block_res * keep_mask_4d

                        degraded_latent_for_unet = degraded_latent * keep_mask_4d
                        empty_embed = self.empty_text_embed.to(device).repeat(
                            (batch_size, 1, 1)
                        )
                        drop_mask_3d = cfg_dropout_mask.float().view(batch_size, 1, 1)
                        keep_mask_3d = 1.0 - drop_mask_3d
                        text_embed_for_unet = text_embed * keep_mask_3d + empty_embed * drop_mask_3d
                    else:
                        degraded_latent_for_unet = degraded_latent
                        text_embed_for_unet = text_embed

                    # Concatenate degraded_latent + noisy_latents → [B, 8, h, w]
                    cat_latents = torch.cat(
                        [degraded_latent_for_unet, noisy_latents], dim=1
                    )

                    # UNet forward with ControlNet residuals
                    model_pred = self.model.unet(
                        cat_latents,
                        timesteps,
                        encoder_hidden_states=text_embed_for_unet,
                        down_block_additional_residuals=down_block_res,
                        mid_block_additional_residual=mid_block_res,
                    ).sample

                    # Get target
                    # Pattern from hybrid-003 trainer (lines 754-763)
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
                        f"NaN detected in model_pred or loss at iter {self.effective_iter + 1}, "
                        f"skipping backward pass for this batch."
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

                # We also need pred_rgb_crop and clean_rgb_crop for frequency loss
                _need_pixel_decode = _use_lpips or _use_pixel_recon or self.use_freq_loss

                pred_rgb_crop = None
                clean_rgb_crop = None

                if _need_pixel_decode:
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

                        # LPIPS loss
                        if _use_lpips:
                            pred_lpips_in = pred_rgb_crop * 2.0 - 1.0
                            clean_lpips_in = clean_rgb_crop * 2.0 - 1.0
                            pixel_lpips = self.lpips_loss(pred_lpips_in, clean_lpips_in).mean()
                            if self.pixel_loss_max_value is not None:
                                pixel_lpips = pixel_lpips.clamp(max=self.pixel_loss_max_value)
                            if torch.isfinite(pixel_lpips):
                                loss = loss + self.lpips_weight * pixel_lpips
                                self.train_metrics.update("pixel_lpips", pixel_lpips.item())
                            else:
                                logging.warning(
                                    f"Non-finite LPIPS at iter {self.effective_iter + 1}, "
                                    f"skipping pixel loss for this batch."
                                )

                        # Pixel reconstruction loss
                        if _use_pixel_recon:
                            pixel_recon = self.pixel_reconstruction_loss(pred_rgb_crop, clean_rgb_crop)
                            if self.pixel_loss_max_value is not None:
                                pixel_recon = pixel_recon.clamp(max=self.pixel_loss_max_value)
                            if torch.isfinite(pixel_recon):
                                loss = loss + self.pixel_loss_weight * pixel_recon
                                self.train_metrics.update("pixel_reconstruction", pixel_recon.item())
                            else:
                                logging.warning(
                                    f"Non-finite pixel reconstruction loss at iter {self.effective_iter + 1}, "
                                    f"skipping for this batch."
                                )

                        # ============================================================
                        # NEW: Frequency-domain loss on VAE-decoded crops
                        # ============================================================
                        if self.use_freq_loss and pred_rgb_crop is not None and clean_rgb_crop is not None:
                            freq_loss = self._compute_frequency_loss(pred_rgb_crop, clean_rgb_crop)
                            if self.pixel_loss_max_value is not None:
                                freq_loss = freq_loss.clamp(max=self.pixel_loss_max_value)
                            if torch.isfinite(freq_loss):
                                loss = loss + self.freq_loss_weight * freq_loss
                                self.train_metrics.update("freq_loss", freq_loss.item())
                            else:
                                logging.warning(
                                    f"Non-finite frequency loss at iter {self.effective_iter + 1}, "
                                    f"skipping for this batch."
                                )

                loss = loss / self.gradient_accumulation_steps

                # Loss clipping
                # Pattern from hybrid-003 trainer (lines 862-872)
                max_loss = 1.0
                if loss.item() > max_loss:
                    logging.warning(
                        f"Loss clipped from {loss.item():.4f} to {max_loss} "
                        f"at iter {self.effective_iter + 1}"
                    )
                    loss = loss.clamp(max=max_loss)

                # Backward pass
                self.scaler.scale(loss).backward()

                # ARNIQA gradient/weight norms
                # Pattern from hybrid-003 trainer (lines 878-893)
                if self.use_arniqa and self.arniqa_conditioner is not None:
                    scale_factor = self.scaler.get_scale() if self.scaler is not None else 1.0
                    scale_factor = max(scale_factor, 1e-8)

                    grad_norms = self.arniqa_conditioner.get_adapter_grad_norms()
                    grad_norms = {k: v / scale_factor for k, v in grad_norms.items()}
                    weight_norms = self.arniqa_conditioner.get_adapter_weight_norms()

                    self._arniqa_grad_norms = grad_norms
                    self._arniqa_weight_norms = weight_norms
                    self._arniqa_grad_norm = sum(v**2 for v in grad_norms.values()) ** 0.5
                    self._arniqa_weight_norm = sum(v**2 for v in weight_norms.values()) ** 0.5

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
                    if self.use_arniqa and self.arniqa_conditioner is not None:
                        # Separate gradient clipping for global vs spatial adapter
                        # (added after train-015 gradient explosion: spatial adapter
                        #  grad norm spiked from ~1 to ~50, causing output norm 300→3000)
                        torch.nn.utils.clip_grad_norm_(
                            self.arniqa_conditioner.global_adapter.parameters(),
                            max_norm=1.0
                        )
                        if self.arniqa_stage >= 2 and self.arniqa_conditioner.spatial_adapter is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.arniqa_conditioner.spatial_adapter.parameters(),
                                max_norm=self.arniqa_spatial_max_grad_norm  # 0.1
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
                    # ARNIQA logging
                    # Pattern from hybrid-003 trainer (lines 930-970)
                    if self.use_arniqa:
                        if hasattr(self, '_arniqa_grad_norm'):
                            tb_logger.writer.add_scalar(
                                "arniqa/grad_norm", self._arniqa_grad_norm,
                                global_step=self.effective_iter,
                            )
                        if hasattr(self, '_arniqa_weight_norm'):
                            tb_logger.writer.add_scalar(
                                "arniqa/weight_norm", self._arniqa_weight_norm,
                                global_step=self.effective_iter,
                            )
                        if hasattr(self, '_arniqa_grad_norms'):
                            for adapter_name, grad_norm in self._arniqa_grad_norms.items():
                                tb_logger.writer.add_scalar(
                                    f"arniqa/{adapter_name}_grad_norm", grad_norm,
                                    global_step=self.effective_iter,
                                )
                        if hasattr(self, '_arniqa_weight_norms'):
                            for adapter_name, weight_norm in self._arniqa_weight_norms.items():
                                tb_logger.writer.add_scalar(
                                    f"arniqa/{adapter_name}_weight_norm", weight_norm,
                                    global_step=self.effective_iter,
                                )
                        if hasattr(self, '_arniqa_output_stats'):
                            tb_logger.writer.add_scalar(
                                "arniqa/output_norm_raw",
                                self._arniqa_output_stats['output_norm_raw'],
                                global_step=self.effective_iter,
                            )
                            tb_logger.writer.add_scalar(
                                "arniqa/output_norm",
                                self._arniqa_output_stats['output_norm'],
                                global_step=self.effective_iter,
                            )
                            tb_logger.writer.add_scalar(
                                "arniqa/output_mean",
                                self._arniqa_output_stats['output_mean'],
                                global_step=self.effective_iter,
                            )
                            tb_logger.writer.add_scalar(
                                "arniqa/output_std",
                                self._arniqa_output_stats['output_std'],
                                global_step=self.effective_iter,
                            )
                            tb_logger.writer.add_scalar(
                                "arniqa/output_clamped",
                                self._arniqa_output_stats['output_clamped'],
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
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

            # Epoch end
            self.n_batch_in_epoch = 0

    # ------------------------------------------------------------------
    # Callbacks (copied from hybrid-003 trainer lines 1046-1108)
    # ------------------------------------------------------------------

    def _train_step_callback(self):
        """Executed after every iteration."""
        if self.checkpoint_mode == 'marigold':
            self._train_step_callback_marigold()
        else:
            self._train_step_callback_decoupled()

    def _train_step_callback_marigold(self):
        """Original Marigold checkpoint pattern."""
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
        """Decoupled checkpoint pattern."""
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
        """Save checkpoint: UNet + ControlNet + ARNIQA adapter + wavelet config.

        Based on hybrid-003 trainer.save_checkpoint (lines 1110-1226).
        Additional: saves wavelet_config.json with wavelet conditioning settings.
        """
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

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=True)
        logging.info(f"UNet is saved to: {unet_path}")

        # Save ControlNet
        controlnet_path = os.path.join(ckpt_dir, "controlnet")
        self.model.controlnet.save_pretrained(controlnet_path, safe_serialization=True)
        logging.info(f"ControlNet is saved to: {controlnet_path}")

        # Save scheduler
        if not (self.checkpoint_test_config and self.checkpoint_test_config.get('save_unet_only', False)):
            scheduler_path = os.path.join(ckpt_dir, "scheduler")
            self.model.scheduler.save_pretrained(scheduler_path)
            logging.info(f"Scheduler is saved to: {scheduler_path}")

        # Save ARNIQA adapter weights (if enabled)
        if self.use_arniqa and self.arniqa_conditioner is not None:
            stage_str = f"stage{self.arniqa_stage}"
            arniqa_checkpoint = {
                "stage": stage_str,
                "version": "2.0",
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

        # Save wavelet + hybrid config for reproducibility
        wavelet_hybrid_config = {
            "architecture": "hybrid-wavelet",
            "unet_trainable": True,
            "controlnet_trainable": True,
            "arniqa_enabled": self.use_arniqa,
            "arniqa_stage": self.arniqa_stage if self.use_arniqa else None,
            "wavelet_conditioning": {
                "enabled": self.use_wavelet,
                "wavelet_type": self.wavelet_type if self.use_wavelet else None,
                "decomposition_levels": self.wavelet_levels if self.use_wavelet else None,
                "subbands": self.wavelet_subbands if self.use_wavelet else None,
                "controlnet_cond_channels": self.controlnet_cond_channels,
                "upsample_to_input_res": self.wavelet_upsample if self.use_wavelet else None,
            },
            "frequency_loss": {
                "enabled": self.use_freq_loss,
                "weight": self.freq_loss_weight if self.use_freq_loss else None,
            },
        }
        config_path = os.path.join(ckpt_dir, "wavelet_config.json")
        with open(config_path, "w") as f:
            json.dump(wavelet_hybrid_config, f, indent=2)
        logging.info(f"Wavelet config saved to: {config_path}")

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

        # Iteration indicator
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
        """Load checkpoint: UNet + ControlNet + ARNIQA adapter + trainer state.
        Based on hybrid-003 trainer.load_checkpoint (lines 1228-1301)."""
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

        # Load ARNIQA adapter
        _arniqa_path = os.path.join(ckpt_path, "arniqa_adapter.pt")
        if self.use_arniqa and self.arniqa_conditioner is not None and os.path.isfile(_arniqa_path):
            arniqa_checkpoint = torch.load(_arniqa_path, map_location=self.device)
            self.arniqa_conditioner.load_state_dict(arniqa_checkpoint["state_dict"])
            logging.info(
                f"ARNIQA adapter loaded from {_arniqa_path} "
                f"(stage={arniqa_checkpoint.get('stage', 'unknown')})"
            )
        elif self.use_arniqa:
            logging.warning(f"ARNIQA adapter not found at {_arniqa_path}, using fresh initialization")

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
        Copied from hybrid-003 trainer (lines 1303-1326)."""
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
        """Log checkpoint size and disk usage.
        Copied from hybrid-003 trainer (lines 1328-1349)."""
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
        Copied from hybrid-003 trainer (lines 1351-1402)."""
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
        Copied from hybrid-003 trainer.validate (lines 1404-1451)."""
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
        Copied from hybrid-003 trainer.validate_decoupled (lines 1453-1497)."""
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

        Based on hybrid-003 trainer.validate_single_dataset (lines 1499-1703).
        Uses model.single_infer() which handles ControlNet conditioning internally.
        NOTE: The pipeline's single_infer() must also apply wavelet decomposition
        to controlnet_cond — this is handled in the wavelet pipeline.
        """
        self.model.to(self.device)
        metric_tracker.reset()

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

            # Calculate validation loss
            for b_idx in range(batch_size):
                batch_single = {
                    k: v[b_idx:b_idx+1] if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v
                    for k, v in batch.items()
                }
                seed = batch_seeds[b_idx]
                generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
                val_loss_dict = self._calculate_validation_loss(batch_single, generator)
                val_loss_dicts.append(val_loss_dict)

            # Predict restored images
            degraded_rgb_norm = degraded_rgb_int.float() / 255.0 * 2.0 - 1.0
            degraded_rgb_norm = degraded_rgb_norm.to(self.device)

            self.model.scheduler.set_timesteps(
                self.cfg.validation.denoising_steps, device=self.device
            )

            # single_infer handles wavelet decomposition internally (in the pipeline)
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
                        f"skipping metrics for this sample."
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

        # Average validation loss
        if val_loss_dicts:
            loss_keys = val_loss_dicts[0].keys()
            avg_losses = {}
            for key in loss_keys:
                values = [d[key] for d in val_loss_dicts]
                avg_losses[key] = sum(values) / len(values)
            metric_tracker.update('val_loss', avg_losses['total'])

        results = metric_tracker.result()

        # Log images to W&B
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

    def visualize(self):
        """Visualization: run inference and save images.
        Copied from hybrid-003 trainer.visualize (lines 1705-1733)."""
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
        """Calculate validation loss using the same process as training loss.

        Based on hybrid-003 trainer._calculate_validation_loss (lines 1735-1849).
        KEY DIFFERENCE: applies wavelet decomposition to controlnet_cond when enabled.
        """
        device = self.device

        degraded_rgb = batch[self.degraded_rgb_type].to(device)
        clean_rgb = batch[self.clean_rgb_type].to(device)
        batch_size = degraded_rgb.shape[0]

        with torch.no_grad():
            clean_latent = self.encode_rgb(clean_rgb)
            degraded_latent = self.encode_rgb(degraded_rgb)

        timesteps = torch.randint(
            0,
            self.scheduler_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        ).long()

        noise = torch.randn(
            clean_latent.shape,
            device=device,
            generator=generator,
        )

        noisy_latents = self.training_noise_scheduler.add_noise(
            clean_latent, noise, timesteps
        )

        # Conditioning
        if self.use_arniqa:
            text_embed = self.arniqa_conditioner(degraded_rgb, apply_dropout=False)
            # Clamp ARNIQA output norm (same protection as train loop)
            if self.arniqa_output_max_norm is not None:
                output_norm = text_embed.norm()
                if output_norm > self.arniqa_output_max_norm:
                    text_embed = text_embed * (self.arniqa_output_max_norm / output_norm)
        else:
            text_embed = self.empty_text_embed.to(device).repeat(
                (batch_size, 1, 1)
            )

        # ControlNet conditioning (no input noise during validation)
        controlnet_cond = degraded_rgb  # [B, 3, H, W]

        # NEW: Apply wavelet decomposition for validation loss
        if self.use_wavelet:
            controlnet_cond = self._apply_wavelet_decomposition(controlnet_cond)

        with autocast('cuda'):
            down_block_res, mid_block_res = self.model.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embed,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )

            cat_latents = torch.cat(
                [degraded_latent, noisy_latents], dim=1
            )

            model_pred = self.model.unet(
                cat_latents,
                timesteps,
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

        loss_dict = {'total': loss.item()}
        return loss_dict

    def _create_comparison_image(self, clean_np, degraded_np, restored_np):
        """Create a side-by-side comparison image: Clean | Degraded | Restored.
        Copied from hybrid-003 trainer._create_comparison_image (lines 1838-1870)."""
        h, w = clean_np.shape[:2]
        gap = 4
        canvas = np.ones((h, w * 3 + gap * 2, 3), dtype=np.float32)

        canvas[:, :w, :] = clean_np
        canvas[:, w + gap:2*w + gap, :] = degraded_np
        canvas[:, 2*w + 2*gap:, :] = restored_np

        canvas = np.clip(canvas, 0, 1)
        canvas_uint8 = (canvas * 255).astype(np.uint8)
        return Image.fromarray(canvas_uint8)
