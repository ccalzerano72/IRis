# MarigoldHybridControlNetArniqa003Trainer - Thesis Implementation
#
# Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction:
# Adapting Marigold Architecture for Distortion-Guided Image Enhancement
#
# Hybrid-003: Joint UNet + ControlNet + ARNIQA Conditioning
# - 8ch UNet (SD2 weights expanded 4→8ch, TRAINABLE with low LR)
# - ControlNet (from SD2 UNet, TRAINABLE with higher LR)
# - ARNIQA Stage 2 quality-aware conditioning (frozen encoder + trainable adapters)
# - Degraded latent concatenation (channels 0:4) + noisy latent (channels 4:8)
# - ControlNet receives degraded RGB in pixel space
#
# Key differences from hybrid-002:
# - UNet is TRAINABLE (not frozen) — starts from SD2 original weights
# - No base_checkpoint_path — no pre-trained restoration checkpoint needed
# - ARNIQA conditioning replaces empty text embedding (configurable)
# - 3-way differential LR: UNet + ControlNet + ARNIQA adapters
# - CFG dropout: randomly zero ControlNet residuals + use empty text embed
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
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from typing import List, Union

from marigold.marigold_hybrid_controlnet_arniqa_003_pipeline import (
    MarigoldHybridControlNetArniqa003Pipeline,
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


class MarigoldHybridControlNetArniqa003Trainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldHybridControlNetArniqa003Pipeline,
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
        self.model: MarigoldHybridControlNetArniqa003Pipeline = model
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

        # ---- ABLATION FLAG: freeze ControlNet (guarded, additive) ----
        # When True: ControlNet zero-convs forced to 0 + requires_grad_(False) +
        # excluded from the optimizer + gradient-checkpointing skipped for ControlNet.
        # Residual contribution == 0 for the whole run (model degenerates to
        # concat-only) while loss (incl. pixel-L1) and batch stay identical.
        # Flag absent/False => behaviour byte-for-byte identical to the normal path.
        self.freeze_controlnet = bool(self.cfg.get('freeze_controlnet', False))

        # ---- Step 1: Replace UNet conv_in to accept 8 channels ----
        # Pattern from hybrid-002 trainer __init__ (line 82-83)
        # No base checkpoint loading — UNet starts from SD2 original weights
        if 8 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        # ---- Step 2: Encode empty text prompt ----
        # Verified: MarigoldHybridControlNetArniqa003Pipeline.encode_empty_text()
        # sets self.empty_text_embed (pipeline line 115-128)
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        # ---- Step 3: XFormers memory-efficient attention for both UNet and ControlNet ----
        # Pattern from hybrid-002 trainer __init__ (line 93-94)
        self.model.unet.enable_xformers_memory_efficient_attention()
        self.model.controlnet.enable_xformers_memory_efficient_attention()

        # ---- Step 4: Gradient checkpointing ----
        # Pattern from hybrid-002 trainer __init__ (line 97-102)
        self.gradient_checkpointing = self.cfg.trainer.get('gradient_checkpointing', True)
        if self.gradient_checkpointing:
            self.model.unet.enable_gradient_checkpointing()
            if not self.freeze_controlnet:
                self.model.controlnet.enable_gradient_checkpointing()
                logging.info("Gradient checkpointing ENABLED for both UNet and ControlNet")
            else:
                logging.info(
                    "Gradient checkpointing ENABLED for UNet "
                    "(ControlNet frozen -> checkpointing skipped)"
                )
        else:
            logging.info("Gradient checkpointing disabled")

        # ---- Step 5: Trainability — UNet + ControlNet trainable, VAE + text_encoder frozen ----
        # KEY DIFFERENCE from hybrid-002: UNet is TRAINABLE (not frozen)
        # Pattern from restoration trainer __init__ (line 91-93) for trainable UNet
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        # UNet is TRAINABLE — requires_grad=True, train() mode
        self.model.unet.requires_grad_(True)
        # ControlNet is TRAINABLE
        self.model.controlnet.requires_grad_(True)

        unet_params = sum(p.numel() for p in self.model.unet.parameters()) / 1e6
        controlnet_params = sum(p.numel() for p in self.model.controlnet.parameters()) / 1e6
        logging.info(
            f"Trainability: UNet=TRAINABLE ({unet_params:.1f}M), "
            f"ControlNet=TRAINABLE ({controlnet_params:.1f}M), "
            f"VAE=frozen, text_encoder=frozen"
        )

        # ---- Step 5b: Reinitialize ControlNet zero convolutions ----
        # Pattern from hybrid-002 trainer __init__ (line 119-140)
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

        # ---- Step 5c (ABLATION): freeze ControlNet (guarded by freeze_controlnet) ----
        # Option A: force zero-conv zero-init + freeze ALL ControlNet params so the
        # injected residual is provably 0 for the entire run. Placed AFTER Step 5b so
        # it overrides any zero-conv reinit. No-op when the flag is False/absent.
        if self.freeze_controlnet:
            with torch.no_grad():
                for _blk in self.model.controlnet.controlnet_down_blocks:
                    _blk.weight.zero_()
                    if _blk.bias is not None:
                        _blk.bias.zero_()
                self.model.controlnet.controlnet_mid_block.weight.zero_()
                if self.model.controlnet.controlnet_mid_block.bias is not None:
                    self.model.controlnet.controlnet_mid_block.bias.zero_()
            self.model.controlnet.requires_grad_(False)
            _cn_total = sum(p.numel() for p in self.model.controlnet.parameters())
            _cn_train = sum(
                p.numel() for p in self.model.controlnet.parameters() if p.requires_grad
            )
            logging.info(
                f"freeze_controlnet=True (ABLATION): zero-convs forced to 0, "
                f"ControlNet FROZEN ({_cn_total / 1e6:.1f}M params, "
                f"requires_grad-True count={_cn_train}). "
                f"Residual contribution == 0 -> concat-only."
            )

        # ---- Step 6: ARNIQA quality-aware conditioning (optional) ----
        # Pattern from restoration trainer __init__ (lines 97-136)
        arniqa_cfg = self.cfg.get('arniqa', {})
        self.use_arniqa = arniqa_cfg.get('enabled', False)
        self.arniqa_conditioner = None
        self.arniqa_stage = 1  # Default to Stage 1

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

        # ---- Step 7: Optimizer — 3-way differential LR ----
        # Pattern from restoration trainer __init__ (lines 140-166)
        # UNet (low LR) + ControlNet (higher LR) + ARNIQA adapters (separate LR)
        lr = self.cfg.lr  # UNet LR
        controlnet_lr = self.cfg.get('controlnet_lr', lr)  # ControlNet LR

        # Only optimize params that require grad. When freeze_controlnet=True the
        # ControlNet group is empty and is excluded entirely. Normal path (all params
        # trainable) yields the same two groups as before -> unchanged behaviour.
        _unet_trainable = [p for p in self.model.unet.parameters() if p.requires_grad]
        _controlnet_trainable = [p for p in self.model.controlnet.parameters() if p.requires_grad]
        param_groups = [
            {'params': _unet_trainable, 'lr': lr, 'name': 'unet'},
        ]
        if len(_controlnet_trainable) > 0:
            param_groups.append(
                {'params': _controlnet_trainable, 'lr': controlnet_lr, 'name': 'controlnet'}
            )

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
        _n_opt_trainable = sum(p.numel() for g in param_groups for p in g['params'])
        logging.info(
            f"Total trainable params in optimizer: {_n_opt_trainable / 1e6:.1f}M "
            f"(groups={[g['name'] for g in param_groups]}, "
            f"freeze_controlnet={self.freeze_controlnet})"
        )

        # ---- Step 8: LR scheduler ----
        # Pattern from hybrid-002 trainer __init__ (lines 148-170)
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
        # Pattern from hybrid-002 trainer __init__ (line 173-174)
        self.scaler = GradScaler()
        logging.info("Mixed precision training (FP16) enabled via GradScaler")

        # ---- Step 10: Loss function ----
        # Pattern from hybrid-002 trainer __init__ (lines 177-181)
        trainer_only_params = {'lpips_weight', 'pixel_loss_weight', 'pixel_loss_type', 'pixel_loss_max_timestep', 'pixel_loss_max_value', 'pixel_loss_timestep_schedule', 'dino_weight', 'dino_model_size', 'dino_version', 'dino_target_size', 'dino_layers'}
        loss_kwargs = {k: v for k, v in self.cfg.loss.kwargs.items()
                       if v is not None and k not in trainer_only_params}
        logging.info(f"Loss kwargs: {loss_kwargs}")
        self.loss = get_loss(loss_name=self.cfg.loss.name, **loss_kwargs)

        # ---- Step 11: Pixel-space LPIPS loss (optional) ----
        # Pattern from hybrid-002 trainer __init__ (lines 184-199)
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

        # ---- Step 11a-bis: Pixel-space DINO perceptual loss (optional) ----
        # Modern alternative to LPIPS using DINOv2/v3 self-supervised features.
        # Shares the same VAE-decoded crop as LPIPS when both are enabled.
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

        # ---- Step 11b: Pixel-space reconstruction loss (optional, reuses LPIPS crop) ----
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
        # Skip pixel-space losses (LPIPS + L1/L2) when timestep > threshold.
        # At high timesteps, pred_original_sample recovery divides by near-zero
        # alpha_prod_t, producing extreme values that cause loss explosion.
        self.pixel_loss_max_timestep = self.cfg.loss.kwargs.get('pixel_loss_max_timestep', 1000)
        if self.pixel_loss_max_timestep < 1000:
            logging.info(
                f"Pixel-space loss timestep threshold: {self.pixel_loss_max_timestep} "
                f"(losses disabled above this timestep)"
            )
        else:
            logging.info("Pixel-space loss timestep threshold: disabled (no limit)")

        # ---- Step 11c-bis: Timestep-adaptive loss weighting schedule ----
        # Controls how pixel-space loss weights scale with timestep below the threshold.
        #   "step"   (default): binary on/off — full weight below threshold, zero above.
        #   "linear": weight scales linearly from 0 at t=threshold to full at t=0.
        #             w(t) = 1 - t / pixel_loss_max_timestep, clamped to [0, 1].
        # This allows a smooth transition: MSE dominates at high t (structural phase),
        # perceptual losses ramp up at low t (detail phase).
        self.pixel_loss_timestep_schedule = self.cfg.loss.kwargs.get(
            'pixel_loss_timestep_schedule', 'step'
        )
        if self.pixel_loss_timestep_schedule not in ('step', 'linear'):
            logging.warning(
                f"Unknown pixel_loss_timestep_schedule '{self.pixel_loss_timestep_schedule}', "
                f"falling back to 'step'"
            )
            self.pixel_loss_timestep_schedule = 'step'
        logging.info(f"Pixel-space loss timestep schedule: {self.pixel_loss_timestep_schedule}")

        # ---- Step 11d: Pixel-space loss value clamping ----
        # Cap individual pixel losses (LPIPS, L1/L2) before adding to total loss.
        # Prevents gradient spikes from outlier batches regardless of timestep.
        self.pixel_loss_max_value = self.cfg.loss.kwargs.get('pixel_loss_max_value', None)
        if self.pixel_loss_max_value is not None:
            logging.info(
                f"Pixel-space loss value clamping ENABLED: max={self.pixel_loss_max_value}"
            )
        else:
            logging.info("Pixel-space loss value clamping: disabled (no limit)")

        # ---- Step 12: Training noise scheduler ----
        # Pattern from hybrid-002 trainer __init__ (lines 202-224)
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
        # Pattern from hybrid-002 trainer __init__ (line 228)
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        # ---- Step 14: Train and validation metrics ----
        # Pattern from hybrid-002 trainer __init__ (lines 232-243)
        train_metric_keys = ["loss"]
        if self.lpips_weight > 0:
            train_metric_keys.append("pixel_lpips")
        if self.pixel_loss_weight > 0:
            train_metric_keys.append("pixel_reconstruction")
        if self.dino_weight > 0:
            train_metric_keys.append("pixel_dino")
        self.train_metrics = MetricTracker(*train_metric_keys)

        val_metric_keys = [m.__name__ for m in self.metric_funcs] + ["val_loss"]
        if self.lpips_weight > 0:
            val_metric_keys.append("val_pixel_lpips")
        if self.dino_weight > 0:
            val_metric_keys.append("val_pixel_dino")
        self.val_metrics = MetricTracker(*val_metric_keys)

        # Main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal

        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."

        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # ---- Step 15: Settings ----
        # Pattern from hybrid-002 trainer __init__ (lines 253-262)
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
        # Pattern from hybrid-002 trainer __init__ (lines 265-270)
        offset_noise_cfg = self.cfg.get('offset_noise', {})
        self.offset_noise_strength = offset_noise_cfg.get('strength', 0.0)
        if self.offset_noise_strength > 0.0:
            logging.info(f"Offset noise ENABLED - strength: {self.offset_noise_strength}")
        else:
            logging.info("Offset noise disabled (strength = 0.0)")

        # ---- Step 17: Input noise augmentation ----
        # Pattern from hybrid-002 trainer __init__ (lines 274-285)
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
        # NEW in hybrid-003: randomly drop conditioning during training
        # When triggered: zero ControlNet residuals + use empty text embed
        # This enables CFG at inference time (guidance_scale > 1.0)
        cfg_dropout_cfg = self.cfg.get('cfg_dropout', {})
        self.cfg_dropout_prob = cfg_dropout_cfg.get('probability', 0.0)
        if self.cfg_dropout_prob > 0.0:
            logging.info(f"CFG dropout ENABLED - probability: {self.cfg_dropout_prob:.1%}")
        else:
            logging.info("CFG dropout disabled (probability = 0.0)")

        # ---- Step 19: Internal variables ----
        # Pattern from hybrid-002 trainer __init__ (lines 296-300)
        self.epoch = 1
        self.n_batch_in_epoch = 0
        self.effective_iter = 0
        self.in_evaluation = False
        self.global_seed_sequence: List = []

        # ---- Step 20: Checkpoint strategy ----
        # Pattern from hybrid-002 trainer __init__ (lines 303-316)
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
    # Helper methods
    # ------------------------------------------------------------------

    def _replace_unet_conv_in(self):
        """Replace the first layer to accept 8 in_channels.

        Exact copy from hybrid-002 trainer._replace_unet_conv_in (line 404-432).
        Channel layout: [0:4] = degraded (condition), [4:8] = noisy (target to denoise).
        """
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]

        _weight = _weight.repeat((1, 2, 1, 1))  # [320, 8, 3, 3]
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

    def encode_rgb(self, image_in):
        """Encode RGB image to latent space.
        Copied from hybrid-002 trainer.encode_rgb (line 435-441).
        Cast to VAE dtype because VAE is loaded in float16 (frozen) but dataset
        tensors arrive as float32.
        """
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        image_in = image_in.to(self.model.vae.dtype)
        latent = self.model.encode_rgb(image_in)
        return latent

    def decode_rgb(self, latent_in):
        """Decode latent to RGB image [0, 1].
        Copied from hybrid-002 trainer.decode_rgb (line 446-455).
        Cast to VAE dtype because VAE is loaded in float16 (frozen).
        """
        assert len(latent_in.shape) == 4 and latent_in.shape[1] == 4
        latent_in = latent_in.to(self.model.vae.dtype)
        rgb = self.model.decode_rgb(latent_in)
        # Convert from [-1, 1] to [0, 1] range
        rgb = (rgb + 1.0) / 2.0
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return rgb

    def _get_next_seed(self):
        """Get next seed from global sequence.
        Copied from hybrid-002 trainer._get_next_seed (line 459-469).
        """
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
        """Get backup checkpoint name based on current iteration.
        Copied from hybrid-002 trainer._get_backup_ckpt_name (line 473-475).
        """
        return f"iter_{self.effective_iter:06d}"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, t_end=None):
        """Main training loop for hybrid-003: joint UNet + ControlNet + ARNIQA.

        Based on hybrid-002 trainer.train() (lines 483-810) with key differences:
        - UNet is in train() mode (not eval)
        - ARNIQA conditioning replaces empty text embedding (when enabled)
        - CFG dropout: zero degraded_latent + ControlNet residuals + empty text embed
        - ARNIQA gradient/weight norm logging after backward
        - Gradient clipping on UNet + ControlNet + ARNIQA params
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
            # Copied from hybrid-002 trainer (lines 527-532)
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(self.epoch)
            elif hasattr(self.train_loader.dataset, 'datasets'):
                for dataset in self.train_loader.dataset.datasets:
                    if hasattr(dataset, 'set_epoch'):
                        dataset.set_epoch(self.epoch)

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                # KEY DIFFERENCE: both UNet and ControlNet in train mode
                self.model.unet.train()
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
                    # Encode degraded RGB to latent for 8ch UNet input
                    degraded_latent = self.encode_rgb(degraded_rgb)  # [B, 4, h, w]

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
                # Copied from hybrid-002 trainer (lines 590-597)
                if self.offset_noise_strength > 0.0:
                    offset = torch.randn(
                        batch_size, clean_latent.shape[1], 1, 1,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, 1, 1]
                    noise = noise + self.offset_noise_strength * offset

                # Add noise to the clean latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    clean_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Conditioning: ARNIQA quality features or empty text embedding
                # Pattern from restoration trainer (lines 587-600)
                if self.use_arniqa:
                    text_embed = self.arniqa_conditioner(degraded_rgb, apply_dropout=True)
                    # Capture ARNIQA output statistics for logging
                    with torch.no_grad():
                        self._arniqa_output_stats = {
                            'output_norm': text_embed.norm().item(),
                            'output_mean': text_embed.mean().item(),
                            'output_std': text_embed.std().item(),
                        }
                else:
                    text_embed = self.empty_text_embed.to(device).repeat(
                        (batch_size, 1, 1)
                    )  # [B, 77, 1024]

                # Input noise augmentation: add small noise to degraded RGB in PIXEL SPACE
                # Copied from hybrid-002 trainer (lines 610-631)
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

                # CFG dropout: per-sample mask for full conditioning dropout
                # Option B: zero degraded_latent + zero ControlNet residuals + empty text embed
                # This teaches the model to generate without any conditioning signal,
                # enabling CFG at inference (guidance_scale > 1.0)
                cfg_dropout_mask = None
                if self.cfg_dropout_prob > 0.0:
                    cfg_dropout_mask = (
                        torch.rand(batch_size, device=device, generator=rand_num_generator)
                        < self.cfg_dropout_prob
                    )  # [B] boolean mask, True = drop conditioning

                # Forward pass with FP16 mixed precision
                with autocast('cuda'):
                    # ControlNet forward: produces residuals from degraded RGB conditioning
                    # ControlNet receives noisy_latents (4ch) as sample, NOT the 8ch concat
                    # Verified: ControlNetModel.__call__ signature
                    down_block_res, mid_block_res = self.model.controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embed,
                        controlnet_cond=controlnet_cond,
                        return_dict=False,
                    )

                    # ABLATION DEBUG (env-guarded, temporary): magnitude of the
                    # ControlNet residual injected into the U-Net. Must be ~0 when
                    # freeze_controlnet=True. No effect unless ABLATION_DEBUG is set.
                    if os.environ.get('ABLATION_DEBUG') and self.effective_iter < 3:
                        with torch.no_grad():
                            _dn = torch.stack(
                                [r.detach().float().norm() for r in down_block_res]
                            ).norm().item()
                            _md = mid_block_res.detach().float().norm().item()
                            _mx = max(
                                [r.detach().abs().max().item() for r in down_block_res]
                                + [mid_block_res.detach().abs().max().item()]
                            )
                        logging.info(
                            f"[ABLATION_DEBUG] iter~{self.effective_iter} "
                            f"ControlNet residual: down_norm={_dn:.3e} "
                            f"mid_norm={_md:.3e} max_abs={_mx:.3e}"
                        )

                    # Apply CFG dropout: zero out conditioning for dropped samples
                    if cfg_dropout_mask is not None and cfg_dropout_mask.any():
                        # Zero ControlNet residuals for dropped samples
                        drop_mask_4d = cfg_dropout_mask.float().view(batch_size, 1, 1, 1)
                        keep_mask_4d = 1.0 - drop_mask_4d
                        down_block_res = [r * keep_mask_4d for r in down_block_res]
                        mid_block_res = mid_block_res * keep_mask_4d

                        # Zero degraded_latent for dropped samples (Option B)
                        degraded_latent_for_unet = degraded_latent * keep_mask_4d
                        # Replace text_embed with empty text for dropped samples
                        empty_embed = self.empty_text_embed.to(device).repeat(
                            (batch_size, 1, 1)
                        )
                        # Per-sample: keep original text_embed or use empty
                        drop_mask_3d = cfg_dropout_mask.float().view(batch_size, 1, 1)
                        keep_mask_3d = 1.0 - drop_mask_3d
                        text_embed_for_unet = text_embed * keep_mask_3d + empty_embed * drop_mask_3d
                    else:
                        degraded_latent_for_unet = degraded_latent
                        text_embed_for_unet = text_embed

                    # Concatenate degraded_latent + noisy_latents → [B, 8, h, w]
                    cat_latents = torch.cat(
                        [degraded_latent_for_unet, noisy_latents], dim=1
                    )  # [B, 8, h, w]

                    # UNet forward: inject ControlNet residuals
                    # Verified: UNet2DConditionModel.__call__ accepts
                    # down_block_additional_residuals and mid_block_additional_residual
                    model_pred = self.model.unet(
                        cat_latents,
                        timesteps,
                        encoder_hidden_states=text_embed_for_unet,
                        down_block_additional_residuals=down_block_res,
                        mid_block_additional_residual=mid_block_res,
                    ).sample  # [B, 4, h, w]

                    # Get the target for loss depending on the prediction type
                    # Copied from hybrid-002 trainer (lines 654-663)
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

                # NaN guard: skip backward pass to prevent gradient poisoning
                if torch.isnan(loss).any() or torch.isnan(model_pred).any():
                    logging.warning(
                        f"NaN detected in model_pred or loss at iter {self.effective_iter + 1}, "
                        f"skipping backward pass for this batch."
                    )
                    # Zero gradients to discard any partial accumulation
                    self.optimizer.zero_grad()
                    accumulated_step = 0
                    self.n_batch_in_epoch += 1
                    continue

                self.train_metrics.update("loss", loss.item())

                # Optional pixel-space losses on random crop (LPIPS and/or L1/L2)
                # Shared pred_original_sample → crop → VAE decode path
                _use_lpips = self.lpips_loss is not None and self.lpips_weight > 0
                _use_pixel_recon = self.pixel_reconstruction_loss is not None and self.pixel_loss_weight > 0
                _use_dino = self.dino_loss is not None and self.dino_weight > 0

                if _use_lpips or _use_pixel_recon or _use_dino:
                    # Per-sample timestep filter: only include samples where t <= threshold.
                    # At high timesteps, pred_original_sample recovery divides by near-zero
                    # alpha_prod_t, producing extreme values that cause loss explosion.
                    _t_mask = timesteps <= self.pixel_loss_max_timestep  # [B] boolean
                    _n_valid = _t_mask.sum().item()

                    if _n_valid > 0:
                        # Select only low-timestep samples for pixel-space losses
                        _valid_noisy = noisy_latents[_t_mask]       # [N, 4, h, w]
                        _valid_pred = model_pred[_t_mask]            # [N, 4, h, w]
                        _valid_clean_lat = clean_latent[_t_mask]     # [N, 4, h, w]
                        _valid_t = timesteps[_t_mask]                # [N]

                        # Timestep-adaptive weight for pixel-space losses.
                        # "step":   w=1 for all valid samples (binary on/off).
                        # "linear": w = 1 - t/threshold, so losses ramp up as t→0.
                        if self.pixel_loss_timestep_schedule == 'linear' and self.pixel_loss_max_timestep > 0:
                            _t_weight = (1.0 - _valid_t.float() / self.pixel_loss_max_timestep).clamp(0.0, 1.0)  # [N]
                        else:
                            _t_weight = torch.ones(_n_valid, device=device)  # [N]

                        # Recover predicted clean latent from model prediction
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

                        # Clamp predicted latent to prevent extreme values
                        # Normal SD2 latents live in roughly [-4, 4]; ±6 is a safe margin.
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
                        # No torch.no_grad(): gradients must flow through VAE decode
                        # back to pred_original_sample → model_pred → UNet + ControlNet weights.
                        # VAE params are frozen (requires_grad=False) so no VAE weight updates.
                        pred_rgb_crop = self.decode_rgb(pred_crop.float())
                        clean_rgb_crop = self.decode_rgb(clean_crop.float())

                        # LPIPS perceptual loss (if enabled)
                        if _use_lpips:
                            # LPIPS expects [-1, 1]; decode_rgb returns [0, 1]
                            pred_lpips_in = pred_rgb_crop * 2.0 - 1.0
                            clean_lpips_in = clean_rgb_crop * 2.0 - 1.0
                            # LPIPS returns [N, 1, 1, 1] per sample
                            pixel_lpips_per_sample = self.lpips_loss(pred_lpips_in, clean_lpips_in).view(-1)  # [N]
                            # Clamp per-sample values before weighting
                            if self.pixel_loss_max_value is not None:
                                pixel_lpips_per_sample = pixel_lpips_per_sample.clamp(max=self.pixel_loss_max_value)
                            # Weighted mean using timestep-adaptive weights
                            pixel_lpips = (pixel_lpips_per_sample * _t_weight).sum() / _t_weight.sum().clamp(min=1e-6)
                            if torch.isfinite(pixel_lpips):
                                loss = loss + self.lpips_weight * pixel_lpips
                                self.train_metrics.update("pixel_lpips", pixel_lpips.item())
                            else:
                                logging.warning(
                                    f"Non-finite LPIPS at iter {self.effective_iter + 1}, "
                                    f"skipping pixel loss for this batch."
                                )

                        # Pixel-space reconstruction loss — L1/L2 on same crop (if enabled)
                        if _use_pixel_recon:
                            # Per-sample reconstruction loss: reduce over C,H,W but keep batch dim
                            pixel_recon_per_sample = torch.nn.functional.l1_loss(
                                pred_rgb_crop, clean_rgb_crop, reduction='none'
                            ).mean(dim=[1, 2, 3]) if self.pixel_loss_type == 'l1' else torch.nn.functional.mse_loss(
                                pred_rgb_crop, clean_rgb_crop, reduction='none'
                            ).mean(dim=[1, 2, 3])  # [N]
                            # Clamp per-sample values before weighting
                            if self.pixel_loss_max_value is not None:
                                pixel_recon_per_sample = pixel_recon_per_sample.clamp(max=self.pixel_loss_max_value)
                            # Weighted mean using timestep-adaptive weights
                            pixel_recon = (pixel_recon_per_sample * _t_weight).sum() / _t_weight.sum().clamp(min=1e-6)
                            if torch.isfinite(pixel_recon):
                                loss = loss + self.pixel_loss_weight * pixel_recon
                                self.train_metrics.update("pixel_reconstruction", pixel_recon.item())
                            else:
                                logging.warning(
                                    f"Non-finite pixel reconstruction loss at iter {self.effective_iter + 1}, "
                                    f"skipping for this batch."
                                )

                        # DINO perceptual loss on same crop (if enabled)
                        if _use_dino:
                            # DINO expects [-1, 1]; decode_rgb returns [0, 1]
                            pred_dino_in = pred_rgb_crop * 2.0 - 1.0
                            clean_dino_in = clean_rgb_crop * 2.0 - 1.0
                            # Get per-sample DINO loss [N] via inner DINOPerceptual module
                            pixel_dino_per_sample = self.dino_loss.dino_loss(pred_dino_in, clean_dino_in)  # [N]
                            # Clamp per-sample values before weighting
                            if self.pixel_loss_max_value is not None:
                                pixel_dino_per_sample = pixel_dino_per_sample.clamp(max=self.pixel_loss_max_value)
                            # Weighted mean using timestep-adaptive weights
                            pixel_dino = (pixel_dino_per_sample * _t_weight).sum() / _t_weight.sum().clamp(min=1e-6)
                            if torch.isfinite(pixel_dino):
                                loss = loss + self.dino_weight * pixel_dino
                                self.train_metrics.update("pixel_dino", pixel_dino.item())
                            else:
                                logging.warning(
                                    f"Non-finite DINO loss at iter {self.effective_iter + 1}, "
                                    f"skipping for this batch."
                                )

                loss = loss / self.gradient_accumulation_steps

                # Loss clipping: cap total loss to prevent gradient explosion cascade.
                # When LPIPS/ARNIQA produce large values at high timesteps, the loss can
                # spike over several iterations, progressively corrupting model weights
                # until NaN. Clamping the loss bounds the gradient magnitude per step.
                # Threshold 1.0 ≈ 7x normal loss (~0.14). Previous thresholds of 5.0 and
                # 2.0 were insufficient — losses of 0.5–1.3 cascaded into NaN within ~8 iters.
                max_loss = 1.0
                if loss.item() > max_loss:
                    logging.warning(
                        f"Loss clipped from {loss.item():.4f} to {max_loss} "
                        f"at iter {self.effective_iter + 1}"
                    )
                    loss = loss.clamp(max=max_loss)

                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()

                # Compute ARNIQA adapter gradient and weight norms (for monitoring)
                # Pattern from restoration trainer (lines 956-980)
                if self.use_arniqa and self.arniqa_conditioner is not None:
                    scale_factor = self.scaler.get_scale() if self.scaler is not None else 1.0
                    scale_factor = max(scale_factor, 1e-8)

                    # Verified: ArniqaConditioner.get_adapter_grad_norms() (model.py line 790)
                    grad_norms = self.arniqa_conditioner.get_adapter_grad_norms()
                    grad_norms = {k: v / scale_factor for k, v in grad_norms.items()}

                    # Verified: ArniqaConditioner.get_adapter_weight_norms() (model.py line 818)
                    weight_norms = self.arniqa_conditioner.get_adapter_weight_norms()

                    self._arniqa_grad_norms = grad_norms
                    self._arniqa_weight_norms = weight_norms
                    self._arniqa_grad_norm = sum(v**2 for v in grad_norms.values()) ** 0.5
                    self._arniqa_weight_norm = sum(v**2 for v in weight_norms.values()) ** 0.5

                accumulated_step += 1
                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping on UNet + ControlNet + ARNIQA params
                    # Pattern from restoration trainer (lines 990-1004)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.unet.parameters(), max_norm=1.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.controlnet.parameters(), max_norm=1.0
                    )
                    if self.use_arniqa and self.arniqa_conditioner is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.arniqa_conditioner.parameters(), max_norm=1.0
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
                    # Log ARNIQA adapter metrics (if enabled)
                    # Pattern from restoration trainer (lines 1020-1060)
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

                    _dbg_extra = ""
                    if os.environ.get('ABLATION_DEBUG'):
                        _res_now = self.train_metrics.result()
                        if 'pixel_reconstruction' in _res_now:
                            _dbg_extra += f" pixel_l1={_res_now['pixel_reconstruction']:.5f}"
                        if 'pixel_lpips' in _res_now:
                            _dbg_extra += f" lpips={_res_now['pixel_lpips']:.5f}"
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}{_dbg_extra}"
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

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _train_step_callback(self):
        """Executed after every iteration.
        Copied from hybrid-002 trainer._train_step_callback (line 812-818)."""
        if self.checkpoint_mode == 'marigold':
            self._train_step_callback_marigold()
        else:
            self._train_step_callback_decoupled()

    def _train_step_callback_marigold(self):
        """Original Marigold checkpoint pattern.
        Copied from hybrid-002 trainer._train_step_callback_marigold (line 820-849)."""
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
        Copied from hybrid-002 trainer._train_step_callback_decoupled (line 851-874)."""
        # Save backup
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

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, ckpt_name, save_train_state):
        """Save hybrid-003 checkpoint: UNet + ControlNet + ARNIQA adapter.

        KEY DIFFERENCE from hybrid-002: saves UNet (trainable now, not frozen).
        Pattern from hybrid-002 trainer.save_checkpoint (line 876-962) with additions:
        - Saves unet/ (trainable in hybrid-003)
        - Saves controlnet/
        - Saves ARNIQA adapter (pattern from restoration trainer lines 1906-1919)
        - Saves scheduler/
        - Saves hybrid_003_config.json
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

        # Save UNet (trainable in hybrid-003, unlike hybrid-002 where it was frozen)
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
        else:
            logging.info("Skipping scheduler save (weights-only mode)")

        # Save ARNIQA adapter weights (if enabled)
        # Pattern from restoration trainer (lines 1906-1919)
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

        # Save hybrid-003 config for reproducibility
        hybrid_config = {
            "architecture": "hybrid-003",
            "unet_trainable": True,
            "controlnet_trainable": True,
            "arniqa_enabled": self.use_arniqa,
            "arniqa_stage": self.arniqa_stage if self.use_arniqa else None,
        }
        hybrid_config_path = os.path.join(ckpt_dir, "hybrid_003_config.json")
        with open(hybrid_config_path, "w") as f:
            json.dump(hybrid_config, f, indent=2)
        logging.info(f"Hybrid-003 config saved to: {hybrid_config_path}")

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

    def load_checkpoint(self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True):
        """Load hybrid-003 checkpoint: UNet + ControlNet + ARNIQA adapter + trainer state.

        KEY DIFFERENCE from hybrid-002: loads UNet weights (trainable in hybrid-003).
        Pattern from hybrid-002 trainer.load_checkpoint (line 963-1010).
        """
        logging.info(f"Loading checkpoint from: {ckpt_path}")

        # Load UNet weights (trainable in hybrid-003)
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

        # Load ARNIQA adapter weights (if enabled and checkpoint exists)
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
        Copied from hybrid-002 trainer._check_disk_space_and_cleanup (line 1012-1035)."""
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
        Copied from hybrid-002 trainer._log_checkpoint_info (line 1037-1058)."""
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
        Copied from hybrid-002 trainer._cleanup_old_checkpoints (line 1060-1116)."""
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
        Copied from hybrid-002 trainer.validate (line 1118-1165)."""
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
        Copied from hybrid-002 trainer.validate_decoupled (line 1167-1211)."""
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

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
        log_images_to_wandb: bool = False,
    ):
        """Validate on a single dataset.

        Copied from hybrid-002 trainer.validate_single_dataset (line 1213-1408).
        Uses model.single_infer() which handles ControlNet conditioning,
        8ch degraded_latent concatenation, and ARNIQA conditioning internally.
        """
        self.model.to(self.device)
        metric_tracker.reset()

        # Free CPU/GPU memory before validation to maximize headroom
        gc.collect()
        torch.cuda.empty_cache()

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

            # Process batch through pipeline
            # single_infer handles ControlNet conditioning, 8ch concat, and ARNIQA internally
            # Verified: pipeline.single_infer() (pipeline line 166-330)
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
            restored_rgb_batch = restored_rgb_batch_ts.detach().cpu().numpy()  # [B, 3, H, W] in [0, 1]

            # Process each image in batch for metrics and logging
            for b_idx in range(batch_size):
                restored_rgb_ts = torch.from_numpy(restored_rgb_batch[b_idx]).to(self.device)  # [3, H, W]
                clean_single_ts = clean_rgb_int[b_idx].to(self.device).float() / 255.0  # [3, H, W]

                # NaN guard: skip metrics if restored image contains NaN
                # (can happen in early training when model outputs are unstable)
                sample_metric_dict = {}
                has_nan = torch.isnan(restored_rgb_ts).any()
                if has_nan:
                    logging.warning(
                        f"NaN detected in restored image (batch {i}, idx {b_idx}), "
                        f"skipping metrics for this sample."
                    )
                    # Replace NaN with 0 for image saving/logging
                    restored_rgb_ts = torch.nan_to_num(restored_rgb_ts, nan=0.0)
                    restored_rgb_batch[b_idx] = restored_rgb_ts.cpu().numpy()

                if not has_nan:
                    for met_func in self.metric_funcs:
                        _metric_name = met_func.__name__
                        _metric = met_func(restored_rgb_ts, clean_single_ts)
                        sample_metric_dict[_metric_name] = float(_metric)
                        metric_tracker.update(_metric_name, _metric)

                # Pixel-space LPIPS on final restored image (if enabled)
                if self.lpips_loss is not None and self.lpips_weight > 0:
                    pred_lpips_in = restored_rgb_ts.unsqueeze(0) * 2.0 - 1.0
                    clean_lpips_in = clean_single_ts.unsqueeze(0) * 2.0 - 1.0
                    with torch.no_grad():
                        pixel_lpips_val = self.lpips_loss(pred_lpips_in, clean_lpips_in).mean()
                    metric_tracker.update("val_pixel_lpips", pixel_lpips_val.item())

                # Pixel-space DINO on final restored image (if enabled)
                if self.dino_loss is not None and self.dino_weight > 0:
                    pred_dino_in = restored_rgb_ts.unsqueeze(0) * 2.0 - 1.0
                    clean_dino_in = clean_single_ts.unsqueeze(0) * 2.0 - 1.0
                    with torch.no_grad():
                        pixel_dino_val = self.dino_loss(pred_dino_in, clean_dino_in)
                    metric_tracker.update("val_pixel_dino", pixel_dino_val.item())

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

        # Free memory accumulated during validation
        del wandb_images, wandb_image_metrics, val_loss_dicts
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def visualize(self):
        """Visualization: run inference and save images.
        Copied from hybrid-002 trainer.visualize (line 1410-1439)."""
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

        Adapted from hybrid-002 trainer._calculate_validation_loss (line 1440-1535).
        KEY DIFFERENCE: uses ARNIQA conditioning (apply_dropout=False) for text_embed
        when self.use_arniqa is True, instead of always using empty_text_embed.
        Verified pattern from restoration trainer (line 1700-1707).

        Returns:
            dict: Dictionary with 'total' loss value
        """
        device = self.device

        # Get data — same as training
        degraded_rgb = batch[self.degraded_rgb_type].to(device)  # [B, 3, H, W] in [-1, 1]
        clean_rgb = batch[self.clean_rgb_type].to(device)  # [B, 3, H, W] in [-1, 1]
        batch_size = degraded_rgb.shape[0]

        with torch.no_grad():
            # Encode clean RGB to latent — same as training
            clean_latent = self.encode_rgb(clean_rgb)  # [B, 4, h, w]
            # HYBRID DIFFERENCE: Encode degraded RGB to latent for 8ch UNet input
            degraded_latent = self.encode_rgb(degraded_rgb)  # [B, 4, h, w]

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

        # Conditioning: ARNIQA quality features or empty text embedding
        # KEY DIFFERENCE from hybrid-002: use ARNIQA with apply_dropout=False during validation
        # Verified pattern from restoration trainer (line 1700-1707)
        if self.use_arniqa:
            text_embed = self.arniqa_conditioner(degraded_rgb, apply_dropout=False)
        else:
            text_embed = self.empty_text_embed.to(device).repeat(
                (batch_size, 1, 1)
            )  # [B, 77, 1024]

        # ControlNet conditioning: degraded RGB in pixel space (no input noise during validation)
        controlnet_cond = degraded_rgb  # [B, 3, H, W] in [-1, 1]

        # Forward pass with FP16 mixed precision (matching training loop pattern)
        with autocast('cuda'):
            # ControlNet forward: produces residuals from degraded RGB conditioning
            # ControlNet receives noisy_latents (4ch) as sample, NOT the 8ch concat
            down_block_res, mid_block_res = self.model.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embed,
                controlnet_cond=controlnet_cond,
                return_dict=False,
            )

            # HYBRID DIFFERENCE: Concatenate degraded_latent + noisy_latents → [B, 8, h, w]
            cat_latents = torch.cat(
                [degraded_latent, noisy_latents], dim=1
            )  # [B, 8, h, w]

            # 8ch UNet forward: inject ControlNet residuals
            model_pred = self.model.unet(
                cat_latents,
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

        Copied from hybrid-002 trainer._create_comparison_image (line 1537-1589).

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
