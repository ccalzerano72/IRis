#!/usr/bin/env python3
"""
Checkpoint 4: Verify that MarigoldControlNetRestorationTrainer initializes
correctly, can run a single training step, and save/load checkpoints.

Tests:
1. Trainer initialization with ControlNet from UNet
2. Frozen UNet / trainable ControlNet invariants
3. Single training step (forward + backward + optimizer step)
4. Checkpoint save + load round-trip
"""

import sys
import os
import tempfile
import logging
import torch
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_minimal_config():
    """Create a minimal OmegaConf config with all required keys for the trainer."""
    cfg = OmegaConf.create({
        "lr": 1e-4,
        "max_epoch": 1,
        "max_iter": 10,
        "degraded_rgb_type": "degraded_rgb_norm",
        "clean_rgb_type": "clean_rgb_norm",
        "lr_scheduler": {
            "name": "IterExponential",
            "kwargs": {
                "total_iter": 10,
                "final_ratio": 0.01,
                "warmup_steps": 0,
            },
        },
        "loss": {
            "name": "mse_loss",
            "kwargs": {
                "reduction": "mean",
            },
        },
        "trainer": {
            "init_seed": 42,
            "save_period": 5,
            "backup_period": 0,
            "validation_period": 0,
            "visualization_period": 0,
            "gradient_checkpointing": False,
            "save_trainer_state": True,
            "checkpoint_strategy": {
                "mode": "marigold",
            },
        },
        "eval": {
            "eval_metrics": ["psnr", "ssim"],
        },
        "validation": {
            "main_val_metric": "psnr",
            "main_val_metric_goal": "maximize",
            "init_seed": 42,
            "denoising_steps": 2,
            "max_images_to_log": 4,
            "log_images_during_validation": False,
        },
    })
    return cfg


def create_fake_dataloader(batch_size=2, num_samples=4, resolution=64):
    """Create a fake dataloader with random degraded/clean RGB pairs."""
    torch.manual_seed(42)
    degraded = torch.randn(num_samples, 3, resolution, resolution).clamp(-1, 1)
    clean = torch.randn(num_samples, 3, resolution, resolution).clamp(-1, 1)
    dataset = TensorDataset(degraded, clean)

    class DictDataLoader:
        """Wraps TensorDataset to return dicts like the real restoration dataset."""
        def __init__(self, tensor_loader):
            self._loader = tensor_loader
            self.dataset = tensor_loader.dataset
        def __iter__(self):
            for degraded_batch, clean_batch in self._loader:
                yield {
                    "degraded_rgb_norm": degraded_batch,
                    "clean_rgb_norm": clean_batch,
                }
        def __len__(self):
            return len(self._loader)

    tensor_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return DictDataLoader(tensor_loader)


def test_trainer():
    print("=== Checkpoint 4: ControlNet Trainer Verification ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: Load SD2 + create ControlNet + pipeline ---
    print("\n[1/4] Loading SD2 components and creating pipeline...")
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDIMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from marigold import MarigoldControlNetRestorationPipeline

    pretrained_path = "stabilityai/stable-diffusion-2"
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_path, subfolder="unet", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_path, subfolder="vae", torch_dtype=torch.float16
    )
    scheduler = DDIMScheduler.from_pretrained(
        pretrained_path, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_path, subfolder="text_encoder", torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_path, subfolder="tokenizer"
    )
    controlnet = ControlNetModel.from_unet(unet)  # float32 — trainable params must be float32 for GradScaler

    pipeline = MarigoldControlNetRestorationPipeline(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        default_denoising_steps=2,
        default_processing_resolution=64,
    )
    print("  OK - Pipeline created")

    # --- Step 2: Initialize trainer ---
    print("\n[2/4] Initializing trainer...")
    from src.trainer.marigold_controlnet_restoration_trainer import (
        MarigoldControlNetRestorationTrainer,
    )

    cfg = create_minimal_config()
    train_loader = create_fake_dataloader(batch_size=2, num_samples=4, resolution=64)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir_ckpt = os.path.join(tmpdir, "checkpoint")
        out_dir_eval = os.path.join(tmpdir, "evaluation")
        out_dir_vis = os.path.join(tmpdir, "visualization")
        os.makedirs(out_dir_ckpt)
        os.makedirs(out_dir_eval)
        os.makedirs(out_dir_vis)

        trainer = MarigoldControlNetRestorationTrainer(
            cfg=cfg,
            model=pipeline,
            train_dataloader=train_loader,
            device=device,
            out_dir_ckpt=out_dir_ckpt,
            out_dir_eval=out_dir_eval,
            out_dir_vis=out_dir_vis,
            accumulation_steps=1,
        )
        print("  OK - Trainer initialized")

        # Verify frozen UNet / trainable ControlNet
        print("\n  Checking invariants:")
        # UNet frozen
        unet_grads = [p.requires_grad for p in trainer.model.unet.parameters()]
        assert not any(unet_grads), "UNet should be fully frozen"
        assert not trainer.model.unet.training, "UNet should be in eval mode"
        print("    OK - UNet: all params frozen, eval mode")

        # ControlNet trainable
        cnet_grads = [p.requires_grad for p in trainer.model.controlnet.parameters()]
        assert all(cnet_grads), "ControlNet should be fully trainable"
        print("    OK - ControlNet: all params trainable")

        # Optimizer only has ControlNet params
        opt_param_count = sum(
            p.numel() for group in trainer.optimizer.param_groups for p in group["params"]
        )
        cnet_param_count = sum(p.numel() for p in trainer.model.controlnet.parameters())
        assert opt_param_count == cnet_param_count, (
            f"Optimizer params ({opt_param_count}) != ControlNet params ({cnet_param_count})"
        )
        print(f"    OK - Optimizer: {opt_param_count / 1e6:.1f}M params (matches ControlNet)")

        # --- Step 3: Single training step ---
        print("\n[3/4] Running single training step...")
        trainer.model.to(device)
        trainer.model.controlnet.train()

        # Get one batch
        batch = next(iter(train_loader))
        degraded_rgb = batch["degraded_rgb_norm"].to(device)
        clean_rgb = batch["clean_rgb_norm"].to(device)
        batch_size = degraded_rgb.shape[0]

        # Encode clean to latent (must use autocast since VAE is in float16)
        with torch.no_grad(), autocast('cuda'):
            clean_latent = trainer.encode_rgb(clean_rgb)

        # Sample timestep and noise
        timesteps = torch.randint(0, trainer.scheduler_timesteps, (batch_size,), device=device).long()
        noise = torch.randn(clean_latent.shape, device=device)
        noisy_latents = trainer.training_noise_scheduler.add_noise(clean_latent, noise, timesteps)

        # Text embed
        text_embed = trainer.empty_text_embed.to(device).repeat((batch_size, 1, 1))

        # Forward pass with autocast
        with autocast('cuda'):
            down_block_res, mid_block_res = trainer.model.controlnet(
                noisy_latents, timesteps,
                encoder_hidden_states=text_embed,
                controlnet_cond=degraded_rgb,
                return_dict=False,
            )
            model_pred = trainer.model.unet(
                noisy_latents, timesteps,
                encoder_hidden_states=text_embed,
                down_block_additional_residuals=down_block_res,
                mid_block_additional_residual=mid_block_res,
            ).sample

        # Target (v_prediction)
        target = trainer.training_noise_scheduler.get_velocity(clean_latent, noise, timesteps)

        # Loss + backward
        loss = trainer.loss(model_pred.float(), target.float()).mean()
        trainer.scaler.scale(loss).backward()

        print(f"  OK - Forward pass completed, loss = {loss.item():.6f}")
        print(f"       model_pred shape: {model_pred.shape}")

        # Check gradient isolation
        unet_has_grad = any(
            p.grad is not None for p in trainer.model.unet.parameters()
        )
        cnet_has_grad = any(
            p.grad is not None for p in trainer.model.controlnet.parameters()
        )
        assert not unet_has_grad, "UNet should have NO gradients"
        assert cnet_has_grad, "ControlNet should have gradients"
        print("    OK - Gradient isolation: UNet=no grads, ControlNet=has grads")

        # Optimizer step
        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer.model.controlnet.parameters(), max_norm=1.0)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.optimizer.zero_grad()
        trainer.effective_iter = 1
        print("    OK - Optimizer step completed")

        # --- Step 4: Checkpoint save + load ---
        print("\n[4/4] Testing checkpoint save/load round-trip...")

        # Save checkpoint
        trainer.save_checkpoint(ckpt_name="test_ckpt", save_train_state=True)
        ckpt_path = os.path.join(out_dir_ckpt, "test_ckpt")
        assert os.path.exists(ckpt_path), f"Checkpoint dir not found: {ckpt_path}"
        assert os.path.exists(os.path.join(ckpt_path, "controlnet")), "controlnet/ dir missing"
        assert os.path.exists(os.path.join(ckpt_path, "controlnet", "diffusion_pytorch_model.safetensors")), "safetensors missing"
        assert os.path.exists(os.path.join(ckpt_path, "scheduler")), "scheduler/ dir missing"
        assert os.path.exists(os.path.join(ckpt_path, "trainer.ckpt")), "trainer.ckpt missing"
        assert os.path.exists(os.path.join(ckpt_path, "restoration_config.json")), "restoration_config.json missing"
        print("  OK - Checkpoint saved with all expected files")

        # Capture weights before load
        weight_before = trainer.model.controlnet.controlnet_down_blocks[0].weight.clone()

        # Perturb weights to verify load actually changes them
        with torch.no_grad():
            trainer.model.controlnet.controlnet_down_blocks[0].weight.fill_(0.0)

        weight_zeroed = trainer.model.controlnet.controlnet_down_blocks[0].weight.clone()
        assert torch.all(weight_zeroed == 0.0), "Weights should be zeroed"

        # Load checkpoint
        trainer.load_checkpoint(ckpt_path, load_trainer_state=True, resume_lr_scheduler=True)

        weight_after = trainer.model.controlnet.controlnet_down_blocks[0].weight.clone()
        assert torch.allclose(weight_before.cpu().float(), weight_after.cpu().float(), atol=1e-6), (
            "Loaded weights should match saved weights"
        )
        print("  OK - Checkpoint loaded, weights match (round-trip verified)")

        # Verify trainer state was restored
        assert trainer.effective_iter == 1, f"effective_iter should be 1, got {trainer.effective_iter}"
        print(f"  OK - Trainer state restored (effective_iter={trainer.effective_iter})")

    print("\n=== Checkpoint 4 PASSED ===")
    return True


if __name__ == "__main__":
    try:
        success = test_trainer()
    except Exception as e:
        print(f"\n=== Checkpoint 4 FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)
