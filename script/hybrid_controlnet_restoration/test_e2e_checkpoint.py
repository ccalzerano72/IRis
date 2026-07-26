#!/usr/bin/env python3
"""
Checkpoint 8: End-to-end verification of Hybrid ControlNet restoration pipeline.

Tests the full workflow:
1. Create hybrid pipeline + trainer with fake data (8ch UNet + ControlNet)
2. Run 2 training iterations (hybrid forward: ControlNet + 8ch UNet with degraded_latent concat)
3. Save checkpoint (controlnet/, scheduler/, hybrid_config.json)
4. Load checkpoint into inference pipeline (run.py pattern via load_hybrid_pipeline)
5. Run inference on a fake image and verify output

This tests the integration between train.py and run.py patterns
without needing real dataset files.

Pattern: Exact copy of script/controlnet_restoration/test_e2e_checkpoint.py
with hybrid-specific differences (8ch UNet, degraded_latent concat, base_checkpoint_path).
"""

import sys
import os
import tempfile
import logging
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def create_minimal_config():
    """Minimal config for hybrid trainer initialization.
    Same as ControlNet e2e config but with lpips_weight: 0.0 in loss kwargs.
    """
    return OmegaConf.create({
        "lr": 1e-4,
        "max_epoch": 1,
        "max_iter": 10,
        "degraded_rgb_type": "degraded_rgb_norm",
        "clean_rgb_type": "clean_rgb_norm",
        "lr_scheduler": {
            "name": "IterExponential",
            "kwargs": {"total_iter": 10, "final_ratio": 0.01, "warmup_steps": 0},
        },
        "loss": {
            "name": "mse_loss",
            "kwargs": {"reduction": "mean", "lpips_weight": 0.0},
        },
        "trainer": {
            "init_seed": 42,
            "save_period": 5,
            "backup_period": 0,
            "validation_period": 0,
            "visualization_period": 0,
            "gradient_checkpointing": False,
            "save_trainer_state": True,
            "checkpoint_strategy": {"mode": "marigold"},
        },
        "eval": {"eval_metrics": ["psnr", "ssim"]},
        "validation": {
            "main_val_metric": "psnr",
            "main_val_metric_goal": "maximize",
            "init_seed": 42,
            "denoising_steps": 2,
            "max_images_to_log": 4,
            "log_images_during_validation": False,
        },
    })


def create_fake_dataloader(batch_size=2, num_samples=4, resolution=64):
    """Create a fake dataloader returning dicts like the real restoration dataset."""
    torch.manual_seed(42)
    degraded = torch.randn(num_samples, 3, resolution, resolution).clamp(-1, 1)
    clean = torch.randn(num_samples, 3, resolution, resolution).clamp(-1, 1)
    dataset = TensorDataset(degraded, clean)

    class DictDataLoader:
        def __init__(self, tensor_loader):
            self._loader = tensor_loader
            self.dataset = tensor_loader.dataset
        def __iter__(self):
            for degraded_batch, clean_batch in self._loader:
                yield {"degraded_rgb_norm": degraded_batch, "clean_rgb_norm": clean_batch}
        def __len__(self):
            return len(self._loader)

    tensor_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return DictDataLoader(tensor_loader)


def test_e2e():
    print("=== Checkpoint 8: Hybrid ControlNet End-to-End Verification ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: Create hybrid pipeline + trainer ---
    print("\n[1/5] Creating hybrid pipeline and trainer...")
    from diffusers import (
        AutoencoderKL, ControlNetModel, DDIMScheduler, UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from marigold import MarigoldHybridControlNetRestorationPipeline
    from src.trainer.marigold_hybrid_controlnet_restoration_trainer import (
        MarigoldHybridControlNetRestorationTrainer,
    )

    pretrained_path = "stabilityai/stable-diffusion-2"
    unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae", torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder", torch_dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
    # ControlNet stays float32 (trainable)
    controlnet = ControlNetModel.from_unet(unet)

    pipeline = MarigoldHybridControlNetRestorationPipeline(
        unet=unet, controlnet=controlnet, vae=vae, scheduler=scheduler,
        text_encoder=text_encoder, tokenizer=tokenizer,
        default_denoising_steps=2, default_processing_resolution=64,
    )

    cfg = create_minimal_config()
    train_loader = create_fake_dataloader(batch_size=2, num_samples=4, resolution=64)

    # Base checkpoint path (symlinked by user to 8ch UNet checkpoint)
    base_checkpoint_path = os.path.join(project_root, "ckpt", "base_restoration_checkpoint")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir_ckpt = os.path.join(tmpdir, "checkpoint")
        out_dir_eval = os.path.join(tmpdir, "evaluation")
        out_dir_vis = os.path.join(tmpdir, "visualization")
        os.makedirs(out_dir_ckpt)
        os.makedirs(out_dir_eval)
        os.makedirs(out_dir_vis)

        # Hybrid trainer requires base_checkpoint_path
        # Verified: __init__ at line 50 of marigold_hybrid_controlnet_restoration_trainer.py
        trainer = MarigoldHybridControlNetRestorationTrainer(
            cfg=cfg, model=pipeline, train_dataloader=train_loader,
            device=device, out_dir_ckpt=out_dir_ckpt, out_dir_eval=out_dir_eval,
            out_dir_vis=out_dir_vis, accumulation_steps=1,
            base_checkpoint_path=base_checkpoint_path,
        )
        print("  OK - Hybrid trainer initialized (8ch UNet + ControlNet)")

        # --- Step 2: Run 2 training iterations (hybrid forward) ---
        print("\n[2/5] Running 2 training iterations (hybrid forward)...")
        trainer.model.to(device)

        for step in range(2):
            trainer.model.controlnet.train()
            batch = next(iter(train_loader))
            degraded_rgb = batch["degraded_rgb_norm"].to(device)
            clean_rgb = batch["clean_rgb_norm"].to(device)
            batch_size = degraded_rgb.shape[0]

            # Encode both degraded and clean to latent space
            # Verified: encode_rgb at line 409 of hybrid trainer
            with torch.no_grad(), autocast('cuda'):
                degraded_latent = trainer.encode_rgb(degraded_rgb)
                clean_latent = trainer.encode_rgb(clean_rgb)

            timesteps = torch.randint(0, trainer.scheduler_timesteps, (batch_size,), device=device).long()
            noise = torch.randn(clean_latent.shape, device=device)
            noisy_latents = trainer.training_noise_scheduler.add_noise(clean_latent, noise, timesteps)
            text_embed = trainer.empty_text_embed.to(device).repeat((batch_size, 1, 1))

            with autocast('cuda'):
                # ControlNet forward: pixel-space degraded RGB conditioning
                down_res, mid_res = trainer.model.controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=text_embed,
                    controlnet_cond=degraded_rgb,
                    return_dict=False,
                )
                # Hybrid: concatenate degraded_latent + noisy_latents for 8ch UNet
                cat_latents = torch.cat([degraded_latent, noisy_latents], dim=1)
                model_pred = trainer.model.unet(
                    cat_latents, timesteps,
                    encoder_hidden_states=text_embed,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample

            target = trainer.training_noise_scheduler.get_velocity(clean_latent, noise, timesteps)
            loss = trainer.loss(model_pred.float(), target.float()).mean()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.controlnet.parameters(), max_norm=1.0)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            trainer.optimizer.zero_grad()
            trainer.effective_iter += 1
            print(f"  Step {step + 1}: loss = {loss.item():.6f}")

        print("  OK - 2 training iterations completed (hybrid forward)")

        # --- Step 3: Save checkpoint ---
        print("\n[3/5] Saving checkpoint...")
        # Verified: save_checkpoint at line 850 saves controlnet/, scheduler/, hybrid_config.json
        trainer.save_checkpoint(ckpt_name="e2e_test", save_train_state=True)
        ckpt_path = os.path.join(out_dir_ckpt, "e2e_test")
        assert os.path.exists(os.path.join(ckpt_path, "controlnet", "diffusion_pytorch_model.safetensors")), \
            "Missing controlnet/diffusion_pytorch_model.safetensors"
        assert os.path.exists(os.path.join(ckpt_path, "scheduler")), \
            "Missing scheduler/ directory"
        assert os.path.exists(os.path.join(ckpt_path, "hybrid_config.json")), \
            "Missing hybrid_config.json"
        print(f"  OK - Checkpoint saved to: {ckpt_path}")
        print(f"  Contents: {os.listdir(ckpt_path)}")

        # --- Step 4: Load checkpoint into inference pipeline (run.py pattern) ---
        print("\n[4/5] Loading checkpoint into inference pipeline (run.py pattern)...")
        # Verified: load_hybrid_pipeline at line 72 of script/hybrid_controlnet_restoration/run.py
        # Signature: (base_checkpoint_path, controlnet_checkpoint_path, sd2_model, dtype, scheduler_type, pipeline_kwargs)
        from script.hybrid_controlnet_restoration.run import load_hybrid_pipeline

        inference_pipe = load_hybrid_pipeline(
            base_checkpoint_path=base_checkpoint_path,
            controlnet_checkpoint_path=ckpt_path,
            sd2_model=pretrained_path,
            dtype=torch.float16,
            scheduler_type="ddim",
        )
        inference_pipe = inference_pipe.to(device)
        print("  OK - Inference pipeline loaded from checkpoint")

        # Verify scheduler was fixed
        assert inference_pipe.scheduler.config.timestep_spacing == "trailing"
        assert inference_pipe.scheduler.config.rescale_betas_zero_snr == True
        print("  OK - Scheduler config: trailing + rescale_betas_zero_snr=True")

        # Verify 8ch UNet
        assert inference_pipe.unet.conv_in.weight.shape[1] == 8, \
            f"Expected 8ch UNet, got {inference_pipe.unet.conv_in.weight.shape[1]}ch"
        print("  OK - UNet conv_in: 8 channels")

        # --- Step 5: Run inference on a fake image ---
        print("\n[5/5] Running inference on a fake image...")
        fake_image_np = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fake_image = Image.fromarray(fake_image_np)

        generator = torch.Generator(device=device).manual_seed(42)

        # Verified: MarigoldHybridControlNetRestorationPipeline.__call__ at line 317
        from marigold.marigold_hybrid_controlnet_restoration_pipeline import MarigoldRestorationOutput
        pipe_out: MarigoldRestorationOutput = inference_pipe(
            fake_image,
            denoising_steps=2,
            ensemble_size=1,
            processing_res=64,
            match_input_res=True,
            batch_size=1,
            show_progress_bar=False,
            generator=generator,
            guidance_scale=1.0,
        )

        assert pipe_out.restored_np is not None, "restored_np should not be None"
        assert pipe_out.restored_img is not None, "restored_img should not be None"
        assert isinstance(pipe_out.restored_img, Image.Image), "restored_img should be PIL Image"
        assert pipe_out.restored_np.shape[0] == 3, f"Expected 3 channels, got {pipe_out.restored_np.shape[0]}"
        assert pipe_out.restored_np.min() >= 0.0, f"Min value {pipe_out.restored_np.min()} < 0"
        assert pipe_out.restored_np.max() <= 1.0, f"Max value {pipe_out.restored_np.max()} > 1"
        print(f"  OK - Inference output: shape={pipe_out.restored_np.shape}, "
              f"range=[{pipe_out.restored_np.min():.3f}, {pipe_out.restored_np.max():.3f}]")
        print(f"  OK - PIL Image size: {pipe_out.restored_img.size}")

        # Test with CFG
        print("\n  [Bonus] Testing inference with CFG (guidance_scale=2.0)...")
        generator = torch.Generator(device=device).manual_seed(42)
        pipe_out_cfg = inference_pipe(
            fake_image,
            denoising_steps=2,
            ensemble_size=1,
            processing_res=64,
            match_input_res=True,
            batch_size=1,
            show_progress_bar=False,
            generator=generator,
            guidance_scale=2.0,
        )
        assert pipe_out_cfg.restored_np is not None
        print(f"  OK - CFG inference: shape={pipe_out_cfg.restored_np.shape}, "
              f"range=[{pipe_out_cfg.restored_np.min():.3f}, {pipe_out_cfg.restored_np.max():.3f}]")

    print("\n=== Checkpoint 8 PASSED ===")
    return True


if __name__ == "__main__":
    try:
        success = test_e2e()
    except Exception as e:
        print(f"\n=== Checkpoint 8 FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)
