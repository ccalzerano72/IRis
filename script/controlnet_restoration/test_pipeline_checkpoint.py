#!/usr/bin/env python3
"""
Checkpoint 2: Verify that MarigoldControlNetRestorationPipeline loads
and runs basic inference end-to-end.

Tests:
1. Load SD2 base model components
2. Create ControlNet from UNet via ControlNetModel.from_unet()
3. Instantiate the pipeline with all components
4. Run single_infer() on a small random input
5. Verify output shape and value range
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_pipeline_basic_inference():
    """Test that the ControlNet pipeline loads and produces valid output."""

    print("=== Checkpoint 2: ControlNet Pipeline Verification ===\n")

    # --- Step 1: Load SD2 base model components ---
    print("[1/5] Loading SD2 base model components...")
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDIMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

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
    print("  OK - All SD2 components loaded")

    # --- Step 2: Create ControlNet from UNet ---
    print("[2/5] Creating ControlNet from UNet via from_unet()...")
    controlnet = ControlNetModel.from_unet(unet)
    controlnet = controlnet.to(dtype=torch.float16)
    print(f"  OK - ControlNet created, params: {sum(p.numel() for p in controlnet.parameters()) / 1e6:.1f}M")

    # --- Step 3: Instantiate pipeline ---
    print("[3/5] Instantiating MarigoldControlNetRestorationPipeline...")
    from marigold import MarigoldControlNetRestorationPipeline

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    print(f"  OK - Pipeline instantiated on {device}")

    # --- Step 4: Run single_infer() on small random input ---
    print("[4/5] Running single_infer() with 2 denoising steps...")

    # Create a small random input tensor simulating a degraded image
    # Shape: [1, 3, 64, 64], values in [-1, 1] (normalized RGB)
    torch.manual_seed(42)
    rgb_in = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16).clamp(-1.0, 1.0)

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    output = pipeline.single_infer(
        rgb_in=rgb_in,
        num_inference_steps=2,
        generator=generator,
        show_pbar=False,
        guidance_scale=1.0,
    )
    print(f"  OK - single_infer() completed, output shape: {output.shape}")

    # --- Step 5: Verify output shape and value range ---
    print("[5/5] Verifying output shape and value range...")

    expected_shape = (1, 3, 64, 64)
    assert output.shape == expected_shape, (
        f"Output shape mismatch: got {output.shape}, expected {expected_shape}"
    )
    print(f"  OK - Shape correct: {output.shape}")

    assert output.min() >= -1.0, f"Output min {output.min().item():.4f} < -1.0"
    assert output.max() <= 1.0, f"Output max {output.max().item():.4f} > 1.0"
    print(f"  OK - Value range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Also test with CFG (guidance_scale > 1.0)
    print("\n[Bonus] Testing with CFG (guidance_scale=2.0)...")
    generator.manual_seed(42)
    output_cfg = pipeline.single_infer(
        rgb_in=rgb_in,
        num_inference_steps=2,
        generator=generator,
        show_pbar=False,
        guidance_scale=2.0,
    )
    assert output_cfg.shape == expected_shape, (
        f"CFG output shape mismatch: got {output_cfg.shape}, expected {expected_shape}"
    )
    assert output_cfg.min() >= -1.0 and output_cfg.max() <= 1.0, (
        f"CFG output range: [{output_cfg.min().item():.4f}, {output_cfg.max().item():.4f}]"
    )
    print(f"  OK - CFG output shape: {output_cfg.shape}, "
          f"range: [{output_cfg.min().item():.4f}, {output_cfg.max().item():.4f}]")

    print("\n=== Checkpoint 2 PASSED ===")
    return True


if __name__ == "__main__":
    try:
        success = test_pipeline_basic_inference()
    except Exception as e:
        print(f"\n=== Checkpoint 2 FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)
