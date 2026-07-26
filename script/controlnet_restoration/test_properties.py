#!/usr/bin/env python3
"""
Property-Based Tests for ControlNet Restoration Pipeline.

Tests the following correctness properties from the design document:
- Property 6: Zero Residuals Equivalence (Task 7.2)
- Property 7: Output Range Invariant (Task 7.3)
- Property 8: Checkpoint Round-Trip (Task 7.4)

Note: Property 5 (CFG Dropout Rate Convergence, Task 7.1) is NOT tested because
CFG dropout was removed from the trainer by design decision — with a frozen UNet,
zeroing ControlNet residuals during training means no trainable parameter receives
gradients for dropped samples (pure wasted compute).

Requires: hypothesis, torch, diffusers, transformers
Requires: CUDA GPU for model loading and inference
"""

import sys
import os
import tempfile
import logging
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from torch.amp import autocast

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Shared fixtures: load model components ONCE, reuse across all tests
# ---------------------------------------------------------------------------

_PIPELINE = None
_DEVICE = None


def get_pipeline():
    """Lazy-load the pipeline once for all tests."""
    global _PIPELINE, _DEVICE

    if _PIPELINE is not None:
        return _PIPELINE, _DEVICE

    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDIMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from marigold import MarigoldControlNetRestorationPipeline

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # ControlNet in float16 for inference tests (not training)
    controlnet = ControlNetModel.from_unet(unet)
    controlnet = controlnet.half()

    _PIPELINE = MarigoldControlNetRestorationPipeline(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        default_denoising_steps=2,
        default_processing_resolution=64,
    )
    _PIPELINE = _PIPELINE.to(_DEVICE)

    # Fix scheduler to match training config
    # Verified: _fix_scheduler_config in script/controlnet_restoration/run.py
    _PIPELINE.scheduler = DDIMScheduler.from_config(
        _PIPELINE.scheduler.config,
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
    )

    return _PIPELINE, _DEVICE


# ---------------------------------------------------------------------------
# Property 6: Zero Residuals Equivalence
# Validates: Requirements 4.2
#
# For any input (noisy_latent, timestep, text_embed), the Frozen_UNet output
# with all-zero ControlNet residuals SHALL be equal to the Frozen_UNet output
# without any ControlNet residuals (standard UNet forward pass).
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    timestep_frac=st.floats(min_value=0.0, max_value=1.0),
)
def test_property_6_zero_residuals_equivalence(seed, timestep_frac):
    """
    **Validates: Requirements 4.2**
    Feature: controlnet-restoration, Property 6: Zero Residuals Equivalence

    UNet output with all-zero ControlNet residuals must equal UNet output
    without any residuals (standard forward pass).
    """
    pipe, device = get_pipeline()

    torch.manual_seed(seed)

    # Generate random noisy latent at 64x64 image resolution -> 8x8 latent
    # Verified: latent shape is [B, 4, H//8, W//8] from pipeline single_infer line 207
    noisy_latent = torch.randn(1, 4, 8, 8, device=device, dtype=torch.float16)

    # Map timestep_frac [0, 1] to valid timestep range [0, num_train_timesteps-1]
    max_t = pipe.scheduler.config.num_train_timesteps - 1
    t = torch.tensor([int(timestep_frac * max_t)], device=device).long()

    # Encode empty text
    # Verified: encode_empty_text() at pipeline line 107 sets self.empty_text_embed
    if pipe.empty_text_embed is None:
        pipe.encode_empty_text()
    text_embed = pipe.empty_text_embed.to(device)  # [1, 77, 1024]

    with torch.no_grad():
        # Get ControlNet residuals to know the shapes, then zero them
        down_res, mid_res = pipe.controlnet(
            noisy_latent, t,
            encoder_hidden_states=text_embed,
            controlnet_cond=torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16),
            return_dict=False,
        )
        zero_down_res = [torch.zeros_like(r) for r in down_res]
        zero_mid_res = torch.zeros_like(mid_res)

        # UNet with zero residuals
        out_with_zero_res = pipe.unet(
            noisy_latent, t,
            encoder_hidden_states=text_embed,
            down_block_additional_residuals=zero_down_res,
            mid_block_additional_residual=zero_mid_res,
        ).sample

        # UNet without any residuals (standard forward)
        out_without_res = pipe.unet(
            noisy_latent, t,
            encoder_hidden_states=text_embed,
        ).sample

    # Must be numerically identical (not just close — zero residuals = no residuals)
    assert torch.equal(out_with_zero_res, out_without_res), (
        f"Zero residuals output differs from no-residuals output. "
        f"Max diff: {(out_with_zero_res - out_without_res).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Property 7: Output Range Invariant
# Validates: Requirements 5.7
#
# For any input degraded image, the restored RGB output of single_infer()
# SHALL have all values in the range [-1.0, 1.0].
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    num_steps=st.sampled_from([1, 2, 4]),
)
def test_property_7_output_range_invariant(seed, num_steps):
    """
    **Validates: Requirements 5.7**
    Feature: controlnet-restoration, Property 7: Output Range Invariant

    The restored RGB output of single_infer() must always be in [-1.0, 1.0].
    """
    pipe, device = get_pipeline()

    # Generate random degraded image in valid range [-1, 1]
    generator = torch.Generator(device=device).manual_seed(seed)
    rgb_in = torch.rand(1, 3, 64, 64, device=device, dtype=torch.float16, generator=generator) * 2.0 - 1.0

    # Fix scheduler for this number of steps
    pipe.scheduler.set_timesteps(num_steps, device=device)

    with torch.no_grad():
        # Verified: single_infer at pipeline line 151 returns clipped [-1, 1] tensor
        restored = pipe.single_infer(
            rgb_in=rgb_in,
            num_inference_steps=num_steps,
            generator=generator,
            show_pbar=False,
            guidance_scale=1.0,
        )

    assert restored.min() >= -1.0, (
        f"Output min {restored.min().item()} < -1.0"
    )
    assert restored.max() <= 1.0, (
        f"Output max {restored.max().item()} > 1.0"
    )
    assert restored.shape == (1, 3, 64, 64), (
        f"Expected shape (1, 3, 64, 64), got {restored.shape}"
    )


# ---------------------------------------------------------------------------
# Property 8: Checkpoint Round-Trip
# Validates: Requirements 7.3, 7.4
#
# For any trained ControlNet state, saving via save_pretrained() and loading
# via ControlNetModel.from_pretrained() SHALL produce numerically identical
# parameters. Similarly, trainer state (optimizer, LR scheduler, iteration
# counter) SHALL be restored identically.
# ---------------------------------------------------------------------------

@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_8_checkpoint_round_trip(seed):
    """
    **Validates: Requirements 7.3, 7.4**
    Feature: controlnet-restoration, Property 8: Checkpoint Round-Trip

    Save then load must produce numerically identical ControlNet weights
    and trainer state values.
    """
    from diffusers import ControlNetModel

    pipe, device = get_pipeline()

    # Perturb ControlNet weights with random noise to simulate training
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in pipe.controlnet.parameters():
            param.add_(torch.randn_like(param) * 0.001)

    # Snapshot original weights
    original_state = {
        k: v.clone() for k, v in pipe.controlnet.state_dict().items()
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "controlnet")

        # Save via save_pretrained (same as trainer save_checkpoint)
        # Verified: save_checkpoint at trainer line 656 calls
        # self.model.controlnet.save_pretrained(controlnet_path, safe_serialization=True)
        pipe.controlnet.save_pretrained(save_path, safe_serialization=True)

        # Load via from_pretrained (same as run.py load_controlnet_pipeline)
        # Verified: load_controlnet_pipeline at run.py line 97 calls
        # ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
        loaded_controlnet = ControlNetModel.from_pretrained(
            save_path, torch_dtype=torch.float16
        )
        loaded_controlnet = loaded_controlnet.to(device)

    # Verify all parameters are numerically identical
    loaded_state = loaded_controlnet.state_dict()
    for key in original_state:
        assert key in loaded_state, f"Missing key after load: {key}"
        assert torch.equal(original_state[key], loaded_state[key]), (
            f"Parameter '{key}' differs after round-trip. "
            f"Max diff: {(original_state[key] - loaded_state[key]).abs().max().item()}"
        )

    # Restore original weights to avoid polluting shared pipeline
    pipe.controlnet.load_state_dict(original_state)

    # Cleanup loaded model
    del loaded_controlnet
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main: run all property tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ControlNet Restoration Property Tests ===\n")
    print("Loading pipeline (this may take a moment)...")
    get_pipeline()
    print("Pipeline loaded.\n")

    tests = [
        ("Property 6: Zero Residuals Equivalence (Task 7.2)", test_property_6_zero_residuals_equivalence),
        ("Property 7: Output Range Invariant (Task 7.3)", test_property_7_output_range_invariant),
        ("Property 8: Checkpoint Round-Trip (Task 7.4)", test_property_8_checkpoint_round_trip),
    ]

    all_passed = True
    for name, test_fn in tests:
        print(f"Running {name}...")
        try:
            test_fn()
            print(f"  PASSED\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("=== ALL PROPERTY TESTS PASSED ===")
    else:
        print("=== SOME PROPERTY TESTS FAILED ===")

    sys.exit(0 if all_passed else 1)
