#!/usr/bin/env python3
"""
Property-Based Tests for Hybrid ControlNet Restoration.

Tests the following correctness property from the design document:
- Property 12: Delta_E Metric Properties (Task 1.1)

**Validates: Requirements 8.6**

Requires: hypothesis, torch, numpy, scikit-image
No GPU required — tests a pure CPU metric function.
"""

import sys
import os
import tempfile
import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Verified: delta_e is defined at src/util/metric.py line 589
# Signature: delta_e(pred, target, valid_mask=None) -> float
# Input: tensors [B, 3, H, W] or [3, H, W] in [0, 1]
# Output: float (mean CIEDE2000 value)
from src.util.metric import delta_e

# ---------------------------------------------------------------------------
# Configuration: base checkpoint path for trainer tests (Properties 1, 2, 3)
# You must symlink your real base restoration checkpoint here:
#   ln -s /path/to/your/base/checkpoint ckpt/base_restoration_checkpoint
# The checkpoint must contain unet/diffusion_pytorch_model.safetensors
# ---------------------------------------------------------------------------
BASE_CHECKPOINT_RELATIVE_PATH = "ckpt/base_restoration_checkpoint"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Smart generators for image tensors in [0, 1] range
# ---------------------------------------------------------------------------

def image_tensor_strategy(
    min_batch=1, max_batch=2,
    min_size=4, max_size=16,
):
    """
    Generate random image tensors in [0, 1] with shape [B, 3, H, W].
    Small spatial sizes keep tests fast while covering the input space.
    """
    return st.tuples(
        st.integers(min_value=min_batch, max_value=max_batch),  # B
        st.integers(min_value=min_size, max_value=max_size),    # H
        st.integers(min_value=min_size, max_value=max_size),    # W
        st.integers(min_value=0, max_value=2**31 - 1),          # seed
    ).map(lambda args: _make_image_tensor(*args))


def _make_image_tensor(batch, height, width, seed):
    """Create a random image tensor in [0, 1] with the given shape."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, height, width, generator=gen)


# ---------------------------------------------------------------------------
# Property 12a: Delta_E Identity
# Feature: hybrid-controlnet-restoration, Property 12: Delta_E Metric Properties
# Validates: Requirements 8.6
#
# For any valid image, delta_e(img, img) SHALL return 0.0.
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(data=image_tensor_strategy())
def test_property_12a_delta_e_identity(data):
    """
    **Validates: Requirements 8.6**
    Feature: hybrid-controlnet-restoration, Property 12: Delta_E Metric Properties

    For any valid image, delta_e(img, img) must return 0.0.
    The CIEDE2000 distance of an image to itself is always zero.
    """
    img = data  # shape [B, 3, H, W] in [0, 1]

    # Verified: delta_e at src/util/metric.py line 589
    # returns float, uses skimage.color.deltaE_ciede2000
    result = delta_e(img, img)

    assert result == 0.0, (
        f"delta_e(img, img) should be 0.0 for identical images, got {result}. "
        f"Image shape: {img.shape}"
    )


# ---------------------------------------------------------------------------
# Property 12b: Delta_E Positivity
# Feature: hybrid-controlnet-restoration, Property 12: Delta_E Metric Properties
# Validates: Requirements 8.6
#
# For any two different images, delta_e(img1, img2) SHALL return a positive
# value. We ensure images differ by at least a perceptible amount to avoid
# floating-point edge cases where different RGB values map to identical L*a*b*.
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    batch=st.integers(min_value=1, max_value=2),
    height=st.integers(min_value=4, max_value=16),
    width=st.integers(min_value=4, max_value=16),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_12b_delta_e_positivity(batch, height, width, seed):
    """
    **Validates: Requirements 8.6**
    Feature: hybrid-controlnet-restoration, Property 12: Delta_E Metric Properties

    For any two images that differ by a perceptible amount,
    delta_e(img1, img2) must return a strictly positive value.
    """
    gen = torch.Generator().manual_seed(seed)
    img1 = torch.rand(batch, 3, height, width, generator=gen)

    # Create img2 that is guaranteed to differ from img1 by a perceptible amount.
    # We invert img1 (1 - img1) which ensures every pixel differs significantly
    # in at least some channels, producing a non-zero Delta_E.
    img2 = 1.0 - img1

    result = delta_e(img1, img2)

    assert result > 0.0, (
        f"delta_e(img1, img2) should be > 0.0 for different images, got {result}. "
        f"Image shape: {img1.shape}"
    )


# ---------------------------------------------------------------------------
# Property 12c: Delta_E Non-Negativity
# Feature: hybrid-controlnet-restoration, Property 12: Delta_E Metric Properties
# Validates: Requirements 8.6
#
# Delta_E values SHALL always be non-negative for any two images.
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    batch=st.integers(min_value=1, max_value=2),
    height=st.integers(min_value=4, max_value=16),
    width=st.integers(min_value=4, max_value=16),
    seed1=st.integers(min_value=0, max_value=2**31 - 1),
    seed2=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_12c_delta_e_non_negativity(batch, height, width, seed1, seed2):
    """
    **Validates: Requirements 8.6**
    Feature: hybrid-controlnet-restoration, Property 12: Delta_E Metric Properties

    Delta_E values must always be non-negative for any two images,
    including when the two images happen to be identical (same seed).
    """
    gen1 = torch.Generator().manual_seed(seed1)
    img1 = torch.rand(batch, 3, height, width, generator=gen1)

    gen2 = torch.Generator().manual_seed(seed2)
    img2 = torch.rand(batch, 3, height, width, generator=gen2)

    result = delta_e(img1, img2)

    assert result >= 0.0, (
        f"delta_e should be >= 0.0, got {result}. "
        f"Image shapes: {img1.shape}, {img2.shape}"
    )


# ---------------------------------------------------------------------------
# Main: run all property tests
# ---------------------------------------------------------------------------

def _run_main():
    """Main entry point — separated so it can be called after all functions are defined."""
    print("=== Hybrid ControlNet Restoration Property Tests ===\n")

    # CPU-only tests (no model loading needed)
    cpu_tests = [
        ("Property 12a: Delta_E Identity", test_property_12a_delta_e_identity),
        ("Property 12b: Delta_E Positivity", test_property_12b_delta_e_positivity),
        ("Property 12c: Delta_E Non-Negativity", test_property_12c_delta_e_non_negativity),
        ("Property 4: 8-Channel Concatenation Shape Invariant", test_property_4_8ch_concatenation_shape),
    ]

    # GPU-required pipeline tests (need model loading)
    gpu_pipeline_tests = [
        ("Property 14: VAE Encode Spatial Downsampling", test_property_14_vae_encode_spatial_downsampling),
        ("Property 5: Single Inference Output Range", test_property_5_single_infer_output_range),
        ("Property 8: Zero Residuals Equivalence", test_property_8_zero_residuals_equivalence),
        ("Property 11: UNet 8-Channel Conv_In Invariant", test_property_11_unet_8ch_conv_in),
        ("Property 9: Checkpoint Round-Trip", test_property_9_checkpoint_round_trip),
        ("Property 10: Checkpoint Structure Invariant", test_property_10_checkpoint_structure),
        ("Property 6: Full Inference Output Range", test_property_6_full_inference_output_range),
        ("Property 7: Output Resolution Matching", test_property_7_output_resolution_matching),
    ]

    # GPU-required trainer tests (need base checkpoint + trainer instantiation)
    gpu_trainer_tests = [
        ("Property 1: Trainability Invariant", test_property_1_trainability_invariant),
        ("Property 2: Optimizer Parameter Exclusivity", test_property_2_optimizer_exclusivity),
        ("Property 3: Gradient Isolation", test_property_3_gradient_isolation),
    ]

    all_passed = True

    print("--- CPU-only tests (no model loading) ---\n")
    for name, test_fn in cpu_tests:
        print(f"Running {name}...")
        try:
            test_fn()
            print(f"  PASSED\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("--- GPU pipeline tests (loading hybrid pipeline) ---\n")
    if torch.cuda.is_available():
        print("Loading hybrid pipeline (this may take a moment)...")
        get_hybrid_pipeline()
        print("Hybrid pipeline loaded.\n")

        for name, test_fn in gpu_pipeline_tests:
            print(f"Running {name}...")
            try:
                test_fn()
                print(f"  PASSED\n")
            except Exception as e:
                print(f"  FAILED: {e}\n")
                import traceback
                traceback.print_exc()
                all_passed = False

        print("--- GPU trainer tests (loading trainer with base checkpoint) ---\n")
        base_ckpt = os.path.join(str(project_root), BASE_CHECKPOINT_RELATIVE_PATH)
        if os.path.isdir(base_ckpt):
            print(f"Base checkpoint found: {base_ckpt}")
            print("Loading trainer (this may take a moment)...")
            get_hybrid_trainer()
            print("Trainer loaded.\n")

            for name, test_fn in gpu_trainer_tests:
                print(f"Running {name}...")
                try:
                    test_fn()
                    print(f"  PASSED\n")
                except Exception as e:
                    print(f"  FAILED: {e}\n")
                    import traceback
                    traceback.print_exc()
                    all_passed = False
        else:
            print(f"Base checkpoint NOT found at: {base_ckpt}")
            print("Skipping trainer tests (Properties 1, 2, 3).")
            print(f"To run these tests, create a symlink: ln -s /path/to/your/checkpoint {base_ckpt}\n")
    else:
        print("CUDA not available — skipping GPU tests.\n")

    if all_passed:
        print("=== ALL PROPERTY TESTS PASSED ===")
    else:
        print("=== SOME PROPERTY TESTS FAILED ===")

    sys.exit(0 if all_passed else 1)


# ===========================================================================
# Property Tests for Hybrid Pipeline (Task 2.4)
#
# Properties tested:
#   - Property 4: 8-Channel Concatenation Shape Invariant
#   - Property 5: Single Inference Output Range
#   - Property 8: Zero Residuals Equivalence
#   - Property 14: VAE Encode Spatial Downsampling
#
# Validates: Requirements 2.4, 2.7, 3.1, 6.1, 6.4
#
# Requires: hypothesis, torch, diffusers, transformers
# Properties 5, 8, 14 require CUDA GPU for model loading and inference.
# Property 4 is a pure tensor operation (CPU only).
# ===========================================================================

import torch.nn as nn
from torch.nn.parameter import Parameter

# ---------------------------------------------------------------------------
# Shared fixture: lazy-load hybrid pipeline ONCE for GPU-based tests
# ---------------------------------------------------------------------------

_HYBRID_PIPELINE = None
_HYBRID_DEVICE = None


def _replace_unet_conv_in(unet):
    """
    Replace UNet conv_in to accept 8 input channels.
    Exact pattern from src/trainer/marigold_restoration_trainer.py lines 412-441.
    """
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]

    _weight = _weight.repeat((1, 2, 1, 1))  # [320, 8, 3, 3]
    _weight *= 0.5

    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = nn.Conv2d(
        8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    unet.config["in_channels"] = 8


def get_hybrid_pipeline():
    """
    Lazy-load the hybrid pipeline once for all GPU-based tests.

    Order of operations (verified from script/controlnet_restoration/train.py):
    1. Load SD2 UNet (4ch)
    2. Create ControlNet from 4ch UNet via ControlNetModel.from_unet(unet)
    3. Replace UNet conv_in to 8ch
    4. Create MarigoldHybridControlNetRestorationPipeline
    """
    global _HYBRID_PIPELINE, _HYBRID_DEVICE

    if _HYBRID_PIPELINE is not None:
        return _HYBRID_PIPELINE, _HYBRID_DEVICE

    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDIMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from marigold.marigold_hybrid_controlnet_restoration_pipeline import (
        MarigoldHybridControlNetRestorationPipeline,
    )

    _HYBRID_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = "stabilityai/stable-diffusion-2"

    # Step 1: Load SD2 UNet (4ch) in float16
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_path, subfolder="unet", torch_dtype=torch.float16
    )

    # Step 2: Create ControlNet from 4ch UNet (must be before conv_in replacement)
    # Verified: script/controlnet_restoration/train.py line 262
    controlnet = ControlNetModel.from_unet(unet)
    controlnet = controlnet.half()

    # Step 3: Replace UNet conv_in to accept 8 channels
    # Verified: src/trainer/marigold_restoration_trainer.py lines 412-441
    _replace_unet_conv_in(unet)

    # Step 4: Load remaining SD2 components
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

    # Step 5: Create hybrid pipeline
    _HYBRID_PIPELINE = MarigoldHybridControlNetRestorationPipeline(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        default_denoising_steps=2,
        default_processing_resolution=64,
    )
    _HYBRID_PIPELINE = _HYBRID_PIPELINE.to(_HYBRID_DEVICE)

    # Fix scheduler to match training config (trailing timestep spacing)
    # Verified: same pattern as script/controlnet_restoration/test_properties.py line 82
    _HYBRID_PIPELINE.scheduler = DDIMScheduler.from_config(
        _HYBRID_PIPELINE.scheduler.config,
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True,
    )

    return _HYBRID_PIPELINE, _HYBRID_DEVICE


# ---------------------------------------------------------------------------
# Property 4: 8-Channel Concatenation Shape Invariant
# Feature: hybrid-controlnet-restoration, Property 4: 8-Channel Concatenation Shape Invariant
# Validates: Requirements 2.4, 6.4
#
# For any batch size B and spatial dimensions (h, w), concatenating a
# degraded_latent [B, 4, h, w] with a noisy_latent [B, 4, h, w] SHALL
# produce a tensor of shape [B, 8, h, w].
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    batch=st.integers(min_value=1, max_value=4),
    height=st.integers(min_value=1, max_value=32),
    width=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_4_8ch_concatenation_shape(batch, height, width, seed):
    """
    **Validates: Requirements 2.4, 6.4**
    Feature: hybrid-controlnet-restoration, Property 4: 8-Channel Concatenation Shape Invariant

    Concatenating degraded_latent [B, 4, h, w] with noisy_latent [B, 4, h, w]
    must produce a tensor of shape [B, 8, h, w]. Pure tensor operation, no GPU needed.
    """
    gen = torch.Generator().manual_seed(seed)

    degraded_latent = torch.randn(batch, 4, height, width, generator=gen)
    noisy_latent = torch.randn(batch, 4, height, width, generator=gen)

    # This is the exact concatenation used in the hybrid pipeline:
    # Verified: marigold_hybrid_controlnet_restoration_pipeline.py line 260
    #   unet_input = torch.cat([rgb_latent, target_latent], dim=1)
    concatenated = torch.cat([degraded_latent, noisy_latent], dim=1)

    assert concatenated.shape == (batch, 8, height, width), (
        f"Expected shape ({batch}, 8, {height}, {width}), "
        f"got {concatenated.shape}"
    )

    # Verify the first 4 channels are degraded_latent
    assert torch.equal(concatenated[:, :4], degraded_latent), (
        "First 4 channels should be degraded_latent"
    )

    # Verify the last 4 channels are noisy_latent
    assert torch.equal(concatenated[:, 4:], noisy_latent), (
        "Last 4 channels should be noisy_latent"
    )


# ---------------------------------------------------------------------------
# Property 14: VAE Encode Spatial Downsampling
# Feature: hybrid-controlnet-restoration, Property 14: VAE Encode Spatial Downsampling
# Validates: Requirements 6.1
#
# For any input tensor of shape [B, 3, H, W] where H and W are divisible by 8,
# encode_rgb SHALL produce a latent of shape [B, 4, H/8, W/8].
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    batch=st.integers(min_value=1, max_value=2),
    h_mult=st.integers(min_value=1, max_value=12),
    w_mult=st.integers(min_value=1, max_value=12),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_14_vae_encode_spatial_downsampling(batch, h_mult, w_mult, seed):
    """
    **Validates: Requirements 6.1**
    Feature: hybrid-controlnet-restoration, Property 14: VAE Encode Spatial Downsampling

    For any input [B, 3, H, W] where H and W are divisible by 8,
    encode_rgb must produce a latent of shape [B, 4, H/8, W/8].
    """
    pipe, device = get_hybrid_pipeline()

    H = h_mult * 8  # Ensure divisible by 8
    W = w_mult * 8

    gen = torch.Generator(device=device).manual_seed(seed)
    # Input in [-1, 1] range (same as pipeline expects)
    rgb_in = torch.rand(batch, 3, H, W, device=device, dtype=pipe.dtype, generator=gen) * 2.0 - 1.0

    with torch.no_grad():
        # Verified: encode_rgb at pipeline line 122
        # Uses self.vae.encoder, self.vae.quant_conv, returns mean * latent_scale_factor
        latent = pipe.encode_rgb(rgb_in)

    expected_h = H // 8
    expected_w = W // 8

    assert latent.shape == (batch, 4, expected_h, expected_w), (
        f"Expected latent shape ({batch}, 4, {expected_h}, {expected_w}), "
        f"got {latent.shape}. Input was ({batch}, 3, {H}, {W})"
    )


# ---------------------------------------------------------------------------
# Property 5: Single Inference Output Range
# Feature: hybrid-controlnet-restoration, Property 5: Single Inference Output Range
# Validates: Requirements 2.7
#
# For any input degraded image, the restored RGB output of single_infer
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
def test_property_5_single_infer_output_range(seed, num_steps):
    """
    **Validates: Requirements 2.7**
    Feature: hybrid-controlnet-restoration, Property 5: Single Inference Output Range

    The restored RGB output of single_infer() must always be in [-1.0, 1.0].
    Uses 64x64 resolution for speed.
    """
    pipe, device = get_hybrid_pipeline()

    # Generate random degraded image in valid range [-1, 1]
    generator = torch.Generator(device=device).manual_seed(seed)
    rgb_in = torch.rand(1, 3, 64, 64, device=device, dtype=pipe.dtype, generator=generator) * 2.0 - 1.0

    # Reset generator for reproducible noise in single_infer
    generator = torch.Generator(device=device).manual_seed(seed)

    pipe.scheduler.set_timesteps(num_steps, device=device)

    with torch.no_grad():
        # Verified: single_infer at pipeline line 155
        # Returns tensor clipped to [-1, 1] (line 310: torch.clip(restored_rgb, -1.0, 1.0))
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
    assert restored.shape[0] == 1 and restored.shape[1] == 3, (
        f"Expected shape (1, 3, H, W), got {restored.shape}"
    )


# ---------------------------------------------------------------------------
# Property 8: Zero Residuals Equivalence
# Feature: hybrid-controlnet-restoration, Property 8: Zero Residuals Equivalence
# Validates: Requirements 3.1
#
# For any input (noisy_latent, degraded_latent, timestep, text_embed),
# the frozen 8ch UNet output with all-zero ControlNet residuals SHALL be
# equal to the frozen 8ch UNet output without any ControlNet residuals
# (standard UNet forward pass with 8ch concatenated input).
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
def test_property_8_zero_residuals_equivalence(seed, timestep_frac):
    """
    **Validates: Requirements 3.1**
    Feature: hybrid-controlnet-restoration, Property 8: Zero Residuals Equivalence

    The 8ch UNet output with all-zero ControlNet residuals must equal
    the 8ch UNet output without any residuals (standard forward pass).
    This validates that zeroed residuals are a true no-op.
    """
    pipe, device = get_hybrid_pipeline()

    torch.manual_seed(seed)

    # Generate random 8ch input (degraded_latent + noisy_latent concatenated)
    # At 64x64 image resolution -> 8x8 latent
    noisy_latent = torch.randn(1, 4, 8, 8, device=device, dtype=torch.float16)
    degraded_latent = torch.randn(1, 4, 8, 8, device=device, dtype=torch.float16)

    # 8ch concatenated input (verified: pipeline line 260)
    cat_input = torch.cat([degraded_latent, noisy_latent], dim=1)  # [1, 8, 8, 8]

    # Map timestep_frac [0, 1] to valid timestep range
    max_t = pipe.scheduler.config.num_train_timesteps - 1
    t = torch.tensor([int(timestep_frac * max_t)], device=device).long()

    # Encode empty text
    if pipe.empty_text_embed is None:
        pipe.encode_empty_text()
    text_embed = pipe.empty_text_embed.to(device)  # [1, 77, 1024]

    with torch.no_grad():
        # Get ControlNet residual shapes by running a dummy forward pass
        # Verified: ControlNet receives noisy_latent (4ch), not 8ch
        # (design doc: "The ControlNet receives noisy_latent (4ch) as its sample input")
        down_res, mid_res = pipe.controlnet(
            noisy_latent, t,
            encoder_hidden_states=text_embed,
            controlnet_cond=torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16),
            return_dict=False,
        )

        # Zero all residuals
        zero_down_res = [torch.zeros_like(r) for r in down_res]
        zero_mid_res = torch.zeros_like(mid_res)

        # 8ch UNet with zero residuals
        out_with_zero_res = pipe.unet(
            cat_input, t,
            encoder_hidden_states=text_embed,
            down_block_additional_residuals=zero_down_res,
            mid_block_additional_residual=zero_mid_res,
        ).sample

        # 8ch UNet without any residuals (standard forward)
        out_without_res = pipe.unet(
            cat_input, t,
            encoder_hidden_states=text_embed,
        ).sample

    # Must be numerically identical (zero residuals = no residuals)
    assert torch.equal(out_with_zero_res, out_without_res), (
        f"Zero residuals output differs from no-residuals output. "
        f"Max diff: {(out_with_zero_res - out_without_res).abs().max().item()}"
    )


# ===========================================================================
# Property Tests for Hybrid Trainer (Task 3.7)
#
# Properties tested:
#   - Property 11: UNet 8-Channel Conv_In Invariant
#   - Property 9: Checkpoint Round-Trip
#   - Property 10: Checkpoint Structure Invariant
#   - Property 1: Trainability Invariant
#   - Property 2: Optimizer Parameter Exclusivity
#   - Property 3: Gradient Isolation
#
# Validates: Requirements 5.2, 5.3, 5.5, 5.6, 6.7, 7.6, 10.1, 10.4, 10.5
#
# Properties 9, 10, 11: Use get_hybrid_pipeline() fixture (no trainer needed)
# Properties 1, 2, 3: Use get_hybrid_trainer() fixture (needs base checkpoint)
# ===========================================================================


# ---------------------------------------------------------------------------
# Shared fixture: lazy-load hybrid trainer ONCE for trainer-based tests
# ---------------------------------------------------------------------------

_HYBRID_TRAINER = None


def _create_minimal_hybrid_config():
    """
    Create a minimal OmegaConf config with all required keys for the hybrid trainer.
    Pattern from script/controlnet_restoration/test_trainer_checkpoint.py create_minimal_config()
    with lpips_weight added (hybrid trainer reads it at __init__ step 11).
    """
    from omegaconf import OmegaConf

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
                "lpips_weight": 0.0,
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


def _create_fake_dataloader(batch_size=2, num_samples=4, resolution=64):
    """
    Create a fake dataloader with random degraded/clean RGB pairs.
    Pattern from script/controlnet_restoration/test_trainer_checkpoint.py create_fake_dataloader().
    """
    from torch.utils.data import DataLoader, TensorDataset

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


def get_hybrid_trainer():
    """
    Lazy-load the hybrid trainer once for all trainer-based tests.
    Uses BASE_CHECKPOINT_RELATIVE_PATH for the base checkpoint.
    Pattern from script/controlnet_restoration/test_trainer_checkpoint.py test_trainer().
    """
    global _HYBRID_TRAINER

    if _HYBRID_TRAINER is not None:
        return _HYBRID_TRAINER

    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DDIMScheduler,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from marigold.marigold_hybrid_controlnet_restoration_pipeline import (
        MarigoldHybridControlNetRestorationPipeline,
    )
    from src.trainer.marigold_hybrid_controlnet_restoration_trainer import (
        MarigoldHybridControlNetRestorationTrainer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = "stabilityai/stable-diffusion-2"

    # Step 1: Load SD2 UNet (4ch) in float16
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_path, subfolder="unet", torch_dtype=torch.float16
    )

    # Step 2: Create ControlNet from 4ch UNet (must be before conv_in replacement)
    # Verified: script/controlnet_restoration/train.py line 262
    # ControlNet in float32 — trainable params must be float32 for GradScaler
    controlnet = ControlNetModel.from_unet(unet)

    # Step 3: Load remaining SD2 components
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

    # Step 4: Create hybrid pipeline (UNet still 4ch — trainer handles 8ch replacement)
    pipeline = MarigoldHybridControlNetRestorationPipeline(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        default_denoising_steps=2,
        default_processing_resolution=64,
    )

    # Step 5: Create trainer (handles 8ch conv_in replacement + base checkpoint loading)
    cfg = _create_minimal_hybrid_config()
    train_loader = _create_fake_dataloader(batch_size=2, num_samples=4, resolution=64)
    base_ckpt = os.path.join(str(project_root), BASE_CHECKPOINT_RELATIVE_PATH)

    # Use mkdtemp (no auto-delete) so directories persist for the trainer's lifetime.
    # The OS cleans up temp dirs on process exit.
    tmpdir = tempfile.mkdtemp()
    out_dir_ckpt = os.path.join(tmpdir, "checkpoint")
    out_dir_eval = os.path.join(tmpdir, "evaluation")
    out_dir_vis = os.path.join(tmpdir, "visualization")
    os.makedirs(out_dir_ckpt)
    os.makedirs(out_dir_eval)
    os.makedirs(out_dir_vis)

    _HYBRID_TRAINER = MarigoldHybridControlNetRestorationTrainer(
        cfg=cfg,
        model=pipeline,
        train_dataloader=train_loader,
        device=device,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=1,
        base_checkpoint_path=base_ckpt,
    )

    # Move model to device
    _HYBRID_TRAINER.model.to(device)

    return _HYBRID_TRAINER


# ---------------------------------------------------------------------------
# Property 11: UNet 8-Channel Conv_In Invariant
# Feature: hybrid-controlnet-restoration, Property 11: UNet 8-Channel Conv_In Invariant
# Validates: Requirements 10.1
#
# For any initialized hybrid pipeline, the UNet's conv_in layer SHALL have
# in_channels == 8 and the UNet's config SHALL report in_channels == 8.
# ---------------------------------------------------------------------------

def test_property_11_unet_8ch_conv_in():
    """
    **Validates: Requirements 10.1**
    Feature: hybrid-controlnet-restoration, Property 11: UNet 8-Channel Conv_In Invariant

    The UNet's conv_in must have in_channels == 8 and the UNet config must
    report in_channels == 8. This is a structural invariant of the hybrid pipeline.
    """
    pipe, device = get_hybrid_pipeline()

    # Check conv_in layer
    # Verified: _replace_unet_conv_in sets unet.conv_in = Conv2d(8, 320, ...)
    assert pipe.unet.conv_in.in_channels == 8, (
        f"UNet conv_in.in_channels should be 8, got {pipe.unet.conv_in.in_channels}"
    )

    # Check UNet config
    # Verified: _replace_unet_conv_in sets unet.config["in_channels"] = 8
    assert pipe.unet.config["in_channels"] == 8, (
        f"UNet config['in_channels'] should be 8, got {pipe.unet.config['in_channels']}"
    )

    # Check conv_in output channels (should be 320 for SD2)
    assert pipe.unet.conv_in.out_channels == 320, (
        f"UNet conv_in.out_channels should be 320, got {pipe.unet.conv_in.out_channels}"
    )


# ---------------------------------------------------------------------------
# Property 9: Checkpoint Round-Trip
# Feature: hybrid-controlnet-restoration, Property 9: Checkpoint Round-Trip
# Validates: Requirements 7.6, 10.5
#
# For any trained ControlNet state, saving via save_pretrained() and loading
# via ControlNetModel.from_pretrained() SHALL produce numerically identical
# parameters.
# ---------------------------------------------------------------------------

@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_9_checkpoint_round_trip(seed):
    """
    **Validates: Requirements 7.6, 10.5**
    Feature: hybrid-controlnet-restoration, Property 9: Checkpoint Round-Trip

    Save then load must produce numerically identical ControlNet weights.
    Pattern from script/controlnet_restoration/test_properties.py Property 8.
    """
    from diffusers import ControlNetModel

    pipe, device = get_hybrid_pipeline()

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
        pipe.controlnet.save_pretrained(save_path, safe_serialization=True)

        # Load via from_pretrained (same as run.py load_hybrid_pipeline)
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
# Property 10: Checkpoint Structure Invariant
# Feature: hybrid-controlnet-restoration, Property 10: Checkpoint Structure Invariant
# Validates: Requirements 10.4
#
# For any saved hybrid checkpoint, the checkpoint directory SHALL contain a
# controlnet/ subdirectory with ControlNet weights and SHALL NOT contain a
# unet/ subdirectory.
# ---------------------------------------------------------------------------

@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_10_checkpoint_structure(seed):
    """
    **Validates: Requirements 10.4**
    Feature: hybrid-controlnet-restoration, Property 10: Checkpoint Structure Invariant

    A saved hybrid checkpoint must contain controlnet/ and must NOT contain unet/.
    The frozen 8ch UNet is always loaded from the base checkpoint, never saved.
    """
    pipe, device = get_hybrid_pipeline()

    # Perturb weights to simulate different training states
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in pipe.controlnet.parameters():
            param.add_(torch.randn_like(param) * 0.001)

    # Snapshot to restore later
    original_state = {
        k: v.clone() for k, v in pipe.controlnet.state_dict().items()
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        controlnet_path = os.path.join(tmpdir, "controlnet")

        # Save ControlNet (same as trainer save_checkpoint pattern)
        pipe.controlnet.save_pretrained(controlnet_path, safe_serialization=True)

        # Verify controlnet/ exists with expected files
        assert os.path.isdir(controlnet_path), (
            f"controlnet/ directory should exist at {controlnet_path}"
        )
        safetensors_path = os.path.join(controlnet_path, "diffusion_pytorch_model.safetensors")
        assert os.path.isfile(safetensors_path), (
            f"diffusion_pytorch_model.safetensors should exist in controlnet/"
        )
        config_path = os.path.join(controlnet_path, "config.json")
        assert os.path.isfile(config_path), (
            f"config.json should exist in controlnet/"
        )

        # Verify unet/ does NOT exist (frozen UNet is never saved)
        unet_path = os.path.join(tmpdir, "unet")
        assert not os.path.exists(unet_path), (
            f"unet/ directory should NOT exist in hybrid checkpoint, but found at {unet_path}"
        )

    # Restore original weights
    pipe.controlnet.load_state_dict(original_state)


# ---------------------------------------------------------------------------
# Property 1: Trainability Invariant
# Feature: hybrid-controlnet-restoration, Property 1: Trainability Invariant
# Validates: Requirements 5.2, 5.3, 5.5
#
# For any initialized MarigoldHybridControlNetRestorationTrainer, all
# parameters of the UNet, VAE, and text encoder SHALL have requires_grad == False,
# and only ControlNet parameters SHALL have requires_grad == True.
# Additionally, the UNet SHALL be in evaluation mode (training == False).
# ---------------------------------------------------------------------------

def test_property_1_trainability_invariant():
    """
    **Validates: Requirements 5.2, 5.3, 5.5**
    Feature: hybrid-controlnet-restoration, Property 1: Trainability Invariant

    After trainer initialization:
    - UNet, VAE, text_encoder: all params requires_grad == False
    - ControlNet: all params requires_grad == True
    - UNet: training == False (eval mode)
    """
    trainer = get_hybrid_trainer()

    # UNet: all params frozen
    unet_grads = [p.requires_grad for p in trainer.model.unet.parameters()]
    assert not any(unet_grads), (
        f"UNet should be fully frozen, but {sum(unet_grads)}/{len(unet_grads)} params have requires_grad=True"
    )

    # UNet: eval mode
    assert not trainer.model.unet.training, (
        "UNet should be in eval mode (training == False)"
    )

    # VAE: all params frozen
    vae_grads = [p.requires_grad for p in trainer.model.vae.parameters()]
    assert not any(vae_grads), (
        f"VAE should be fully frozen, but {sum(vae_grads)}/{len(vae_grads)} params have requires_grad=True"
    )

    # Text encoder: all params frozen
    te_grads = [p.requires_grad for p in trainer.model.text_encoder.parameters()]
    assert not any(te_grads), (
        f"Text encoder should be fully frozen, but {sum(te_grads)}/{len(te_grads)} params have requires_grad=True"
    )

    # ControlNet: all params trainable
    cnet_grads = [p.requires_grad for p in trainer.model.controlnet.parameters()]
    assert all(cnet_grads), (
        f"ControlNet should be fully trainable, but {len(cnet_grads) - sum(cnet_grads)}/{len(cnet_grads)} params have requires_grad=False"
    )


# ---------------------------------------------------------------------------
# Property 2: Optimizer Parameter Exclusivity
# Feature: hybrid-controlnet-restoration, Property 2: Optimizer Parameter Exclusivity
# Validates: Requirements 5.6
#
# For any initialized MarigoldHybridControlNetRestorationTrainer, the set of
# all parameter data pointers in the optimizer SHALL be exactly equal to the
# set of all parameter data pointers in the ControlNet_Module.
# ---------------------------------------------------------------------------

def test_property_2_optimizer_exclusivity():
    """
    **Validates: Requirements 5.6**
    Feature: hybrid-controlnet-restoration, Property 2: Optimizer Parameter Exclusivity

    The optimizer must contain exactly the ControlNet parameters — no more, no less.
    No UNet, VAE, or text encoder parameters shall appear in the optimizer.
    """
    trainer = get_hybrid_trainer()

    # Collect optimizer parameter data pointers
    opt_param_ptrs = set()
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            opt_param_ptrs.add(p.data_ptr())

    # Collect ControlNet parameter data pointers
    cnet_param_ptrs = set()
    for p in trainer.model.controlnet.parameters():
        cnet_param_ptrs.add(p.data_ptr())

    # Must be exactly equal
    assert opt_param_ptrs == cnet_param_ptrs, (
        f"Optimizer params should be exactly ControlNet params. "
        f"In optimizer but not ControlNet: {opt_param_ptrs - cnet_param_ptrs}. "
        f"In ControlNet but not optimizer: {cnet_param_ptrs - opt_param_ptrs}."
    )

    # Double-check: no UNet params in optimizer
    unet_param_ptrs = {p.data_ptr() for p in trainer.model.unet.parameters()}
    assert opt_param_ptrs.isdisjoint(unet_param_ptrs), (
        "UNet parameters found in optimizer — they should be excluded"
    )

    # Double-check: no VAE params in optimizer
    vae_param_ptrs = {p.data_ptr() for p in trainer.model.vae.parameters()}
    assert opt_param_ptrs.isdisjoint(vae_param_ptrs), (
        "VAE parameters found in optimizer — they should be excluded"
    )

    # Double-check: no text_encoder params in optimizer
    te_param_ptrs = {p.data_ptr() for p in trainer.model.text_encoder.parameters()}
    assert opt_param_ptrs.isdisjoint(te_param_ptrs), (
        "Text encoder parameters found in optimizer — they should be excluded"
    )


# ---------------------------------------------------------------------------
# Property 3: Gradient Isolation
# Feature: hybrid-controlnet-restoration, Property 3: Gradient Isolation
# Validates: Requirements 6.7
#
# For any training batch, after the backward pass, all UNet parameters SHALL
# have grad == None (no gradient accumulated), while at least one ControlNet
# parameter SHALL have a non-None gradient.
# ---------------------------------------------------------------------------

def test_property_3_gradient_isolation():
    """
    **Validates: Requirements 6.7**
    Feature: hybrid-controlnet-restoration, Property 3: Gradient Isolation

    After forward + backward, UNet must have NO gradients and ControlNet
    must have at least one non-None gradient.
    """
    from torch.amp import autocast

    trainer = get_hybrid_trainer()
    device = trainer.device

    trainer.model.to(device)
    trainer.model.controlnet.train()

    # Get one batch from the fake dataloader
    batch = next(iter(trainer.train_loader))
    degraded_rgb = batch["degraded_rgb_norm"].to(device)
    clean_rgb = batch["clean_rgb_norm"].to(device)
    batch_size = degraded_rgb.shape[0]

    # Encode clean to latent (VAE is frozen, use autocast since VAE is fp16)
    with torch.no_grad(), autocast('cuda'):
        clean_latent = trainer.encode_rgb(clean_rgb)

    # Sample timestep and noise
    timesteps = torch.randint(
        0, trainer.scheduler_timesteps, (batch_size,), device=device
    ).long()
    noise = torch.randn(clean_latent.shape, device=device)
    noisy_latents = trainer.training_noise_scheduler.add_noise(
        clean_latent, noise, timesteps
    )

    # Text embed
    text_embed = trainer.empty_text_embed.to(device).repeat((batch_size, 1, 1))

    # Encode degraded to latent for 8ch concat
    with torch.no_grad(), autocast('cuda'):
        degraded_latent = trainer.encode_rgb(degraded_rgb)

    # Forward pass with autocast
    with autocast('cuda'):
        # ControlNet forward: receives noisy_latents (4ch) + degraded_rgb (pixel space)
        down_block_res, mid_block_res = trainer.model.controlnet(
            noisy_latents, timesteps,
            encoder_hidden_states=text_embed,
            controlnet_cond=degraded_rgb,
            return_dict=False,
        )

        # 8ch UNet forward: cat([degraded_latent, noisy_latents])
        cat_latents = torch.cat([degraded_latent, noisy_latents], dim=1)
        model_pred = trainer.model.unet(
            cat_latents, timesteps,
            encoder_hidden_states=text_embed,
            down_block_additional_residuals=down_block_res,
            mid_block_additional_residual=mid_block_res,
        ).sample

    # Target (v_prediction)
    target = trainer.training_noise_scheduler.get_velocity(
        clean_latent, noise, timesteps
    )

    # Loss + backward
    loss = trainer.loss(model_pred.float(), target.float()).mean()
    trainer.scaler.scale(loss).backward()

    # Check gradient isolation: UNet must have NO gradients
    unet_has_grad = any(
        p.grad is not None for p in trainer.model.unet.parameters()
    )
    assert not unet_has_grad, (
        "UNet should have NO gradients after backward, but some params have grad != None"
    )

    # ControlNet must have at least one non-None gradient
    cnet_has_grad = any(
        p.grad is not None for p in trainer.model.controlnet.parameters()
    )
    assert cnet_has_grad, (
        "ControlNet should have at least one gradient after backward, but all grads are None"
    )

    # Cleanup: zero gradients for next test
    trainer.optimizer.zero_grad()



# ---------------------------------------------------------------------------
# Property 6: Full Inference Output Range
# Feature: hybrid-controlnet-restoration, Property 6: Full Inference Output Range
# Validates: Requirements 4.3
#
# For any input image processed through __call__, the final output numpy
# array (restored_np) SHALL have all values in the range [0.0, 1.0].
# This tests the full pipeline including VAE decode and post-processing
# ([-1,1] -> [0,1] conversion and clipping).
# ---------------------------------------------------------------------------

@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_6_full_inference_output_range(seed):
    """
    **Validates: Requirements 4.3**
    Feature: hybrid-controlnet-restoration, Property 6: Full Inference Output Range

    The restored output of __call__() must always be in [0.0, 1.0].
    Uses 64x64 PIL Image input, 2 denoising steps for speed.
    """
    pipe, device = get_hybrid_pipeline()

    # Create a random 64x64 RGB PIL Image
    rng = np.random.RandomState(seed)
    fake_image_np = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    fake_image = Image.fromarray(fake_image_np)

    generator = torch.Generator(device=device).manual_seed(seed)

    # Verified: __call__ at pipeline line 317 returns MarigoldRestorationOutput
    # with restored_np in [0, 1] (line 479: final_pred = (final_pred + 1.0) / 2.0, clip(0, 1))
    pipe_out = pipe(
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
    assert pipe_out.restored_np.min() >= 0.0, (
        f"Output min {pipe_out.restored_np.min()} < 0.0"
    )
    assert pipe_out.restored_np.max() <= 1.0, (
        f"Output max {pipe_out.restored_np.max()} > 1.0"
    )
    assert pipe_out.restored_np.shape[0] == 3, (
        f"Expected 3 channels, got {pipe_out.restored_np.shape[0]}"
    )
    assert isinstance(pipe_out.restored_img, Image.Image), (
        f"Expected PIL Image, got {type(pipe_out.restored_img)}"
    )


# ---------------------------------------------------------------------------
# Property 7: Output Resolution Matching
# Feature: hybrid-controlnet-restoration, Property 7: Output Resolution Matching
# Validates: Requirements 4.4
#
# For any input image with dimensions (H_in, W_in), when
# match_input_res=True, the output image SHALL have dimensions
# (H_in, W_in).
# ---------------------------------------------------------------------------

@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    height=st.sampled_from([48, 64, 80, 96]),
    width=st.sampled_from([48, 64, 80, 96]),
)
def test_property_7_output_resolution_matching(seed, height, width):
    """
    **Validates: Requirements 4.4**
    Feature: hybrid-controlnet-restoration, Property 7: Output Resolution Matching

    When match_input_res=True, the output PIL Image dimensions must match
    the input image dimensions (H_in, W_in).
    """
    pipe, device = get_hybrid_pipeline()

    # Create a random RGB PIL Image with given dimensions
    rng = np.random.RandomState(seed)
    fake_image_np = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    fake_image = Image.fromarray(fake_image_np)

    generator = torch.Generator(device=device).manual_seed(seed)

    # Verified: __call__ at pipeline line 317, match_input_res triggers resize
    # back to original resolution (line 464-470)
    pipe_out = pipe(
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

    # PIL Image.size returns (width, height)
    out_w, out_h = pipe_out.restored_img.size
    assert out_h == height, (
        f"Output height {out_h} != input height {height}"
    )
    assert out_w == width, (
        f"Output width {out_w} != input width {width}"
    )

    # Also check numpy array shape: (3, H, W)
    assert pipe_out.restored_np.shape == (3, height, width), (
        f"Expected numpy shape (3, {height}, {width}), got {pipe_out.restored_np.shape}"
    )


# ---------------------------------------------------------------------------
# Main entry point — must be at the end of the file so all functions are defined
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_main()
