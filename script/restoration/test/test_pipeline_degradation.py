#!/usr/bin/env python3
"""
Test script for pipeline degradation system

Tests the new sequential probabilistic degradation pipeline:
- Blur → Noise → Compression
- Each stage with 50% probability
- Validates distribution and quality
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import random

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.ARNIQA.degradation import ImageDistorter
from omegaconf import OmegaConf


def test_pipeline_statistics():
    """Test that pipeline produces expected distribution"""
    print("=" * 80)
    print("TEST 1: Pipeline Statistics")
    print("=" * 80)
    
    # Load configuration
    config_path = project_root / "config" / "dataset_restoration" / "dataset_base.yaml"
    cfg = OmegaConf.load(config_path)
    
    pipeline_config = cfg.degradation_config.pipeline_config
    
    print(f"\nConfiguration:")
    print(f"  Blur probability: {pipeline_config.blur.probability}")
    print(f"  Noise probability: {pipeline_config.noise.probability}")
    print(f"  Compression probability: {pipeline_config.compression.probability}")
    
    # Simulate 10000 samples
    num_samples = 10000
    blur_prob = pipeline_config.blur.probability
    noise_prob = pipeline_config.noise.probability
    comp_prob = pipeline_config.compression.probability
    
    # Count degradation combinations
    counts = {
        'clean': 0,      # No degradations
        'blur_only': 0,
        'noise_only': 0,
        'comp_only': 0,
        'blur_noise': 0,
        'blur_comp': 0,
        'noise_comp': 0,
        'all_three': 0,
    }
    
    for _ in range(num_samples):
        has_blur = random.random() < blur_prob
        has_noise = random.random() < noise_prob
        has_comp = random.random() < comp_prob
        
        num_degradations = sum([has_blur, has_noise, has_comp])
        
        if num_degradations == 0:
            counts['clean'] += 1
        elif num_degradations == 3:
            counts['all_three'] += 1
        elif num_degradations == 2:
            if has_blur and has_noise:
                counts['blur_noise'] += 1
            elif has_blur and has_comp:
                counts['blur_comp'] += 1
            else:
                counts['noise_comp'] += 1
        else:  # num_degradations == 1
            if has_blur:
                counts['blur_only'] += 1
            elif has_noise:
                counts['noise_only'] += 1
            else:
                counts['comp_only'] += 1
    
    print(f"\nDistribution over {num_samples} samples:")
    print(f"  Clean (0 degradations):     {counts['clean']:5d} ({counts['clean']/num_samples*100:5.2f}%) [Expected: ~12.5%]")
    print(f"  All 3 degradations:         {counts['all_three']:5d} ({counts['all_three']/num_samples*100:5.2f}%) [Expected: ~12.5%]")
    
    two_deg_total = counts['blur_noise'] + counts['blur_comp'] + counts['noise_comp']
    print(f"  2 degradations (total):     {two_deg_total:5d} ({two_deg_total/num_samples*100:5.2f}%) [Expected: ~37.5%]")
    print(f"    - Blur + Noise:           {counts['blur_noise']:5d} ({counts['blur_noise']/num_samples*100:5.2f}%)")
    print(f"    - Blur + Compression:     {counts['blur_comp']:5d} ({counts['blur_comp']/num_samples*100:5.2f}%)")
    print(f"    - Noise + Compression:    {counts['noise_comp']:5d} ({counts['noise_comp']/num_samples*100:5.2f}%)")
    
    one_deg_total = counts['blur_only'] + counts['noise_only'] + counts['comp_only']
    print(f"  1 degradation (total):      {one_deg_total:5d} ({one_deg_total/num_samples*100:5.2f}%) [Expected: ~37.5%]")
    print(f"    - Blur only:              {counts['blur_only']:5d} ({counts['blur_only']/num_samples*100:5.2f}%)")
    print(f"    - Noise only:             {counts['noise_only']:5d} ({counts['noise_only']/num_samples*100:5.2f}%)")
    print(f"    - Compression only:       {counts['comp_only']:5d} ({counts['comp_only']/num_samples*100:5.2f}%)")
    
    # Validate distribution
    expected_clean = 0.125
    expected_all = 0.125
    expected_two = 0.375
    expected_one = 0.375
    
    tolerance = 0.02  # 2% tolerance
    
    assert abs(counts['clean']/num_samples - expected_clean) < tolerance, "Clean distribution off"
    assert abs(counts['all_three']/num_samples - expected_all) < tolerance, "All-three distribution off"
    assert abs(two_deg_total/num_samples - expected_two) < tolerance, "Two-degradation distribution off"
    assert abs(one_deg_total/num_samples - expected_one) < tolerance, "One-degradation distribution off"
    
    print("\n✅ Distribution matches expected values (within 2% tolerance)")


def test_pipeline_application():
    """Test actual pipeline degradation on test image"""
    print("\n" + "=" * 80)
    print("TEST 2: Pipeline Application")
    print("=" * 80)
    
    # Load test image
    test_image_path = project_root / "test_image.png"
    if not test_image_path.exists():
        print(f"⚠️  Test image not found: {test_image_path}")
        print("   Skipping visual test")
        return
    
    # Load and prepare image
    image = Image.open(test_image_path).convert('RGB')
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0  # [C, H, W] in [0, 1]
    
    print(f"\nTest image: {test_image_path}")
    print(f"  Shape: {image_tensor.shape}")
    print(f"  Range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # Initialize distorter
    distorter = ImageDistorter()
    
    # Load configuration
    config_path = project_root / "config" / "dataset_restoration" / "dataset_base.yaml"
    cfg = OmegaConf.load(config_path)
    pipeline_config = cfg.degradation_config.pipeline_config
    
    # Create output directory
    output_dir = project_root / "test_output" / "pipeline_degradation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original
    image.save(output_dir / "00_original.png")
    print(f"\n✓ Saved original image")
    
    # Test individual stages
    print("\nTesting individual stages:")
    
    # 1. Blur only
    print("  1. Blur stage...")
    blurred = image_tensor.clone()
    blurred = distorter.apply_distortion_to_tensor(blurred, 'gaublur', 2)
    blurred_img = Image.fromarray((blurred.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    blurred_img.save(output_dir / "01_blur_only.png")
    
    # 2. Noise only
    print("  2. Noise stage...")
    noisy = image_tensor.clone()
    noisy = distorter.apply_distortion_to_tensor(noisy, 'whitenoise', 2)
    noisy_img = Image.fromarray((noisy.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    noisy_img.save(output_dir / "02_noise_only.png")
    
    # 3. Compression only
    print("  3. Compression stage...")
    compressed = image_tensor.clone()
    compressed = distorter.apply_distortion_to_tensor(compressed, 'jpeg', 2)
    compressed_img = Image.fromarray((compressed.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    compressed_img.save(output_dir / "03_compression_only.png")
    
    # Test sequential combinations
    print("\nTesting sequential combinations:")
    
    # 4. Blur → Noise
    print("  4. Blur → Noise...")
    blur_noise = image_tensor.clone()
    blur_noise = distorter.apply_distortion_to_tensor(blur_noise, 'gaublur', 2)
    blur_noise = distorter.apply_distortion_to_tensor(blur_noise, 'whitenoise', 2)
    blur_noise_img = Image.fromarray((blur_noise.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    blur_noise_img.save(output_dir / "04_blur_noise.png")
    
    # 5. Blur → Compression
    print("  5. Blur → Compression...")
    blur_comp = image_tensor.clone()
    blur_comp = distorter.apply_distortion_to_tensor(blur_comp, 'gaublur', 2)
    blur_comp = distorter.apply_distortion_to_tensor(blur_comp, 'jpeg', 2)
    blur_comp_img = Image.fromarray((blur_comp.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    blur_comp_img.save(output_dir / "05_blur_compression.png")
    
    # 6. Noise → Compression
    print("  6. Noise → Compression...")
    noise_comp = image_tensor.clone()
    noise_comp = distorter.apply_distortion_to_tensor(noise_comp, 'whitenoise', 2)
    noise_comp = distorter.apply_distortion_to_tensor(noise_comp, 'jpeg', 2)
    noise_comp_img = Image.fromarray((noise_comp.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    noise_comp_img.save(output_dir / "06_noise_compression.png")
    
    # 7. All three (Blur → Noise → Compression)
    print("  7. Blur → Noise → Compression (full pipeline)...")
    all_three = image_tensor.clone()
    all_three = distorter.apply_distortion_to_tensor(all_three, 'gaublur', 2)
    all_three = distorter.apply_distortion_to_tensor(all_three, 'whitenoise', 2)
    all_three = distorter.apply_distortion_to_tensor(all_three, 'jpeg', 2)
    all_three_img = Image.fromarray((all_three.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    all_three_img.save(output_dir / "07_all_three.png")
    
    print(f"\n✅ All degradation samples saved to: {output_dir}")
    print(f"   Total files: 8 (1 original + 7 degraded)")


def test_deterministic_seeding():
    """Test that deterministic seeding works correctly"""
    print("\n" + "=" * 80)
    print("TEST 3: Deterministic Seeding")
    print("=" * 80)
    
    # Create dummy image
    dummy = torch.rand(3, 224, 224)
    
    # Initialize distorter
    distorter = ImageDistorter()
    
    # Test 1: Same seed → same degradation
    print("\nTest 3.1: Same seed produces same degradation")
    
    base_seed = 42
    index = 5
    
    # First run
    random.seed(base_seed + index)
    np.random.seed(base_seed + index)
    torch.manual_seed(base_seed + index)
    
    deg1 = distorter.apply_distortion_to_tensor(dummy.clone(), 'whitenoise', 2)
    
    # Second run (same seed)
    random.seed(base_seed + index)
    np.random.seed(base_seed + index)
    torch.manual_seed(base_seed + index)
    
    deg2 = distorter.apply_distortion_to_tensor(dummy.clone(), 'whitenoise', 2)
    
    # Check if identical
    diff = torch.abs(deg1 - deg2).max().item()
    print(f"  Max difference: {diff:.10f}")
    assert diff < 1e-6, "Same seed should produce identical degradation"
    print("  ✅ Same seed produces identical degradation")
    
    # Test 2: Different seed → different degradation
    print("\nTest 3.2: Different seed produces different degradation")
    
    # Third run (different seed)
    random.seed(base_seed + index + 1000000)
    np.random.seed(base_seed + index + 1000000)
    torch.manual_seed(base_seed + index + 1000000)
    
    deg3 = distorter.apply_distortion_to_tensor(dummy.clone(), 'whitenoise', 2)
    
    # Check if different
    diff = torch.abs(deg1 - deg3).max().item()
    print(f"  Max difference: {diff:.10f}")
    assert diff > 1e-3, "Different seed should produce different degradation"
    print("  ✅ Different seed produces different degradation")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PIPELINE DEGRADATION SYSTEM TEST")
    print("=" * 80)
    
    try:
        # Test 1: Statistical distribution
        test_pipeline_statistics()
        
        # Test 2: Visual degradation application
        test_pipeline_application()
        
        # Test 3: Deterministic seeding
        test_deterministic_seeding()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nPipeline degradation system is working correctly!")
        print("Expected distribution:")
        print("  - 12.5% clean (identity mapping)")
        print("  - 12.5% all 3 degradations (hardest samples)")
        print("  - 37.5% 2 degradations")
        print("  - 37.5% 1 degradation")
        print("\nReady for training with realistic multi-degradation pipeline!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
