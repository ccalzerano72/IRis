#!/usr/bin/env python3
"""
Test script for ARNIQA integration validation.

This script tests:
1. ImageDistorter initialization
2. All 25 degradation types
3. Tensor format compatibility
4. GPU acceleration (if available)
5. File I/O operations
6. Marigold format conversion

Usage:
    python script/restoration/test_arniqa_integration.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path (go up 3 levels: test -> restoration -> script -> project_root)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ARNIQA.degradation import ImageDistorter


# Test image dimensions constants
TEST_IMAGE_WIDTH = 512
TEST_IMAGE_HEIGHT = 512
TEST_IMAGE_CHANNELS = 3


def test_initialization():
    """Test ImageDistorter initialization"""
    print("\n" + "="*70)
    print("TEST 1: ImageDistorter Initialization")
    print("="*70)
    
    try:
        distorter = ImageDistorter()
        print(f"✓ ImageDistorter initialized successfully")
        print(f"✓ Available degradations: {len(distorter.distortion_functions)}")
        print(f"✓ Degradation types: {list(distorter.distortion_functions.keys())[:5]}... (showing first 5)")
        return distorter, True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return None, False


def test_tensor_operations(distorter):
    """Test degradation on tensors"""
    print("\n" + "="*70)
    print("TEST 2: Tensor Operations")
    print("="*70)
    
    # Create dummy tensor
    dummy = torch.rand(TEST_IMAGE_CHANNELS, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)  # [C, H, W] in [0, 1]
    print(f"✓ Created dummy tensor: {dummy.shape}")
    print(f"  - Min value: {dummy.min().item():.4f}")
    print(f"  - Max value: {dummy.max().item():.4f}")
    print(f"  - Mean value: {dummy.mean().item():.4f}")
    
    # Test each degradation type
    print(f"\nTesting all {len(distorter.distortion_functions)} degradation types:")
    print("-" * 70)
    
    failed_degradations = []
    
    for i, deg_type in enumerate(distorter.distortion_functions.keys(), 1):
        try:
            # Apply degradation at medium level
            degraded = distorter.apply_distortion_to_tensor(
                dummy.clone(), deg_type, level=2
            )
            
            # Validate output
            assert degraded.shape == dummy.shape, f"Shape mismatch: {degraded.shape} vs {dummy.shape}"
            assert degraded.min() >= 0, f"Min value out of range: {degraded.min()}"
            assert degraded.max() <= 1, f"Max value out of range: {degraded.max()}"
            
            print(f"✓ [{i:2d}/25] {deg_type:20s} - OK (min={degraded.min():.4f}, max={degraded.max():.4f})")
            
        except Exception as e:
            print(f"✗ [{i:2d}/25] {deg_type:20s} - FAILED: {e}")
            failed_degradations.append((deg_type, str(e)))
    
    print("-" * 70)
    
    if failed_degradations:
        print(f"\n⚠ {len(failed_degradations)} degradation(s) failed:")
        for deg_type, error in failed_degradations:
            print(f"  - {deg_type}: {error}")
        return False
    else:
        print(f"\n✓ All 25 degradations passed!")
        return True


def test_severity_levels(distorter):
    """Test all severity levels for a sample degradation"""
    print("\n" + "="*70)
    print("TEST 3: Severity Levels")
    print("="*70)
    
    dummy = torch.rand(TEST_IMAGE_CHANNELS, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)
    deg_type = 'whitenoise'
    
    print(f"Testing severity levels for '{deg_type}':")
    print("-" * 70)
    
    for level in range(5):
        try:
            degraded = distorter.apply_distortion_to_tensor(
                dummy.clone(), deg_type, level=level
            )
            
            # Calculate difference from original
            diff = torch.abs(degraded - dummy).mean().item()
            
            print(f"✓ Level {level}: mean_diff={diff:.6f}, "
                  f"min={degraded.min():.4f}, max={degraded.max():.4f}")
            
        except Exception as e:
            print(f"✗ Level {level}: FAILED - {e}")
            return False
    
    print("-" * 70)
    print("✓ All severity levels working correctly")
    return True


def test_marigold_compatibility(distorter):
    """Test conversion to Marigold format"""
    print("\n" + "="*70)
    print("TEST 4: Marigold Format Compatibility")
    print("="*70)
    
    # Create dummy tensor
    dummy = torch.rand(TEST_IMAGE_CHANNELS, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)  # [C, H, W] in [0, 1]
    
    # Apply degradation
    degraded = distorter.apply_distortion_to_tensor(
        dummy.clone(), 'whitenoise', level=2
    )
    
    print(f"ARNIQA format:")
    print(f"  - Shape: {degraded.shape}")
    print(f"  - Range: [{degraded.min():.4f}, {degraded.max():.4f}]")
    print(f"  - Mean: {degraded.mean():.4f}")
    
    # Convert to Marigold format [-1, 1]
    degraded_norm = degraded * 2.0 - 1.0
    clean_norm = dummy * 2.0 - 1.0
    
    print(f"\nMarigold format (after conversion):")
    print(f"  - Degraded range: [{degraded_norm.min():.4f}, {degraded_norm.max():.4f}]")
    print(f"  - Clean range: [{clean_norm.min():.4f}, {clean_norm.max():.4f}]")
    print(f"  - Degraded mean: {degraded_norm.mean():.4f}")
    
    # Validate
    assert -1.0 <= degraded_norm.min() <= 1.0, "Degraded min out of range"
    assert -1.0 <= degraded_norm.max() <= 1.0, "Degraded max out of range"
    assert -1.0 <= clean_norm.min() <= 1.0, "Clean min out of range"
    assert -1.0 <= clean_norm.max() <= 1.0, "Clean max out of range"
    
    print("\n✓ Marigold format conversion successful")
    return True


def test_gpu_acceleration(distorter):
    """Test GPU acceleration if available"""
    print("\n" + "="*70)
    print("TEST 5: GPU Acceleration")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available - skipping GPU test")
        return True
    
    device = torch.device('cuda')
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Note: ARNIQA degradations work on CPU tensors
    # In real training, degradations are applied in dataset (CPU) before GPU transfer
    print("\n💡 Note: ARNIQA degradations are designed for CPU tensors")
    print("   During training:")
    print("   1. Dataset loads clean image (CPU)")
    print("   2. Apply degradation (CPU)")
    print("   3. DataLoader moves to GPU")
    print("   This is the standard and efficient approach.")
    
    # Test the actual training workflow
    try:
        # Simulate dataset behavior
        dummy_cpu = torch.rand(TEST_IMAGE_CHANNELS, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)  # CPU tensor
        print(f"\n✓ Created tensor on CPU: {dummy_cpu.device}")
        
        # Apply degradation on CPU (as in dataset)
        degraded_cpu = distorter.apply_distortion_to_tensor(
            dummy_cpu, 'whitenoise', level=2
        )
        print(f"✓ Degradation applied on CPU: {degraded_cpu.device}")
        
        # Move to GPU (as DataLoader does)
        degraded_gpu = degraded_cpu.to(device)
        print(f"✓ Moved to GPU: {degraded_gpu.device}")
        
        # Verify
        assert degraded_gpu.device.type == 'cuda', "Failed to move to GPU"
        assert degraded_gpu.shape == dummy_cpu.shape, "Shape mismatch"
        
        print("\n✓ Training workflow (CPU degradation → GPU training) working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        return False


def test_file_operations(distorter):
    """Test file I/O operations"""
    print("\n" + "="*70)
    print("TEST 6: File I/O Operations")
    print("="*70)
    
    # Create temporary test image
    test_dir = Path("temp_test_arniqa")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Create and save test image
        test_image = np.random.randint(0, 256, (TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, TEST_IMAGE_CHANNELS), dtype=np.uint8)
        test_image_path = test_dir / "test_image.png"
        Image.fromarray(test_image).save(test_image_path)
        print(f"✓ Created test image: {test_image_path}")
        
        # Load image
        loaded_tensor = distorter.load_image(test_image_path)
        print(f"✓ Loaded image as tensor: {loaded_tensor.shape}")
        print(f"  - Range: [{loaded_tensor.min():.4f}, {loaded_tensor.max():.4f}]")
        
        # Apply degradation and save
        degraded_tensor = distorter.apply_distortion_to_tensor(
            loaded_tensor, 'whitenoise', level=2
        )
        
        output_path = test_dir / "test_degraded.png"
        saved_path = distorter.save_image(degraded_tensor, output_path)
        print(f"✓ Saved degraded image: {saved_path}")
        
        # Verify saved file exists
        assert Path(saved_path).exists(), "Saved file not found"
        
        # Load saved image to verify
        saved_image = Image.open(saved_path)
        print(f"✓ Verified saved image: {saved_image.size}")
        saved_image.close()
        
        print("\n✓ File I/O operations working correctly")
        return True
        
    except Exception as e:
        print(f"✗ File I/O test failed: {e}")
        return False
        
    finally:
        # Cleanup
        import shutil
        import time
        if test_dir.exists():
            # Small delay to ensure file handles are released on Windows
            time.sleep(0.1)
            try:
                shutil.rmtree(test_dir)
                print(f"✓ Cleaned up test directory")
            except PermissionError:
                print(f"⚠ Could not remove test directory (files may be in use)")
                print(f"  Please manually delete: {test_dir.absolute()}")


def test_batch_processing(distorter):
    """Test batch processing capability"""
    print("\n" + "="*70)
    print("TEST 7: Batch Processing")
    print("="*70)
    
    # Create batch of tensors
    batch_size = 4
    batch = torch.rand(batch_size, TEST_IMAGE_CHANNELS, TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH)
    print(f"✓ Created batch: {batch.shape}")
    
    try:
        # Process each image in batch
        degraded_batch = []
        for i in range(batch_size):
            degraded = distorter.apply_distortion_to_tensor(
                batch[i], 'whitenoise', level=2
            )
            degraded_batch.append(degraded)
        
        degraded_batch = torch.stack(degraded_batch)
        print(f"✓ Processed batch: {degraded_batch.shape}")
        
        # Validate
        assert degraded_batch.shape == batch.shape, "Batch shape mismatch"
        
        print("\n✓ Batch processing working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        return False


def test_real_image_degradation(distorter):
    """Test degradation on a real image with visual output"""
    print("\n" + "="*70)
    print("TEST 8: Real Image Degradation (Visual Test)")
    print("="*70)
    
    # Look for test image
    test_image_path = Path("test_image.png")
    
    if not test_image_path.exists():
        print("⚠ No test image found at 'test_image.png'")
        print("  To run this test:")
        print("  1. Place a PNG image named 'test_image.png' in the project root")
        print("  2. Re-run the test script")
        print("\n⊘ Skipping real image test")
        return True  # Not a failure, just skipped
    
    print(f"✓ Found test image: {test_image_path}")
    
    # Create output directory
    output_dir = Path("test_arniqa_output")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")
    
    try:
        # Load the real image
        image_tensor = distorter.load_image(test_image_path)
        print(f"✓ Loaded image: {image_tensor.shape}")
        print(f"  - Range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
        print(f"  - Mean: {image_tensor.mean():.4f}")
        
        # Save original (for comparison)
        original_output = output_dir / "00_original.png"
        distorter.save_image(image_tensor, original_output)
        print(f"✓ Saved original: {original_output}")
        
        # Test a selection of degradations (not all 125 to save time)
        test_degradations = [
            ('whitenoise', 2, 'White Noise (medium)'),
            ('jpeg', 2, 'JPEG Compression (medium)'),
            ('gaublur', 2, 'Gaussian Blur (medium)'),
            ('motionblur', 2, 'Motion Blur (medium)'),
            ('brighten', 2, 'Brighten (medium)'),
            ('darken', 2, 'Darken (medium)'),
            ('colorsat1', 2, 'Desaturate (medium)'),
            ('pixelate', 2, 'Pixelate (medium)'),
            ('highsharpen', 2, 'High Sharpen (medium)'),
        ]
        
        print(f"\nApplying {len(test_degradations)} sample degradations:")
        print("-" * 70)
        
        for i, (deg_type, level, description) in enumerate(test_degradations, 1):
            try:
                # Apply degradation
                degraded = distorter.apply_distortion_to_tensor(
                    image_tensor.clone(), deg_type, level
                )
                
                # Save degraded image
                output_filename = f"{i:02d}_{deg_type}_level{level}.png"
                output_path = output_dir / output_filename
                distorter.save_image(degraded, output_path)
                
                # Calculate difference
                diff = torch.abs(degraded - image_tensor).mean().item()
                
                print(f"✓ [{i:2d}/{len(test_degradations)}] {description:30s} "
                      f"(diff={diff:.4f}) → {output_filename}")
                
            except Exception as e:
                print(f"✗ [{i:2d}/{len(test_degradations)}] {description:30s} FAILED: {e}")
                return False
        
        print("-" * 70)
        
        # Test all severity levels for one degradation
        print(f"\nTesting all severity levels for 'whitenoise':")
        print("-" * 70)
        
        for level in range(5):
            try:
                degraded = distorter.apply_distortion_to_tensor(
                    image_tensor.clone(), 'whitenoise', level
                )
                
                output_filename = f"whitenoise_level{level}.png"
                output_path = output_dir / output_filename
                distorter.save_image(degraded, output_path)
                
                diff = torch.abs(degraded - image_tensor).mean().item()
                
                print(f"✓ Level {level}: diff={diff:.6f} → {output_filename}")
                
            except Exception as e:
                print(f"✗ Level {level}: FAILED - {e}")
                return False
        
        print("-" * 70)
        
        # Generate all 125 combinations (optional, commented out by default)
        print(f"\n💡 To generate all 125 degradation combinations, uncomment the code below")
        print(f"   or run: distorter.generate_all_distortions('{test_image_path}', '{output_dir}')")
        
        # Uncomment to generate all combinations:
        # print(f"\nGenerating all 125 degradation combinations...")
        # all_files = distorter.generate_all_distortions(
        #     str(test_image_path), 
        #     str(output_dir / "all_combinations")
        # )
        # print(f"✓ Generated {len(all_files)} images")
        
        print(f"\n✓ Real image degradation test completed successfully")
        print(f"✓ Output saved to: {output_dir.absolute()}")
        print(f"\n📁 Check the output directory to visually inspect degradations:")
        print(f"   {output_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("ARNIQA INTEGRATION TEST SUITE")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Initialize
    distorter, init_success = test_initialization()
    if not init_success:
        print("\n" + "="*70)
        print("FATAL: Initialization failed - aborting tests")
        print("="*70)
        return False
    
    # Run tests
    results = {
        "Tensor Operations": test_tensor_operations(distorter),
        "Severity Levels": test_severity_levels(distorter),
        "Marigold Compatibility": test_marigold_compatibility(distorter),
        "GPU Acceleration": test_gpu_acceleration(distorter),
        "File I/O": test_file_operations(distorter),
        "Batch Processing": test_batch_processing(distorter),
        "Real Image Degradation": test_real_image_degradation(distorter),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:10s} - {test_name}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! ARNIQA integration is working correctly.")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
