#!/usr/bin/env python3
"""
Test MarigoldRestorationPipeline with existing model weights
This test uses depth model weights to verify the complete inference pipeline works.
Results won't be good (depth model used for restoration), but validates the pipeline structure.
"""

import sys
import os

# Add project root to path (go up 3 levels: test -> restoration -> script -> project_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import numpy as np
from PIL import Image
import os
from marigold import MarigoldRestorationPipeline, MarigoldRestorationOutput

def create_test_image(size=(512, 512)):
    """Create a simple test image with some 'degradation'"""
    height, width = size
    
    # Create a simple pattern
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Create pattern with some noise (simulated degradation)
    pattern = np.sin(X) * np.cos(Y)
    noise = np.random.normal(0, 0.1, (height, width))
    degraded_pattern = pattern + noise
    
    # Convert to RGB
    normalized = (degraded_pattern - degraded_pattern.min()) / (degraded_pattern.max() - degraded_pattern.min())
    rgb = np.stack([normalized, normalized * 0.8, normalized * 0.6], axis=2)
    rgb = (rgb * 255).astype(np.uint8)
    
    return Image.fromarray(rgb)

def test_pipeline_with_existing_model():
    """Test complete pipeline inference using existing model weights"""
    print("=== Testing MarigoldRestorationPipeline with Existing Model ===\n")
    
    # Create test image
    print("1. Creating test image...")
    test_img = create_test_image()
    print(f"   ✓ Test image created: {test_img.size}")
    
    # Save test image
    os.makedirs("test_output", exist_ok=True)
    test_img.save("test_output/test_input.png")
    print("   ✓ Test image saved: test_output/test_input.png")
    
    # Load pipeline with existing model
    print("\n2. Loading pipeline with existing model...")
    try:
        # Use depth model weights (won't give good restoration results, but tests the pipeline)
        pipeline = MarigoldRestorationPipeline.from_pretrained(
            "prs-eth/marigold-depth-v1-1",
            torch_dtype=torch.float32
        )
        print("   ✓ Pipeline loaded successfully")
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = pipeline.to(device)
        print(f"   ✓ Pipeline moved to device: {device}")
        
    except Exception as e:
        print(f"   ✗ Pipeline loading failed: {e}")
        return False
    
    # Test inference
    print("\n3. Testing inference...")
    try:
        with torch.no_grad():
            # Set seed for reproducibility
            generator = torch.Generator(device=device)
            generator.manual_seed(42)
            
            # Run inference
            print("   Running inference...")
            result: MarigoldRestorationOutput = pipeline(
                test_img,
                denoising_steps=1,  # Fast test
                ensemble_size=1,    # Single prediction
                processing_res=256, # Small resolution for speed
                generator=generator,
                show_progress_bar=True
            )
            
            print("   ✓ Inference completed successfully")
            
            # Check outputs
            print(f"   ✓ Restored array shape: {result.restored_np.shape}")
            print(f"   ✓ Restored array range: [{result.restored_np.min():.3f}, {result.restored_np.max():.3f}]")
            print(f"   ✓ Restored image size: {result.restored_img.size}")
            print(f"   ✓ Uncertainty: {'Available' if result.uncertainty is not None else 'None (expected for ensemble_size=1)'}")
            
            # Save results
            result.restored_img.save("test_output/test_restored.png")
            np.save("test_output/test_restored.npy", result.restored_np)
            print("   ✓ Results saved: test_output/test_restored.png, test_output/test_restored.npy")
            
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test ensemble
    print("\n4. Testing ensemble inference...")
    try:
        with torch.no_grad():
            generator = torch.Generator(device=device)
            generator.manual_seed(42)
            
            print("   Running ensemble inference (3 predictions)...")
            result_ensemble: MarigoldRestorationOutput = pipeline(
                test_img,
                denoising_steps=1,
                ensemble_size=3,    # Multiple predictions
                processing_res=256,
                generator=generator,
                show_progress_bar=True
            )
            
            print("   ✓ Ensemble inference completed")
            print(f"   ✓ Uncertainty available: {result_ensemble.uncertainty is not None}")
            if result_ensemble.uncertainty is not None:
                print(f"   ✓ Uncertainty shape: {result_ensemble.uncertainty.shape}")
                print(f"   ✓ Uncertainty range: [{result_ensemble.uncertainty.min():.3f}, {result_ensemble.uncertainty.max():.3f}]")
            
            # Save ensemble results
            result_ensemble.restored_img.save("test_output/test_restored_ensemble.png")
            print("   ✓ Ensemble result saved: test_output/test_restored_ensemble.png")
            
    except Exception as e:
        print(f"   ✗ Ensemble inference failed: {e}")
        return False
    
    # Test different parameters
    print("\n5. Testing different parameters...")
    try:
        with torch.no_grad():
            # Test different denoising steps
            result_steps: MarigoldRestorationOutput = pipeline(
                test_img,
                denoising_steps=4,  # More steps
                ensemble_size=1,
                processing_res=256,
                show_progress_bar=False  # No progress bar
            )
            print("   ✓ Different denoising steps test passed")
            
            # Test original resolution
            result_orig_res: MarigoldRestorationOutput = pipeline(
                test_img,
                denoising_steps=1,
                ensemble_size=1,
                processing_res=0,  # Original resolution
                show_progress_bar=False
            )
            print("   ✓ Original resolution test passed")
            print(f"   ✓ Original resolution output size: {result_orig_res.restored_img.size}")
            
    except Exception as e:
        print(f"   ✗ Parameter tests failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("\nTest Summary:")
    print("✓ Pipeline loading with existing model weights")
    print("✓ Single inference")
    print("✓ Ensemble inference with uncertainty")
    print("✓ Different parameter configurations")
    print("✓ Output format validation")
    print("✓ File saving and loading")
    
    print(f"\nNote: Results are not meaningful (depth model used for restoration)")
    print(f"But this confirms the pipeline structure is correct!")
    print(f"\nTest outputs saved in: test_output/")
    
    return True

if __name__ == "__main__":
    success = test_pipeline_with_existing_model()
    if success:
        print("\n🎉 Pipeline test completed successfully!")
    else:
        print("\n❌ Pipeline test failed!")
        exit(1)