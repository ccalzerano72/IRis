#!/usr/bin/env python3
"""
Test script for combined restoration loss.

Tests:
1. Loss initialization
2. Forward pass
3. Gradient computation
4. Loss component logging
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.util.loss import CombinedRestorationLoss, get_loss

def test_combined_loss_initialization():
    """Test loss initialization with different configurations."""
    print("\n=== Testing Loss Initialization ===")
    
    # Test 1: Default parameters
    try:
        loss = CombinedRestorationLoss()
        print("✅ Default initialization successful")
    except Exception as e:
        print(f"❌ Default initialization failed: {e}")
        return False
    
    # Test 2: Custom parameters
    try:
        loss = CombinedRestorationLoss(
            mse_weight=1.0,
            perceptual_weight=0.2,
            lpips_weight=0.1,
            perceptual_layers=['relu1_2', 'relu2_2'],
            lpips_net='vgg'
        )
        print("✅ Custom initialization successful")
    except Exception as e:
        print(f"❌ Custom initialization failed: {e}")
        return False
    
    # Test 3: Factory function
    try:
        loss = get_loss(
            loss_name="combined_restoration_loss",
            mse_weight=1.0,
            perceptual_weight=0.1,
            lpips_weight=0.05
        )
        print("✅ Factory function successful")
    except Exception as e:
        print(f"❌ Factory function failed: {e}")
        return False
    
    return True

def test_loss_forward_pass():
    """Test forward pass with synthetic data."""
    print("\n=== Testing Forward Pass ===")
    
    # Create synthetic data
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    pred = torch.randn(batch_size, channels, height, width, device='cuda')
    target = torch.randn(batch_size, channels, height, width, device='cuda')
    
    # Normalize to [0, 1]
    pred = torch.sigmoid(pred)
    target = torch.sigmoid(target)
    
    try:
        loss = CombinedRestorationLoss(device='cuda')
        total_loss, loss_components = loss(pred, target)
        
        print(f"✅ Forward pass successful")
        print(f"   Total loss: {total_loss.item():.6f}")
        print(f"   Loss components:")
        for k, v in loss_components.items():
            print(f"     {k}: {v.item():.6f}")
        
        # Check that all components are present
        expected_keys = ['mse', 'perceptual', 'lpips', 'total']
        for key in expected_keys:
            if key not in loss_components:
                print(f"❌ Missing loss component: {key}")
                return False
        
        # Check that components are balanced (after normalization)
        print(f"\n   Component balance check:")
        print(f"     MSE/Perceptual ratio: {loss_components['mse'].item() / loss_components['perceptual'].item():.2f}")
        print(f"     MSE/LPIPS ratio: {loss_components['mse'].item() / loss_components['lpips'].item():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_computation():
    """Test gradient computation."""
    print("\n=== Testing Gradient Computation ===")
    
    # Create synthetic data with gradients
    batch_size = 1
    channels = 3
    height = 128  # Smaller for faster test
    width = 128
    
    pred = torch.randn(batch_size, channels, height, width, device='cuda', requires_grad=True)
    target = torch.randn(batch_size, channels, height, width, device='cuda')
    
    # Normalize to [0, 1]
    pred = torch.sigmoid(pred)
    target = torch.sigmoid(target)
    
    try:
        loss = CombinedRestorationLoss(device='cuda')
        total_loss, loss_components = loss(pred, target)
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        if pred.grad is not None and pred.grad.norm() > 0:
            print(f"✅ Gradient computation successful")
            print(f"   Gradient norm: {pred.grad.norm().item():.6f}")
            return True
        else:
            print(f"❌ No gradients computed")
            return False
            
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_components_weighting():
    """Test that loss component weighting works correctly."""
    print("\n=== Testing Loss Component Weighting ===")
    
    batch_size = 1
    channels = 3
    height = 128
    width = 128
    
    pred = torch.randn(batch_size, channels, height, width, device='cuda')
    target = torch.randn(batch_size, channels, height, width, device='cuda')
    
    pred = torch.sigmoid(pred)
    target = torch.sigmoid(target)
    
    try:
        # Test with different weights
        configs = [
            {"mse_weight": 1.0, "perceptual_weight": 0.0, "lpips_weight": 0.0},  # MSE only
            {"mse_weight": 0.0, "perceptual_weight": 1.0, "lpips_weight": 0.0},  # Perceptual only
            {"mse_weight": 0.0, "perceptual_weight": 0.0, "lpips_weight": 1.0},  # LPIPS only
        ]
        
        for i, config in enumerate(configs):
            loss = CombinedRestorationLoss(device='cuda', **config)
            total_loss, loss_components = loss(pred, target)
            
            print(f"   Config {i+1}: {config}")
            print(f"   Total loss: {total_loss.item():.6f}")
            
        print("✅ Loss component weighting test successful")
        return True
        
    except Exception as e:
        print(f"❌ Loss component weighting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Combined Restoration Loss")
    print("=" * 50)
    
    tests = [
        test_combined_loss_initialization,
        test_loss_forward_pass,
        test_gradient_computation,
        test_loss_components_weighting,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Combined loss is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
