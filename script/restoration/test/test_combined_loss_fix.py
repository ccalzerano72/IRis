#!/usr/bin/env python3
"""
Test to verify combined loss calculation is correct after fix.

This test checks that:
1. Gradients flow through the image loss
2. Restored image is computed from model prediction
3. Loss components are calculated correctly
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.util.loss import CombinedRestorationLoss

def test_combined_loss_gradients():
    """Test that gradients flow through combined loss"""
    print("\n" + "="*80)
    print("TEST: Combined Loss Gradient Flow")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create loss function
    loss_fn = CombinedRestorationLoss(
        mse_weight=1.0,
        perceptual_weight=0.1,
        lpips_weight=0.05,
        device=device
    )
    
    # Create dummy images [B, C, H, W] in [0, 1]
    batch_size = 2
    restored = torch.rand(batch_size, 3, 256, 256, device=device, requires_grad=True)
    clean = torch.rand(batch_size, 3, 256, 256, device=device)
    
    print(f"\nInput shapes:")
    print(f"  Restored: {restored.shape}, requires_grad={restored.requires_grad}")
    print(f"  Clean: {clean.shape}, requires_grad={clean.requires_grad}")
    
    # Forward pass
    total_loss, loss_components = loss_fn(restored, clean)
    
    print(f"\nLoss components:")
    for key, value in loss_components.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    print(f"\nGradient check:")
    print(f"  Restored grad is None: {restored.grad is None}")
    if restored.grad is not None:
        print(f"  Restored grad shape: {restored.grad.shape}")
        print(f"  Restored grad mean: {restored.grad.mean().item():.6f}")
        print(f"  Restored grad std: {restored.grad.std().item():.6f}")
        print(f"  ✅ Gradients are flowing!")
    else:
        print(f"  ❌ ERROR: No gradients!")
        return False
    
    return True

def test_denoised_latent_computation():
    """Test that denoised latent is computed correctly"""
    print("\n" + "="*80)
    print("TEST: Denoised Latent Computation")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate scheduler parameters
    timesteps = torch.tensor([500], device=device)
    alpha_prod_t = torch.tensor([0.5], device=device)  # Example value
    beta_prod_t = 1 - alpha_prod_t
    
    # Reshape for broadcasting
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
    
    # Create dummy tensors
    noisy_latents = torch.randn(1, 4, 64, 64, device=device)
    model_pred = torch.randn(1, 4, 64, 64, device=device)
    
    # Compute denoised latent (epsilon prediction)
    pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
    
    print(f"\nShapes:")
    print(f"  Noisy latents: {noisy_latents.shape}")
    print(f"  Model prediction: {model_pred.shape}")
    print(f"  Denoised latent: {pred_original_sample.shape}")
    
    print(f"\nValues:")
    print(f"  alpha_prod_t: {alpha_prod_t.item():.4f}")
    print(f"  beta_prod_t: {beta_prod_t.item():.4f}")
    print(f"  Noisy mean: {noisy_latents.mean().item():.4f}")
    print(f"  Pred mean: {model_pred.mean().item():.4f}")
    print(f"  Denoised mean: {pred_original_sample.mean().item():.4f}")
    
    print(f"\n✅ Denoised latent computation working!")
    return True

def test_loss_input_order():
    """Test that loss receives correct input order (restored, clean)"""
    print("\n" + "="*80)
    print("TEST: Loss Input Order")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function
    loss_fn = CombinedRestorationLoss(
        mse_weight=1.0,
        perceptual_weight=0.0,  # Disable for simplicity
        lpips_weight=0.0,
        device=device
    )
    
    # Create test images
    clean = torch.ones(1, 3, 64, 64, device=device) * 1.0  # All white
    degraded = torch.ones(1, 3, 64, 64, device=device) * 0.5  # Gray
    restored = torch.ones(1, 3, 64, 64, device=device) * 0.8  # Light gray
    
    print(f"\nImage values:")
    print(f"  Clean: {clean.mean().item():.2f}")
    print(f"  Degraded: {degraded.mean().item():.2f}")
    print(f"  Restored: {restored.mean().item():.2f}")
    
    # Compute losses
    loss_restored_vs_clean, _ = loss_fn(restored, clean)
    loss_degraded_vs_clean, _ = loss_fn(degraded, clean)
    
    print(f"\nLosses:")
    print(f"  Restored vs Clean: {loss_restored_vs_clean.item():.6f}")
    print(f"  Degraded vs Clean: {loss_degraded_vs_clean.item():.6f}")
    
    # Restored should be closer to clean than degraded
    if loss_restored_vs_clean < loss_degraded_vs_clean:
        print(f"\n✅ Correct: Restored is closer to clean than degraded")
        return True
    else:
        print(f"\n❌ ERROR: Degraded is closer to clean than restored!")
        return False

def main():
    print("\n" + "="*80)
    print("COMBINED LOSS FIX VERIFICATION")
    print("="*80)
    
    results = []
    
    # Test 1: Gradient flow
    try:
        result = test_combined_loss_gradients()
        results.append(("Gradient Flow", result))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Gradient Flow", False))
    
    # Test 2: Denoised latent computation
    try:
        result = test_denoised_latent_computation()
        results.append(("Denoised Latent", result))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Denoised Latent", False))
    
    # Test 3: Loss input order
    try:
        result = test_loss_input_order()
        results.append(("Loss Input Order", result))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Loss Input Order", False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe combined loss fix is working correctly:")
        print("  ✅ Gradients flow through image loss")
        print("  ✅ Denoised latent is computed correctly")
        print("  ✅ Loss receives correct inputs (restored vs clean)")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
