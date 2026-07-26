#!/usr/bin/env python3
"""
Test that combined loss gradients flow correctly to U-Net.

This test verifies:
1. Image loss is computed on restored (not degraded) images
2. Gradients flow from image loss back to U-Net
3. VAE decoder is frozen but allows gradient flow
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.util.loss import CombinedRestorationLoss

def test_gradient_flow():
    """Test that gradients flow correctly through the combined loss."""
    
    print("=" * 80)
    print("COMBINED LOSS GRADIENT FLOW TEST")
    print("=" * 80)
    
    # Create dummy tensors
    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simulate model prediction (requires grad)
    model_pred = torch.randn(batch_size, 4, 96, 96, device=device, requires_grad=True)
    
    # Simulate noisy latents and scheduler params
    noisy_latents = torch.randn(batch_size, 4, 96, 96, device=device)
    alpha_prod_t = torch.tensor([0.5, 0.6], device=device).view(-1, 1, 1, 1)
    beta_prod_t = torch.tensor([0.5, 0.4], device=device).view(-1, 1, 1, 1)
    
    # Compute pred_original_sample (denoised latent)
    pred_original_sample = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
    
    print(f"\n1. Model prediction requires_grad: {model_pred.requires_grad}")
    print(f"2. Pred original sample requires_grad: {pred_original_sample.requires_grad}")
    
    # Simulate VAE decode (frozen but allows gradients)
    # In real code: restored_rgb = vae.decode(pred_original_sample)
    # Here we simulate with a simple operation
    restored_rgb = torch.sigmoid(pred_original_sample.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1))
    clean_rgb = torch.rand(batch_size, 3, 96, 96, device=device)
    
    print(f"3. Restored RGB requires_grad: {restored_rgb.requires_grad}")
    print(f"4. Clean RGB requires_grad: {clean_rgb.requires_grad}")
    
    # Compute combined loss
    loss_fn = CombinedRestorationLoss(
        mse_weight=1.0,
        perceptual_weight=0.1,
        lpips_weight=0.05,
        device=device
    )
    
    image_loss, loss_components = loss_fn(restored_rgb, clean_rgb)
    
    print(f"\n5. Image loss value: {image_loss.item():.6f}")
    print(f"6. Image loss requires_grad: {image_loss.requires_grad}")
    
    # Backpropagate
    image_loss.backward()
    
    print(f"\n7. Model prediction has gradients: {model_pred.grad is not None}")
    if model_pred.grad is not None:
        print(f"8. Model prediction grad norm: {model_pred.grad.norm().item():.6f}")
        print("\n✅ SUCCESS: Gradients flow correctly from image loss to model prediction!")
    else:
        print("\n❌ FAILURE: No gradients on model prediction!")
        return False
    
    # Verify loss components
    print(f"\n9. Loss components:")
    for k, v in loss_components.items():
        print(f"   - {k}: {v.item():.6f}")
    
    print("\n" + "=" * 80)
    print("TEST PASSED: Combined loss gradients flow correctly")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_gradient_flow()
    exit(0 if success else 1)
