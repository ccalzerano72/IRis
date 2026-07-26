#!/usr/bin/env python3
"""
Test script for restoration metrics implementation
"""

import sys
import os

# Add project root to path (go up 3 levels: test -> restoration -> script -> project_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_restoration_metrics():
    """Test PSNR, SSIM, and LPIPS metrics"""
    
    print("Testing Restoration Metrics...")
    
    try:
        from src.util.metric import psnr, ssim, lpips_alex
        print("✓ Successfully imported restoration metrics")
    except ImportError as e:
        print(f"✗ Failed to import metrics: {e}")
        return False
    
    # Create test tensors
    batch_size, channels, height, width = 1, 3, 64, 64
    
    # Perfect match (should give best scores)
    pred_perfect = torch.rand(batch_size, channels, height, width)
    target = pred_perfect.clone()
    
    # Noisy version (should give worse scores)
    pred_noisy = pred_perfect + 0.1 * torch.randn_like(pred_perfect)
    pred_noisy = torch.clamp(pred_noisy, 0, 1)
    
    print(f"\nTest tensors shape: {target.shape}")
    print(f"Data range: [{target.min():.3f}, {target.max():.3f}]")
    
    # Test PSNR
    try:
        psnr_perfect = psnr(pred_perfect, target)
        psnr_noisy = psnr(pred_noisy, target)
        print(f"\n✓ PSNR test passed:")
        print(f"  Perfect match: {psnr_perfect:.2f} dB (should be inf)")
        print(f"  Noisy version: {psnr_noisy:.2f} dB (should be lower)")
        assert psnr_perfect > psnr_noisy, "PSNR should be higher for perfect match"
    except Exception as e:
        print(f"✗ PSNR test failed: {e}")
        return False
    
    # Test SSIM
    try:
        ssim_perfect = ssim(pred_perfect, target)
        ssim_noisy = ssim(pred_noisy, target)
        print(f"\n✓ SSIM test passed:")
        print(f"  Perfect match: {ssim_perfect:.4f} (should be ~1.0)")
        print(f"  Noisy version: {ssim_noisy:.4f} (should be lower)")
        assert ssim_perfect > ssim_noisy, "SSIM should be higher for perfect match"
        assert 0.99 <= ssim_perfect <= 1.0, "Perfect SSIM should be close to 1.0"
    except Exception as e:
        print(f"✗ SSIM test failed: {e}")
        return False
    
    # Test LPIPS (optional, requires lpips package)
    try:
        lpips_perfect = lpips_alex(pred_perfect, target)
        lpips_noisy = lpips_alex(pred_noisy, target)
        print(f"\n✓ LPIPS test passed:")
        print(f"  Perfect match: {lpips_perfect:.4f} (should be ~0.0)")
        print(f"  Noisy version: {lpips_noisy:.4f} (should be higher)")
        assert lpips_perfect < lpips_noisy, "LPIPS should be lower for perfect match"
    except ImportError:
        print(f"\n⚠ LPIPS test skipped (lpips package not installed)")
    except Exception as e:
        print(f"✗ LPIPS test failed: {e}")
        return False
    
    # Test with 3D tensors (no batch dimension)
    try:
        pred_3d = pred_perfect.squeeze(0)  # [3, H, W]
        target_3d = target.squeeze(0)      # [3, H, W]
        
        psnr_3d = psnr(pred_3d, target_3d)
        ssim_3d = ssim(pred_3d, target_3d)
        
        print(f"\n✓ 3D tensor test passed:")
        print(f"  PSNR: {psnr_3d:.2f} dB")
        print(f"  SSIM: {ssim_3d:.4f}")
        
    except Exception as e:
        print(f"✗ 3D tensor test failed: {e}")
        return False
    
    print(f"\n✅ All restoration metrics tests passed!")
    return True


if __name__ == "__main__":
    success = test_restoration_metrics()
    if not success:
        sys.exit(1)