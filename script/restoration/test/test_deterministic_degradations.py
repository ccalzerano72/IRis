#!/usr/bin/env python3
"""
Test script to verify deterministic-variable degradation behavior.

Expected behavior:
1. EVAL mode: Same image → Same degradation (always)
2. TRAIN mode: Same image → Different degradation each epoch (but deterministic)
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.dataset.base_restoration_dataset import BaseRestorationDataset, DatasetMode
from omegaconf import OmegaConf


def test_eval_mode_consistency():
    """Test that EVAL mode produces same degradation for same image"""
    print("\n" + "="*80)
    print("TEST 1: EVAL Mode - Same Degradation Across Multiple Calls")
    print("="*80)
    
    # Create minimal config
    config = {
        'mode': 'online',
        'base_seed': 42,
        'types': ['whitenoise'],
        'levels': [2],
        'mixed_prob': 0.0,
    }
    
    # Create dataset (using a small test dataset)
    dataset = BaseRestorationDataset(
        mode=DatasetMode.EVAL,
        filename_ls_path='data_split/restoration/laion_vis_sample.txt',
        clean_dir='laion/val/clean',  # Will be resolved to data/laion/val/clean
        degraded_dir='',  # Not used in online mode
        disp_name='test_eval',
        degradation_config=config,
        resize_to_hw=[256, 256],  # Small size for fast testing
    )
    
    print(f"✓ Dataset created with {len(dataset)} images")
    print(f"✓ Mode: {dataset.mode}")
    print(f"✓ Base seed: {dataset.degradation_base_seed}")
    
    # Test first 3 images
    num_test_images = min(3, len(dataset))
    
    for img_idx in range(num_test_images):
        print(f"\n--- Testing Image {img_idx} ---")
        
        # Get degradation 3 times
        degradations = []
        for call_num in range(3):
            sample = dataset[img_idx]
            degraded = sample['degraded_rgb_norm']
            degradations.append(degraded.clone())
        
        # Check if all 3 are identical
        all_same = True
        for i in range(1, 3):
            if not torch.allclose(degradations[0], degradations[i], atol=1e-6):
                all_same = False
                break
        
        if all_same:
            print(f"  ✅ Image {img_idx}: All 3 calls produced IDENTICAL degradation")
        else:
            print(f"  ❌ Image {img_idx}: Degradations are DIFFERENT!")
            for i in range(3):
                print(f"     Call {i}: mean={degradations[i].mean():.6f}, std={degradations[i].std():.6f}")
    
    print("\n✓ EVAL mode test completed")
    return all_same


def test_train_mode_variability():
    """Test that TRAIN mode produces different degradations across epochs"""
    print("\n" + "="*80)
    print("TEST 2: TRAIN Mode - Different Degradation Each Epoch")
    print("="*80)
    
    # Create minimal config
    config = {
        'mode': 'online',
        'base_seed': 42,
        'types': ['whitenoise'],
        'levels': [2],
        'mixed_prob': 0.0,
    }
    
    # Create dataset
    dataset = BaseRestorationDataset(
        mode=DatasetMode.TRAIN,
        filename_ls_path='data_split/restoration/laion_vis_sample.txt',
        clean_dir='laion/val/clean',  # Will be resolved to data/laion/val/clean
        degraded_dir='',
        disp_name='test_train',
        degradation_config=config,
        resize_to_hw=[256, 256],
    )
    
    print(f"✓ Dataset created with {len(dataset)} images")
    print(f"✓ Mode: {dataset.mode}")
    print(f"✓ Base seed: {dataset.degradation_base_seed}")
    
    # Test first 3 images across 3 epochs
    num_test_images = min(3, len(dataset))
    num_epochs = 3
    
    for img_idx in range(num_test_images):
        print(f"\n--- Testing Image {img_idx} Across Epochs ---")
        
        # Get degradation for each epoch
        epoch_degradations = []
        for epoch in range(num_epochs):
            dataset.set_epoch(epoch)
            sample = dataset[img_idx]
            degraded = sample['degraded_rgb_norm']
            epoch_degradations.append(degraded.clone())
            print(f"  Epoch {epoch}: mean={degraded.mean():.6f}, std={degraded.std():.6f}")
        
        # Check if degradations are different across epochs
        all_different = True
        for i in range(1, num_epochs):
            if torch.allclose(epoch_degradations[0], epoch_degradations[i], atol=1e-6):
                all_different = False
                break
        
        if all_different:
            print(f"  ✅ Image {img_idx}: Degradations are DIFFERENT across epochs")
        else:
            print(f"  ❌ Image {img_idx}: Some degradations are IDENTICAL!")
    
    print("\n✓ TRAIN mode test completed")
    return all_different


def test_train_mode_determinism():
    """Test that TRAIN mode is deterministic (same epoch → same degradation)"""
    print("\n" + "="*80)
    print("TEST 3: TRAIN Mode - Deterministic (Same Epoch → Same Degradation)")
    print("="*80)
    
    # Create minimal config
    config = {
        'mode': 'online',
        'base_seed': 42,
        'types': ['whitenoise'],
        'levels': [2],
        'mixed_prob': 0.0,
    }
    
    # Create dataset
    dataset = BaseRestorationDataset(
        mode=DatasetMode.TRAIN,
        filename_ls_path='data_split/restoration/laion_vis_sample.txt',
        clean_dir='laion/val/clean',  # Will be resolved to data/laion/val/clean
        degraded_dir='',
        disp_name='test_train_det',
        degradation_config=config,
        resize_to_hw=[256, 256],
    )
    
    print(f"✓ Dataset created with {len(dataset)} images")
    
    # Test first 3 images
    num_test_images = min(3, len(dataset))
    test_epoch = 5
    
    for img_idx in range(num_test_images):
        print(f"\n--- Testing Image {img_idx} at Epoch {test_epoch} ---")
        
        # Get degradation 3 times for same epoch
        dataset.set_epoch(test_epoch)
        degradations = []
        for call_num in range(3):
            sample = dataset[img_idx]
            degraded = sample['degraded_rgb_norm']
            degradations.append(degraded.clone())
        
        # Check if all 3 are identical
        all_same = True
        for i in range(1, 3):
            if not torch.allclose(degradations[0], degradations[i], atol=1e-6):
                all_same = False
                break
        
        if all_same:
            print(f"  ✅ Image {img_idx}: All 3 calls at epoch {test_epoch} produced IDENTICAL degradation")
        else:
            print(f"  ❌ Image {img_idx}: Degradations are DIFFERENT!")
    
    print("\n✓ TRAIN mode determinism test completed")
    return all_same


def main():
    print("\n" + "="*80)
    print("DETERMINISTIC-VARIABLE DEGRADATION TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: EVAL mode consistency
        eval_pass = test_eval_mode_consistency()
        
        # Test 2: TRAIN mode variability
        train_var_pass = test_train_mode_variability()
        
        # Test 3: TRAIN mode determinism
        train_det_pass = test_train_mode_determinism()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"1. EVAL Mode Consistency:      {'✅ PASS' if eval_pass else '❌ FAIL'}")
        print(f"2. TRAIN Mode Variability:     {'✅ PASS' if train_var_pass else '❌ FAIL'}")
        print(f"3. TRAIN Mode Determinism:     {'✅ PASS' if train_det_pass else '❌ FAIL'}")
        
        if eval_pass and train_var_pass and train_det_pass:
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())