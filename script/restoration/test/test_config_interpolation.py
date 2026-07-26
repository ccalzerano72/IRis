"""
Test script to verify variable interpolation in restoration dataset configs.

This script tests:
1. OmegaConf native interpolation
2. Custom interpolation fallback (if native doesn't work)
3. Hierarchical config loading with base_config
"""

import sys
import os

# Add project root to path (go up 3 levels: test -> restoration -> script -> project_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from omegaconf import OmegaConf
from src.util.config_util import recursive_load_config
from src.util.config_utils import resolve_config_variables, load_config_with_interpolation


def test_native_omegaconf():
    """Test if OmegaConf natively supports variable interpolation."""
    print("=" * 60)
    print("TEST 1: OmegaConf Native Interpolation (with base_config)")
    print("=" * 60)
    
    try:
        # Load config with base_config merging (Marigold's standard approach)
        cfg = recursive_load_config('config/dataset_restoration/dataset_train.yaml')
        
        print(f"✓ Config loaded with base_config merged")
        print(f"\nMerged config (before resolution):")
        print(OmegaConf.to_yaml(cfg)[:500] + "...")  # Show first 500 chars
        
        # Try to resolve
        OmegaConf.resolve(cfg)
        
        print(f"\n✓ OmegaConf.resolve() succeeded!")
        
        # Check if variables are resolved
        train_name = cfg.dataset.train.name
        train_dir = cfg.dataset.train.dir
        
        print(f"\n✓ Variables resolved:")
        print(f"  - name: {train_name}")
        print(f"  - dir: {train_dir}")
        print(f"  - filenames: {cfg.dataset.train.filenames}")
        print(f"  - degraded_dir: {cfg.dataset.train.degradation_config.degraded_dir}")
        
        if '${' in train_name or '${' in train_dir:
            print(f"\n✗ Variables NOT resolved (still contain ${{...}})")
            return False
        else:
            print(f"\n✓ Native interpolation WORKS!")
            return True
            
    except Exception as e:
        print(f"\n✗ Native interpolation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_interpolation():
    """Test custom variable interpolation fallback."""
    print("\n" + "=" * 60)
    print("TEST 2: Custom Variable Interpolation (with base_config)")
    print("=" * 60)
    
    try:
        # Load config with base_config merging
        cfg = recursive_load_config('config/dataset_restoration/dataset_train.yaml')
        
        print(f"✓ Config loaded with base_config merged")
        
        # Apply custom resolution
        resolved_cfg = resolve_config_variables(cfg)
        
        print(f"✓ Custom resolution applied")
        
        # Check if variables are resolved
        train_name = resolved_cfg.dataset.train.name
        train_dir = resolved_cfg.dataset.train.dir
        
        print(f"\n✓ Variables resolved:")
        print(f"  - name: {train_name}")
        print(f"  - dir: {train_dir}")
        print(f"  - filenames: {resolved_cfg.dataset.train.filenames}")
        print(f"  - degraded_dir: {resolved_cfg.dataset.train.degradation_config.degraded_dir}")
        
        if '${' in train_name or '${' in train_dir:
            print(f"\n✗ Variables NOT resolved")
            return False
        else:
            print(f"\n✓ Custom interpolation WORKS!")
            return True
            
    except Exception as e:
        print(f"\n✗ Custom interpolation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_configs():
    """Test all dataset configs (train, val, vis)."""
    print("\n" + "=" * 60)
    print("TEST 3: All Dataset Configs")
    print("=" * 60)
    
    configs = [
        'config/dataset_restoration/dataset_train.yaml',
        'config/dataset_restoration/dataset_val.yaml',
        'config/dataset_restoration/dataset_vis.yaml',
    ]
    
    all_passed = True
    
    for config_path in configs:
        print(f"\nTesting: {config_path}")
        try:
            cfg = load_config_with_interpolation(config_path)
            
            # Get the dataset section (train, val, or vis)
            if 'train' in cfg.dataset:
                dataset_cfg = cfg.dataset.train
                print(f"  ✓ Train config:")
                print(f"    - name: {dataset_cfg.name}")
                print(f"    - dir: {dataset_cfg.dir}")
                print(f"    - degraded_dir: {dataset_cfg.degradation_config.degraded_dir}")
            elif 'val' in cfg.dataset:
                dataset_cfg = cfg.dataset.val[0]
                print(f"  ✓ Val config:")
                print(f"    - name: {dataset_cfg.name}")
                print(f"    - dir: {dataset_cfg.dir}")
                print(f"    - degraded_dir: {dataset_cfg.degradation_config.degraded_dir}")
            elif 'vis' in cfg.dataset:
                dataset_cfg = cfg.dataset.vis[0]
                print(f"  ✓ Vis config:")
                print(f"    - name: {dataset_cfg.name}")
                print(f"    - dir: {dataset_cfg.dir}")
                print(f"    - degraded_dir: {dataset_cfg.degradation_config.degraded_dir}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING VARIABLE INTERPOLATION IN RESTORATION CONFIGS")
    print("=" * 60)
    
    # Test 1: Native OmegaConf
    native_works = test_native_omegaconf()
    
    # Test 2: Custom interpolation
    custom_works = test_custom_interpolation()
    
    # Test 3: All configs
    all_configs_work = test_all_configs()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Native OmegaConf interpolation: {'✓ WORKS' if native_works else '✗ FAILED'}")
    print(f"Custom interpolation fallback: {'✓ WORKS' if custom_works else '✗ FAILED'}")
    print(f"All configs load correctly: {'✓ WORKS' if all_configs_work else '✗ FAILED'}")
    
    if native_works:
        print("\n✓ OmegaConf natively supports variable interpolation!")
        print("  No custom handling needed in scripts.")
    elif custom_works:
        print("\n⚠ OmegaConf doesn't support variable interpolation natively.")
        print("  Use custom utility: load_config_with_interpolation()")
    else:
        print("\n✗ Variable interpolation not working!")
        print("  Need to debug config_utils.py")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
