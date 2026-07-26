#!/usr/bin/env python3
"""
Test script for restoration dataset implementation
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_restoration_dataset():
    """Test restoration dataset loading"""
    
    print("Testing Restoration Dataset...")
    
    try:
        from src.dataset.base_restoration_dataset import BaseRestorationDataset, DatasetMode
        from src.dataset.div2k_restoration_dataset import DIV2KRestorationDataset
        from src.dataset.restoration_dataset_factory import RestorationDatasetFactory
        print("✓ Successfully imported restoration dataset classes")
    except ImportError as e:
        print(f"✗ Failed to import dataset classes: {e}")
        return False
    
    # Test factory registry
    try:
        available_datasets = RestorationDatasetFactory.list_available_datasets()
        print(f"✓ Available datasets: {available_datasets}")
        assert 'div2k' in available_datasets
        assert 'kadis700k' in available_datasets
    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        return False
    
    # Test configuration loading
    try:
        from omegaconf import OmegaConf
        from src.util.config_util import recursive_load_config
        
        # Load train config
        config_path = "config/dataset_restoration/dataset_train.yaml"
        if os.path.exists(config_path):
            cfg = recursive_load_config(config_path)
            OmegaConf.resolve(cfg)
            print(f"✓ Successfully loaded config: {config_path}")
            print(f"  Dataset name: {cfg.dataset.train.name}")
            print(f"  Clean dir: {cfg.dataset.train.dir}")
            print(f"  Degraded dir: {cfg.dataset.train.degradation_config.degraded_dir}")
        else:
            print(f"⚠ Config not found: {config_path}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    # Test dataset creation (will fail if data doesn't exist, but that's expected)
    try:
        if os.path.exists(config_path):
            train_cfg = cfg.dataset.train
            
            # Test factory creation (with auto_generate=False to avoid errors)
            try:
                dataset = RestorationDatasetFactory.create_dataset(
                    train_cfg, DatasetMode.TRAIN, auto_generate=False
                )
                print(f"✓ Dataset created successfully: {dataset.disp_name}")
                print(f"  Dataset length: {len(dataset)}")
            except FileNotFoundError as e:
                print(f"⚠ Dataset creation failed (expected if data not prepared): {str(e)[:100]}...")
                print("  This is normal if you haven't run dataset preparation yet")
            except Exception as e:
                print(f"✗ Unexpected dataset creation error: {e}")
                return False
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False
    
    print(f"\n✅ Restoration dataset implementation tests completed!")
    print(f"\nTo prepare actual data, run:")
    print(f"  python script/restoration/dataset_preprocess/prepare_dataset.py --source_dir /path/to/DIV2K")
    
    return True


if __name__ == "__main__":
    success = test_restoration_dataset()
    if not success:
        sys.exit(1)