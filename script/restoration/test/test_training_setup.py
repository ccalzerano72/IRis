#!/usr/bin/env python3
"""
Test script for restoration training setup

Thesis Implementation: Blind Image Restoration via Diffusion-Based Quality-Aware Reconstruction
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_training_setup():
    """Test restoration training setup"""
    
    print("Testing Restoration Training Setup...")
    
    # Test imports
    try:
        from marigold import MarigoldRestorationPipeline
        from src.trainer.marigold_restoration_trainer import MarigoldRestorationTrainer
        from src.dataset.restoration_dataset_factory import RestorationDatasetFactory
        from src.dataset import DatasetMode
        print("✓ Successfully imported training components")
    except ImportError as e:
        print(f"✗ Failed to import training components: {e}")
        return False
    
    # Test configuration loading
    try:
        from omegaconf import OmegaConf
        from src.util.config_util import recursive_load_config
        
        config_path = "config/train_marigold_restoration.yaml"
        if os.path.exists(config_path):
            cfg = recursive_load_config(config_path)
            OmegaConf.resolve(cfg)
            print(f"✓ Successfully loaded training config: {config_path}")
            print(f"  Pipeline: {cfg.pipeline.name}")
            print(f"  Trainer: {cfg.trainer.name}")
            print(f"  Max iterations: {cfg.max_iter}")
            print(f"  Learning rate: {cfg.lr}")
            print(f"  Effective batch size: {cfg.dataloader.effective_batch_size}")
        else:
            print(f"✗ Training config not found: {config_path}")
            return False
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    # Test pipeline creation (without loading weights)
    try:
        # This will fail if no pretrained model is available, but that's expected
        print("⚠ Pipeline creation test skipped (requires pretrained model)")
        print("  To test with actual model, ensure BASE_CKPT_DIR is set and contains stable-diffusion-2")
    except Exception as e:
        print(f"⚠ Pipeline creation test skipped: {e}")
    
    # Test trainer class instantiation (without actual data)
    try:
        # We can't fully instantiate without data, but we can check the class exists
        trainer_cls = MarigoldRestorationTrainer
        print(f"✓ MarigoldRestorationTrainer class available: {trainer_cls}")
    except Exception as e:
        print(f"✗ Trainer class test failed: {e}")
        return False
    
    # Test training script exists
    train_script_path = "script/restoration/train.py"
    if os.path.exists(train_script_path):
        print(f"✓ Training script exists: {train_script_path}")
    else:
        print(f"✗ Training script not found: {train_script_path}")
        return False
    
    print(f"\n✅ Restoration training setup tests completed!")
    print(f"\nTo start training (after preparing data):")
    print(f"  export BASE_DATA_DIR=/path/to/data")
    print(f"  export BASE_CKPT_DIR=/path/to/checkpoints")
    print(f"  python script/restoration/train.py --config config/train_marigold_restoration.yaml")
    
    return True


if __name__ == "__main__":
    success = test_training_setup()
    if not success:
        sys.exit(1)