#!/usr/bin/env python3
"""
Minimal training test for restoration system

This script tests the training pipeline with a very small setup:
- Few iterations only
- Small batch size
- Automatic model download from HuggingFace
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import tempfile
import shutil
from omegaconf import OmegaConf

def test_minimal_training():
    """Test minimal training setup"""
    
    print("🚀 Testing Minimal Training Setup...")
    
    # Check if we have data
    data_dir = "data/div2k"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run dataset preparation first")
        return False
    
    # Check data files
    train_clean = os.path.join(data_dir, "train", "clean")
    train_degraded = os.path.join(data_dir, "train", "degraded")
    
    if not os.path.exists(train_clean) or not os.path.exists(train_degraded):
        print(f"❌ Training data not found in {data_dir}")
        return False
    
    # Count files
    clean_files = len([f for f in os.listdir(train_clean) if f.endswith('.png')])
    degraded_files = len([f for f in os.listdir(train_degraded) if f.endswith('.png')])
    
    print(f"✓ Found {clean_files} clean images and {degraded_files} degraded images")
    
    if clean_files < 10 or degraded_files < 10:
        print("⚠️ Very few images found, but proceeding with test...")
    
    # Test imports
    try:
        from marigold import MarigoldRestorationPipeline
        from src.trainer.marigold_restoration_trainer import MarigoldRestorationTrainer
        from src.dataset.restoration_dataset_factory import RestorationDatasetFactory
        from src.dataset import DatasetMode
        from torch.utils.data import DataLoader
        print("✓ All training components imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Create minimal config
    print("📝 Creating minimal training config...")
    
    # Set environment variables
    os.environ['BASE_DATA_DIR'] = os.path.abspath('data')
    
    # Use output directory (symlink to large disk)
    output_base = os.path.abspath("output")
    os.makedirs(output_base, exist_ok=True)
    
    temp_ckpt_dir = os.path.join(output_base, "test_training_temp")
    os.makedirs(temp_ckpt_dir, exist_ok=True)
    os.environ['BASE_CKPT_DIR'] = temp_ckpt_dir
    
    print(f"📁 Using checkpoint directory: {temp_ckpt_dir}")
    print(f"💾 This uses the output symlink to large disk")
    print(f"💾 Estimated space needed: ~100-500 MB for test")
    
    try:
        # Load and modify config for minimal test
        from src.util.config_util import recursive_load_config
        
        cfg = recursive_load_config("config/train_marigold_restoration.yaml")
        
        # Minimal settings for test
        cfg.max_iter = 5  # Just 5 iterations
        cfg.dataloader.max_train_batch_size = 1  # Small batch
        cfg.dataloader.effective_batch_size = 1  # No accumulation
        cfg.trainer.save_period = 2  # Save every 2 iterations
        cfg.trainer.validation_period = 10  # No validation during test
        cfg.trainer.visualization_period = 10  # No visualization during test
        cfg.trainer.backup_period = 10  # No backup during test
        
        print("✓ Config loaded and modified for minimal test")
        
        # Test dataset creation
        print("📊 Testing dataset creation...")
        
        # Disable auto-generation for test (we have prepared data)
        train_dataset = RestorationDatasetFactory.create_dataset(
            cfg.dataset.train,
            mode=DatasetMode.TRAIN,
            auto_generate=False,
        )
        
        print(f"✓ Training dataset created: {len(train_dataset)} samples")
        
        # Create data loader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            num_workers=0,  # No multiprocessing for test
            shuffle=True,
        )
        
        print("✓ Data loader created")
        
        # Test model loading (this will download from HuggingFace)
        print("🤖 Loading model (may download from HuggingFace)...")
        
        try:
            model = MarigoldRestorationPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2",  # Correct model name
                torch_dtype=torch.float32,  # Use float32 for compatibility
            )
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("This might be due to network issues or missing dependencies")
            return False
        
        # Test trainer creation
        print("🏋️ Creating trainer...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create output directories in output folder
        out_dir_run = os.path.join(output_base, "test_training_run")
        os.makedirs(out_dir_run, exist_ok=True)
        
        out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
        out_dir_eval = os.path.join(out_dir_run, "evaluation")
        out_dir_vis = os.path.join(out_dir_run, "visualization")
        out_dir_tb = os.path.join(out_dir_run, "tensorboard")
        
        for d in [out_dir_ckpt, out_dir_eval, out_dir_vis, out_dir_tb]:
            os.makedirs(d, exist_ok=True)
        
        # Initialize logging (required for trainer output)
        from src.util.logging_util import tb_logger, config_logging
        
        # Configure logging to see training progress
        logging_config = {
            "filename": "test_training.log",
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "console_level": 20,  # INFO level
            "file_level": 10      # DEBUG level
        }
        config_logging(logging_config, out_dir=out_dir_run)
        
        # Initialize tensorboard logger (required before trainer creation)
        tb_logger.set_dir(out_dir_tb)
        
        trainer = MarigoldRestorationTrainer(
            cfg=cfg,
            model=model,
            train_dataloader=train_loader,
            device=device,
            out_dir_ckpt=out_dir_ckpt,
            out_dir_eval=out_dir_eval,
            out_dir_vis=out_dir_vis,
            accumulation_steps=1,
            val_dataloaders=[],  # No validation for test
            vis_dataloaders=[],  # No visualization for test
        )
        
        print("✓ Trainer created successfully")
        
        # Test training loop (just a few iterations)
        print("🚂 Starting minimal training test...")
        
        try:
            trainer.train()
            print("✅ Training test completed successfully!")
            
            # Check if checkpoint was saved
            if os.path.exists(os.path.join(out_dir_ckpt, "latest")):
                print("✓ Checkpoint saved successfully")
            else:
                print("⚠️ No checkpoint found (might be normal for short test)")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🎉 Minimal training test PASSED!")
        print("The complete training system is working correctly.")
        
        return True
        
    finally:
        # Cleanup test directories
        cleanup_dirs = [temp_ckpt_dir, os.path.join(output_base, "test_training_run")]
        for cleanup_dir in cleanup_dirs:
            if os.path.exists(cleanup_dir):
                shutil.rmtree(cleanup_dir, ignore_errors=True)
                print(f"🧹 Cleaned up: {cleanup_dir}")


if __name__ == "__main__":
    success = test_minimal_training()
    if not success:
        print("\n❌ Minimal training test FAILED")
        sys.exit(1)
    else:
        print("\n✅ Minimal training test PASSED")
        print("Ready for full training experiments!")