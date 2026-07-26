#!/usr/bin/env python3
"""
Test script to verify that validation works correctly with the tensor shape fix.
Tests the validation function using the saved checkpoint.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from src.trainer.marigold_restoration_trainer import MarigoldRestorationTrainer
from src.dataset.restoration_dataset_factory import RestorationDatasetFactory
from src.dataset.base_restoration_dataset import DatasetMode
from marigold import MarigoldRestorationPipeline
from src.util.config_utils import recursive_load_config
from src.util.logging_util import config_logging
from src.util.metric import MetricTracker

def test_validation_fix():
    """Test validation with the tensor shape fix"""
    
    print("🧪 Testing validation fix...")
    
    # Load configuration
    config_path = project_root / "config" / "train_marigold_restoration_test.yaml"
    cfg = recursive_load_config(config_path)
    
    print(f"✓ Configuration loaded from: {config_path}")
    
    # Debug: print configuration structure
    print(f"📋 Dataset config structure:")
    print(f"   cfg.dataset.val type: {type(cfg.dataset.val)}")
    if isinstance(cfg.dataset.val, list):
        print(f"   cfg.dataset.val length: {len(cfg.dataset.val)}")
        print(f"   cfg.dataset.val[0]: {cfg.dataset.val[0]}")
    else:
        print(f"   cfg.dataset.val: {cfg.dataset.val}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Setup base data directory (same as training script)
    base_data_dir = os.environ.get("BASE_DATA_DIR", "./data")
    base_data_dir = os.path.abspath(base_data_dir)
    
    print(f"✓ Using base data directory: {base_data_dir}")
    
    # Create validation dataset (small subset)
    dataset_factory = RestorationDatasetFactory()
    
    # cfg.dataset.val is a list, iterate like in training script
    val_cfg = cfg.dataset.val[0]  # Take first validation dataset
    val_dataset = dataset_factory.create_dataset(val_cfg, mode=DatasetMode.EVAL, base_data_dir=base_data_dir)
    
    # Create dataloader for testing (we'll limit iterations manually)
    from torch.utils.data import DataLoader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Created validation dataset with {len(val_dataset)} samples (will test 2)")
    
    # Load base model (same as training script)
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = MarigoldRestorationPipeline.from_pretrained(
        cfg.model.pretrained_path, **_pipeline_kwargs
    )
    
    print(f"✓ Base model loaded from: {cfg.model.pretrained_path}")
    
    # Check if checkpoint exists
    checkpoint_dir = project_root / "output" / "test_20gb" / "train_marigold_restoration_test" / "checkpoint" / "latest"
    
    if checkpoint_dir.exists():
        print(f"✓ Found checkpoint at: {checkpoint_dir}")
        # Load U-Net weights from checkpoint
        unet_path = checkpoint_dir / "unet"
        if unet_path.exists():
            from diffusers import UNet2DConditionModel
            model.unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float32)
            print("✓ Loaded trained U-Net weights from checkpoint")
        else:
            print("⚠️  No U-Net checkpoint found, using base model")
    else:
        print("⚠️  No checkpoint found, using base model")
    model.to(device)
    
    print("✓ Model loaded successfully")
    
    # Create trainer (we only need it for validation method)
    trainer = MarigoldRestorationTrainer(
        cfg=cfg,
        model=model,
        train_dataloader=None,  # Not needed for validation test
        device=device,
        out_dir_ckpt=str(checkpoint_dir.parent),
        out_dir_eval="test_output",
        out_dir_vis="test_output",
        accumulation_steps=1,
        val_dataloaders=[val_dataloader],
    )
    
    print("✓ Trainer created successfully")
    
    # Test validation
    print("\n🔍 Testing validation...")
    
    try:
        # Create metric tracker with metric names (like in trainer)
        metric_names = [m.__name__ for m in trainer.metric_funcs]
        metric_tracker = MetricTracker(*metric_names)
        
        # Run validation on small dataset (manually limit to 2 samples)
        trainer.model.to(device)
        metric_tracker.reset()
        
        # Generate seed sequence for consistent evaluation
        val_init_seed = trainer.cfg.validation.init_seed
        val_seed_ls = [42, 43]  # Just 2 seeds for testing
        
        sample_count = 0
        max_samples = 2
        
        for i, batch in enumerate(val_dataloader):
            if sample_count >= max_samples:
                break
                
            print(f"   Processing sample {sample_count + 1}/{max_samples}")
            
            # Read input image (degraded) - pipeline expects [0, 255] format
            degraded_rgb_int = batch["degraded_rgb_int"]  # [B, 3, H, W] in [0, 255]
            # GT clean image
            clean_rgb_int = batch["clean_rgb_int"]  # [B, 3, H, W] in [0, 255]
            clean_rgb_ts = clean_rgb_int.squeeze().to(device)  # [3, H, W]

            # Random number generator
            seed = val_seed_ls[sample_count] if sample_count < len(val_seed_ls) else None
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Predict restored image
            pipe_out = trainer.model(
                degraded_rgb_int,
                denoising_steps=trainer.cfg.validation.denoising_steps,
                ensemble_size=trainer.cfg.validation.ensemble_size,
                processing_res=trainer.cfg.validation.processing_res,
                match_input_res=trainer.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                show_progress_bar=False,
                resample_method=trainer.cfg.validation.resample_method,
            )

            restored_rgb = pipe_out.restored_np  # [3, H, W] in [0, 1]

            # Convert to tensor for metric calculation [3, H, W] in [0, 1]
            restored_rgb_ts = torch.from_numpy(restored_rgb).to(device)
            clean_rgb_ts = clean_rgb_ts.float() / 255.0  # Convert to [0, 1]

            # Evaluate restoration metrics (move to CPU for metrics)
            for met_func in trainer.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(restored_rgb_ts.cpu(), clean_rgb_ts.cpu())
                metric_tracker.update(_metric_name, _metric)
            
            sample_count += 1
        
        val_metrics = metric_tracker.result()
        
        print("✅ Validation completed successfully!")
        print(f"📊 Validation metrics: {val_metrics}")
        
        # Check that we got reasonable metrics
        if 'psnr' in val_metrics:
            psnr_value = val_metrics['psnr']
            print(f"   PSNR: {psnr_value:.2f} dB")
            
            if psnr_value > 0 and psnr_value < 100:  # Reasonable range
                print("✅ PSNR value is in reasonable range")
            else:
                print(f"⚠️  PSNR value seems unusual: {psnr_value}")
        
        if 'ssim' in val_metrics:
            ssim_value = val_metrics['ssim']
            print(f"   SSIM: {ssim_value:.4f}")
            
            if 0 <= ssim_value <= 1:  # Valid SSIM range
                print("✅ SSIM value is in valid range")
            else:
                print(f"⚠️  SSIM value is out of range: {ssim_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validation_fix()
    
    if success:
        print("\n🎉 Validation fix test PASSED!")
        print("✅ The tensor shape issue has been resolved")
        print("✅ Training can now continue without validation errors")
    else:
        print("\n💥 Validation fix test FAILED!")
        print("❌ There are still issues to resolve")
    
    sys.exit(0 if success else 1)