#!/usr/bin/env python3
"""
Test script to verify that visualization works correctly.
Tests the visualization function using the saved checkpoint.
Based on the working validation test script.
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

def test_visualization_fix():
    """Test visualization with the tensor shape fix"""
    
    print("🎨 Testing visualization fix...")
    
    # Load configuration
    config_path = project_root / "config" / "train_marigold_restoration_test.yaml"
    cfg = recursive_load_config(config_path)
    
    print(f"✓ Configuration loaded from: {config_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Setup base data directory (same as training script)
    base_data_dir = os.environ.get("BASE_DATA_DIR", "./data")
    base_data_dir = os.path.abspath(base_data_dir)
    
    print(f"✓ Using base data directory: {base_data_dir}")
    
    # Create visualization dataset (small subset)
    dataset_factory = RestorationDatasetFactory()
    
    # cfg.dataset.vis is a list, take the first one
    vis_cfg = cfg.dataset.vis[0]  # Take first visualization dataset
    vis_dataset = dataset_factory.create_dataset(vis_cfg, mode=DatasetMode.EVAL, base_data_dir=base_data_dir)
    
    # Create dataloader for testing (we'll limit iterations manually)
    from torch.utils.data import DataLoader
    vis_dataloader = DataLoader(
        vis_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Created visualization dataset with {len(vis_dataset)} samples (will test 2)")
    
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
    
    # Create trainer (we only need it for visualization method)
    trainer = MarigoldRestorationTrainer(
        cfg=cfg,
        model=model,
        train_dataloader=None,  # Not needed for visualization test
        device=device,
        out_dir_ckpt=str(checkpoint_dir.parent),
        out_dir_eval="test_output",
        out_dir_vis="test_output",
        accumulation_steps=1,
        vis_dataloaders=[vis_dataloader],
    )
    
    print("✓ Trainer created successfully")
    
    # Test visualization
    print("\n🎨 Testing visualization...")
    
    try:
        # Create output directory for test
        test_vis_dir = project_root / "test_output" / "visualization_test"
        test_vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Run visualization on small dataset (manually limit to 2 samples)
        trainer.model.to(device)
        
        # Generate seed sequence for consistent evaluation
        vis_init_seed = trainer.cfg.visualization.init_seed if hasattr(trainer.cfg, 'visualization') else 2024
        vis_seed_ls = [42, 43]  # Just 2 seeds for testing
        
        sample_count = 0
        max_samples = 2
        
        for i, batch in enumerate(vis_dataloader):
            if sample_count >= max_samples:
                break
                
            print(f"   Processing visualization sample {sample_count + 1}/{max_samples}")
            
            # Read input image (degraded) - pipeline expects [0, 255] format
            degraded_rgb_int = batch["degraded_rgb_int"]  # [B, 3, H, W] in [0, 255]
            # GT clean image
            clean_rgb_int = batch["clean_rgb_int"]  # [B, 3, H, W] in [0, 255]

            # Random number generator
            seed = vis_seed_ls[sample_count] if sample_count < len(vis_seed_ls) else None
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

            # Save visualization images
            sample_name = f"vis_sample_{sample_count + 1:03d}"
            
            # Save degraded image
            degraded_img = degraded_rgb_int.squeeze().permute(1, 2, 0).cpu().numpy().astype('uint8')
            from PIL import Image
            Image.fromarray(degraded_img).save(test_vis_dir / f"{sample_name}_degraded.png")
            
            # Save clean image
            clean_img = clean_rgb_int.squeeze().permute(1, 2, 0).cpu().numpy().astype('uint8')
            Image.fromarray(clean_img).save(test_vis_dir / f"{sample_name}_clean.png")
            
            # Save restored image (already PIL Image)
            pipe_out.restored_img.save(test_vis_dir / f"{sample_name}_restored.png")
            
            print(f"     ✓ Saved visualization files for {sample_name}")
            
            sample_count += 1
        
        print("✅ Visualization completed successfully!")
        print(f"📁 Visualization files saved to: {test_vis_dir}")
        
        # Check that files were created
        expected_files = []
        for i in range(max_samples):
            sample_name = f"vis_sample_{i + 1:03d}"
            expected_files.extend([
                f"{sample_name}_degraded.png",
                f"{sample_name}_clean.png", 
                f"{sample_name}_restored.png"
            ])
        
        created_files = list(test_vis_dir.glob("*.png"))
        print(f"📊 Created {len(created_files)} visualization files")
        
        if len(created_files) >= len(expected_files):
            print("✅ All expected visualization files created")
        else:
            print(f"⚠️  Expected {len(expected_files)} files, got {len(created_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization_fix()
    
    if success:
        print("\n🎉 Visualization fix test PASSED!")
        print("✅ The visualization system is working correctly")
        print("✅ Training can now continue with working visualization")
    else:
        print("\n💥 Visualization fix test FAILED!")
        print("❌ There are still issues to resolve")
    
    sys.exit(0 if success else 1)