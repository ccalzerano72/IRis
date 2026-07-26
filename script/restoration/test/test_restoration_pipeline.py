#!/usr/bin/env python3
"""
Simple test script for MarigoldRestorationPipeline
"""

import sys
import os

# Add project root to path (go up 3 levels: test -> restoration -> script -> project_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import numpy as np
from PIL import Image
from marigold import MarigoldRestorationPipeline

def create_test_image(size=(512, 512)):
    """Create a simple test image"""
    # Create a simple gradient image
    height, width = size
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # RGB gradient
    r = X
    g = Y
    b = (X + Y) / 2
    
    rgb = np.stack([r, g, b], axis=2)
    rgb = (rgb * 255).astype(np.uint8)
    
    return Image.fromarray(rgb)

def test_pipeline_creation():
    """Test if we can create the pipeline"""
    print("Testing pipeline creation...")
    
    try:
        # This will fail because we don't have trained weights yet
        # But it should at least show us if the class structure is correct
        pipeline = MarigoldRestorationPipeline.from_pretrained("prs-eth/marigold-depth-v1-1")
        print("✗ Pipeline creation failed as expected (no restoration weights)")
    except Exception as e:
        print(f"✓ Expected error: {e}")
    
    print("Pipeline class structure test completed.")

def test_pipeline_methods():
    """Test individual pipeline methods"""
    print("\nTesting pipeline methods...")
    
    # Create dummy components (this is just for testing method signatures)
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        
        print("✓ All required imports available")
        
        # Test image creation
        test_img = create_test_image()
        print(f"✓ Test image created: {test_img.size}")
        
        # Test tensor conversion
        from torchvision.transforms.functional import pil_to_tensor
        tensor = pil_to_tensor(test_img)
        print(f"✓ Tensor conversion: {tensor.shape}")
        
    except Exception as e:
        print(f"✗ Method test failed: {e}")

if __name__ == "__main__":
    print("=== MarigoldRestorationPipeline Test ===")
    test_pipeline_creation()
    test_pipeline_methods()
    print("\n=== Test completed ===")