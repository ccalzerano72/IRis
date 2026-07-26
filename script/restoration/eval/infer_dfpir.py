#!/usr/bin/env python3
"""
DFPIR Inference Script for Evaluation

Runs DFPIR (Degradation-aware Feature Perturbation for All-in-One Image Restoration)
on a folder of degraded images.

Usage:
    python script/restoration/eval/infer_dfpir.py \
        --input_dir path/to/degraded \
        --output_dir path/to/output \
        --checkpoint external/DFPIR/pretrained/DFPIR-3D_xxx.pth.tar \
        --degradation general

Degradation types:
    noise15, noise25, noise50 - Gaussian noise
    rain - Rain streaks
    haze - Haze/fog
    blur - Motion blur (requires 5D model)
    lowlight - Low light enhancement (requires 5D model)
    general - Generic prompt for unknown degradation (default)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add DFPIR to path
DFPIR_DIR = Path(__file__).parent.parent.parent.parent / "external" / "DFPIR"
sys.path.insert(0, str(DFPIR_DIR))

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Text prompts for each degradation type (from DFPIR test scripts)
DEGRADATION_PROMPTS = {
    'noise15': "Gaussian noise with a standard deviation of 15",
    'noise25': "Gaussian noise with a standard deviation of 25",
    'noise50': "Gaussian noise with a standard deviation of 50",
    'rain': "Rain degradation with rain lines",
    'haze': "Hazy degradation with normal haze",
    'blur': "Blur degradation with motion blur",
    'lowlight': "Lowlight degradation",
    'general': "Degraded image with quality loss",
}


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load DFPIR model from checkpoint.
    
    Pattern from external/DFPIR/test_3D_DFPIR.py lines 108-118
    """
    from net.model import ChannelShuffle_skip_textguaid
    
    model = ChannelShuffle_skip_textguaid(device=device)
    model = model.to(device)
    
    if os.path.isfile(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Model loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    return model


def load_clip_model(device: torch.device):
    """
    Load CLIP model for text encoding.
    
    Pattern from external/DFPIR/test_3D_DFPIR.py lines 24-27
    """
    import clip
    
    clip_model, _ = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False
    
    return clip_model


def encode_text_prompt(clip_model, prompt: str, device: torch.device) -> torch.Tensor:
    """
    Encode text prompt using CLIP.
    
    Pattern from external/DFPIR/test_3D_DFPIR.py lines 56-57
    """
    import clip
    
    text_token = clip.tokenize(prompt).to(device)
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)
    return text_code


def pad_to_multiple(img_tensor: torch.Tensor, multiple: int = 16) -> tuple:
    """
    Pad image tensor so dimensions are divisible by multiple.
    
    Args:
        img_tensor: Tensor of shape [1, 3, H, W]
        multiple: Dimensions must be divisible by this (default: 16)
    
    Returns:
        (padded_tensor, original_h, original_w)
    """
    _, _, h, w = img_tensor.shape
    
    # Calculate padding needed
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h == 0 and pad_w == 0:
        return img_tensor, h, w
    
    # Pad on right and bottom (reflection padding for better edge handling)
    padded = torch.nn.functional.pad(
        img_tensor, 
        (0, pad_w, 0, pad_h),  # (left, right, top, bottom)
        mode='reflect'
    )
    
    return padded, h, w


def load_image(image_path: Path) -> torch.Tensor:
    """
    Load image and convert to tensor [0, 1] range.
    
    Returns tensor of shape [1, 3, H, W]
    """
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW -> BCHW
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def save_image(tensor: torch.Tensor, output_path: Path):
    """
    Save tensor to image file.
    
    Pattern from external/DFPIR/utils/image_io.py save_image_tensor function
    """
    # Clamp to [0, 1] and convert to numpy
    img_np = torch.clamp(tensor, 0, 1).detach().cpu().numpy()[0]
    # CHW -> HWC, scale to [0, 255]
    img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(output_path)


def run_inference(
    model,
    clip_model,
    input_dir: Path,
    output_dir: Path,
    degradation: str,
    device: torch.device,
    max_images: int = 0,
):
    """
    Run DFPIR inference on all images in input directory.
    """
    # Get text embedding for degradation type
    prompt = DEGRADATION_PROMPTS.get(degradation, DEGRADATION_PROMPTS['general'])
    print(f"Using prompt: '{prompt}'")
    text_code = encode_text_prompt(clip_model, prompt, device)
    
    # Find all images
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Filter out already-processed images
    filtered_files = []
    skipped_count = 0
    for img_path in image_files:
        output_path = output_dir / img_path.name
        if output_path.exists():
            skipped_count += 1
        else:
            filtered_files.append(img_path)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already-processed images")
    
    if not filtered_files:
        print("All images already processed. Nothing to do.")
        return
    
    image_files = filtered_files
    
    # Apply max_images limit if set
    if max_images > 0 and len(image_files) > max_images:
        print(f"Limiting to {max_images} images (out of {len(image_files)})")
        image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} images")
    
    # Process each image
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processing"):
            try:
                # Load image
                img_tensor = load_image(img_path).to(device)
                
                # Pad to multiple of 16 for encoder-decoder architecture
                img_padded, orig_h, orig_w = pad_to_multiple(img_tensor, multiple=16)
                
                # Run model
                restored = model(img_padded, text_code)
                
                # Crop back to original size
                restored = restored[:, :, :orig_h, :orig_w]
                
                # Save result
                output_path = output_dir / img_path.name
                save_image(restored, output_path)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
    
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="DFPIR Inference for Image Restoration Evaluation"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing degraded images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save restored images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DFPIR checkpoint (default: auto-detect 3D model)"
    )
    parser.add_argument(
        "--degradation",
        type=str,
        default="general",
        choices=list(DEGRADATION_PROMPTS.keys()),
        help="Degradation type (default: general)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Maximum number of images to process. 0 = all images. Default: 0"
    )
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect checkpoint if not specified
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Look for 3D model in pretrained folder
        pretrained_dir = DFPIR_DIR / "pretrained"
        candidates = list(pretrained_dir.glob("DFPIR-3D*.pth.tar"))
        if candidates:
            checkpoint_path = str(candidates[0])
            print(f"Auto-detected checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {pretrained_dir}. "
                "Please specify --checkpoint"
            )
    
    # Load models
    print("Loading CLIP model...")
    clip_model = load_clip_model(device)
    
    print("Loading DFPIR model...")
    model = load_model(checkpoint_path, device)
    
    # Run inference
    print(f"\nRunning inference with degradation type: {args.degradation}")
    run_inference(
        model=model,
        clip_model=clip_model,
        input_dir=input_dir,
        output_dir=output_dir,
        degradation=args.degradation,
        device=device,
        max_images=args.max_images,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
