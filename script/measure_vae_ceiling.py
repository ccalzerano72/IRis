"""
Measure VAE reconstruction ceiling (encode → decode roundtrip).

Computes PSNR and SSIM between original images and their VAE-reconstructed
versions (no diffusion, no restoration — just encode and decode).
This establishes the theoretical upper bound on fidelity for any latent-space method.

Usage:
    python script/measure_vae_ceiling.py --input_dir <path_to_clean_images> [--output_csv results.csv]

Example:
    python script/measure_vae_ceiling.py --input_dir Z:/comparison/DIV2K/original
    python script/measure_vae_ceiling.py --input_dir Z:/comparison/Urban100/original
"""

import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL


def load_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load image as tensor in [-1, 1] range, shape [1, 3, H, W]."""
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1]
    img_np = img_np * 2.0 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor.to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor in [-1, 1] to numpy in [0, 255] uint8."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = ((img + 1.0) / 2.0).clip(0, 1) * 255.0
    return img.astype(np.uint8)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0**2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two uint8 images (Wang et al. 2004: Gaussian 11x11, sigma=1.5)."""
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(
            img1.astype(np.float64) / 255.0,
            img2.astype(np.float64) / 255.0,
            channel_axis=2,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
        )
    except ImportError:
        # Fallback: simplified SSIM on luminance only
        gray1 = np.mean(img1.astype(np.float64), axis=2)
        gray2 = np.mean(img2.astype(np.float64), axis=2)
        mu1, mu2 = gray1.mean(), gray2.mean()
        sigma1_sq = gray1.var()
        sigma2_sq = gray2.var()
        sigma12 = ((gray1 - mu1) * (gray2 - mu2)).mean()
        C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
        ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim)


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> tuple:
    """Pad tensor height/width to nearest multiple. Returns padded tensor and original size."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, (h, w)


def main():
    parser = argparse.ArgumentParser(description="Measure VAE encode-decode reconstruction ceiling")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing clean images (PNG/JPG)")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2",
                        help="HuggingFace model ID for VAE (default: SD2)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to output CSV file (default: prints summary only)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 for VAE (matches inference conditions)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.float32

    # Load VAE
    print(f"Loading VAE from {args.model_id}...")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()

    # Find images
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"ERROR: No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    results = []
    latent_scale_factor = 0.18215  # Same as in pipeline

    with torch.no_grad():
        for path in tqdm(image_paths, desc="VAE roundtrip"):
            # Load and prepare
            img_tensor = load_image_as_tensor(path, device).to(dtype)
            img_tensor_padded, (orig_h, orig_w) = pad_to_multiple(img_tensor, multiple=8)

            # Encode
            posterior = vae.encode(img_tensor_padded)
            latent = posterior.latent_dist.mean * latent_scale_factor

            # Decode
            latent_scaled = latent / latent_scale_factor
            reconstructed = vae.decode(latent_scaled).sample

            # Crop back to original size
            reconstructed = reconstructed[:, :, :orig_h, :orig_w]
            img_tensor = img_tensor[:, :, :orig_h, :orig_w]

            # Convert to numpy uint8
            orig_np = tensor_to_numpy(img_tensor.float())
            recon_np = tensor_to_numpy(reconstructed.float())

            # Compute metrics
            psnr = compute_psnr(orig_np, recon_np)
            ssim = compute_ssim(orig_np, recon_np)

            results.append({
                "filename": os.path.basename(path),
                "psnr": psnr,
                "ssim": ssim,
            })

    # Summary statistics
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]

    print("\n" + "=" * 60)
    print("VAE RECONSTRUCTION CEILING (encode → decode, no diffusion)")
    print("=" * 60)
    print(f"Model:    {args.model_id}")
    print(f"Dtype:    {'FP16' if args.fp16 else 'FP32'}")
    print(f"Images:   {len(results)}")
    print(f"Input:    {args.input_dir}")
    print("-" * 60)
    print(f"PSNR:     {np.mean(psnrs):.2f} dB  (std: {np.std(psnrs):.2f})")
    print(f"SSIM:     {np.mean(ssims):.4f}    (std: {np.std(ssims):.4f})")
    print(f"Min PSNR: {np.min(psnrs):.2f} dB")
    print(f"Max PSNR: {np.max(psnrs):.2f} dB")
    print("=" * 60)

    # Save CSV if requested
    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(results)
            # Add summary row
            writer.writerow({
                "filename": "MEAN",
                "psnr": f"{np.mean(psnrs):.4f}",
                "ssim": f"{np.mean(ssims):.6f}",
            })
        print(f"\nResults saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
