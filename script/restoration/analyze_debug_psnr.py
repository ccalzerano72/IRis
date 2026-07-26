# Analyze PSNR of debug step outputs against clean image
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from PIL import Image
from pathlib import Path
import glob

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)

def main():
    debug_dir = "output/debug_steps"
    clean_path = "external/test_images/DIV2K_valid_512/clean/0803.png"
    degraded_path = "external/test_images/DIV2K_valid_512/degraded_multi/0803.png"
    
    # Load reference images
    clean = np.array(Image.open(clean_path).convert("RGB"))
    degraded = np.array(Image.open(degraded_path).convert("RGB"))
    
    print("=" * 70)
    print("PSNR ANALYSIS vs CLEAN IMAGE")
    print("=" * 70)
    print(f"Clean image: {clean_path}")
    print(f"Degraded input: {degraded_path}")
    print()
    
    # PSNR of degraded vs clean (baseline)
    psnr_degraded = calculate_psnr(degraded, clean)
    print(f"Degraded input vs Clean:  {psnr_degraded:.2f} dB  (baseline)")
    print("-" * 70)
    
    # Find all pred_x0 images
    pred_x0_files = sorted(glob.glob(os.path.join(debug_dir, "step_*_pred_x0.png")))
    
    print("\nPSNR of pred_x0 at each step vs Clean:")
    print("-" * 70)
    
    for f in pred_x0_files:
        img = np.array(Image.open(f).convert("RGB"))
        psnr = calculate_psnr(img, clean)
        step_name = os.path.basename(f).replace("_pred_x0.png", "")
        delta = psnr - psnr_degraded
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        print(f"  {step_name}: {psnr:.2f} dB  ({delta_str} vs degraded)")
    
    # Final result
    final_path = os.path.join(debug_dir, "99_final.png")
    if os.path.exists(final_path):
        final = np.array(Image.open(final_path).convert("RGB"))
        psnr_final = calculate_psnr(final, clean)
        delta = psnr_final - psnr_degraded
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        print("-" * 70)
        print(f"  FINAL (99_final): {psnr_final:.2f} dB  ({delta_str} vs degraded)")
    
    # Also check step_XX_tXXXX.png (the actual latent at each step, not pred_x0)
    print("\n" + "=" * 70)
    print("PSNR of target_latent (decoded) at each step vs Clean:")
    print("-" * 70)
    
    step_files = sorted([f for f in glob.glob(os.path.join(debug_dir, "step_*.png")) 
                        if "pred_x0" not in f])
    
    for f in step_files:
        img = np.array(Image.open(f).convert("RGB"))
        psnr = calculate_psnr(img, clean)
        step_name = os.path.basename(f).replace(".png", "")
        delta = psnr - psnr_degraded
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        print(f"  {step_name}: {psnr:.2f} dB  ({delta_str} vs degraded)")

if __name__ == "__main__":
    main()
