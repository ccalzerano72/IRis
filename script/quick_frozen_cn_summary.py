"""Quick summary of frozen-CN ablation results."""
import pandas as pd
import os

base = "comparison"
model = "hybrid_re_ablation_cn_frozen_015000_prediction_0_s05_e01_ddim"
datasets = {
    "DIV2K": "DIV2K",
    "Urban100": "Urban100",
    "BSDS100": "BSDS100",
    "BSDS100_UP": "BSDS100_UPSCALED",
}

# Also load the full model for comparison
full_model = "hybrid_002_re_015000_prediction_0_s05_e01_ddim"

print("=" * 70)
print("FROZEN-CN ABLATION: Clean single-variable comparison")
print("=" * 70)
print()
print(f"{'Dataset':<12} {'Variant':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'DeltaE':>8}")
print("-" * 70)

for name, subdir in datasets.items():
    frozen_path = os.path.join(base, subdir, "metrics", f"metrics_{model}.csv")
    full_path = os.path.join(base, subdir, "metrics", f"metrics_{full_model}.csv")
    
    if not os.path.exists(frozen_path):
        print(f"  {name}: frozen-CN CSV not found")
        continue
    
    df_frozen = pd.read_csv(frozen_path)
    
    print(f"{name:<12} {'Frozen-CN':<12} {df_frozen.psnr.mean():>8.2f} "
          f"{df_frozen.ssim.mean():>8.4f} {df_frozen.lpips.mean():>8.4f} "
          f"{df_frozen.delta_e.mean():>8.2f}")
    
    if os.path.exists(full_path):
        df_full = pd.read_csv(full_path)
        print(f"{'':12} {'Full model':<12} {df_full.psnr.mean():>8.2f} "
              f"{df_full.ssim.mean():>8.4f} {df_full.lpips.mean():>8.4f} "
              f"{df_full.delta_e.mean():>8.2f}")
        
        dp = df_full.psnr.mean() - df_frozen.psnr.mean()
        ds = df_full.ssim.mean() - df_frozen.ssim.mean()
        dl = df_full.lpips.mean() - df_frozen.lpips.mean()
        print(f"{'':12} {'Delta':<12} {dp:>+8.2f} {ds:>+8.4f} {dl:>+8.4f}")
    print()

print("=" * 70)
print("Delta = Full model - Frozen-CN = PURE ControlNet contribution")
print("(same loss, same batch, same everything except CN trainability)")
print("=" * 70)
