"""
Analyze ControlNet contribution as a function of degradation severity.

Compares "Ours" (8ch + ControlNet) vs "Ours w/o CN" (8ch only) across
severity bins, showing how the ControlNet's contribution varies with
degradation intensity.

Images are binned into terciles by degraded-input PSNR (proxy for severity).
For each bin, the delta (Ours - Ours_w/o_CN) is computed on PSNR and LPIPS.

Usage:
    python script/controlnet_contribution_by_severity.py --base_dir comparison
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats


# Configuration
MODEL_FULL = "hybrid_002_re_015000_prediction_0_s05_e01_ddim"
MODEL_NO_CN = "controlnet_4ch_004_re_015000_prediction_0_s05_e01_ddim"
MODEL_DEGRADED = "baseline_degraded"

DATASETS = ["DIV2K", "Urban100", "BSDS100"]

DATASET_PATHS = {
    "DIV2K": "DIV2K",
    "Urban100": "Urban100",
    "BSDS100": "BSDS100",
}

N_BINS = 3
BIN_LABELS = ["Mild", "Moderate", "Severe"]


def load_csv(base_dir, dataset, model_key):
    """Load per-image metrics CSV."""
    subdir = DATASET_PATHS[dataset]
    path = os.path.join(base_dir, subdir, "metrics", f"metrics_{model_key}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: Not found: {path}")
        return None
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="comparison")
    parser.add_argument("--output_dir", type=str, default="comparison/statistical_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    print("=" * 80)
    print("CONTROLNET CONTRIBUTION BY DEGRADATION SEVERITY")
    print("=" * 80)
    print()
    print("Delta = Ours(8ch+CN) - Ours_w/o_CN(8ch only)")
    print("Positive ΔPSNR = ControlNet helps fidelity")
    print("Negative ΔLPIPS = ControlNet helps perception (lower is better)")
    print()

    for dataset in DATASETS:
        df_full = load_csv(args.base_dir, dataset, MODEL_FULL)
        df_nocn = load_csv(args.base_dir, dataset, MODEL_NO_CN)
        df_deg = load_csv(args.base_dir, dataset, MODEL_DEGRADED)

        if df_full is None or df_nocn is None or df_deg is None:
            print(f"  Skipping {dataset}: missing data")
            continue

        # Merge all three on filename
        merged = pd.merge(
            df_deg[["filename", "psnr"]].rename(columns={"psnr": "deg_psnr"}),
            df_full[["filename", "psnr", "ssim", "lpips"]].rename(
                columns={"psnr": "full_psnr", "ssim": "full_ssim", "lpips": "full_lpips"}),
            on="filename"
        )
        merged = pd.merge(
            merged,
            df_nocn[["filename", "psnr", "ssim", "lpips"]].rename(
                columns={"psnr": "nocn_psnr", "ssim": "nocn_ssim", "lpips": "nocn_lpips"}),
            on="filename"
        )

        # Compute deltas (ControlNet contribution)
        merged["delta_psnr"] = merged["full_psnr"] - merged["nocn_psnr"]
        merged["delta_ssim"] = merged["full_ssim"] - merged["nocn_ssim"]
        merged["delta_lpips"] = merged["full_lpips"] - merged["nocn_lpips"]

        # Bin by degraded PSNR (higher deg_psnr = milder degradation)
        merged["bin"] = pd.qcut(merged["deg_psnr"], N_BINS, labels=BIN_LABELS)

        print(f"{'─' * 70}")
        print(f"  {dataset} (N={len(merged)})")
        print(f"{'─' * 70}")
        print(f"  {'Bin':<10} {'N':>4} {'Deg PSNR range':>20} "
              f"{'ΔPSNR':>8} {'ΔSSIM':>8} {'ΔLPIPS':>8} "
              f"{'p(PSNR)':>10}")
        print(f"  {'─'*10} {'─'*4} {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

        for bin_label in BIN_LABELS:
            subset = merged[merged["bin"] == bin_label]
            n = len(subset)
            deg_range = f"[{subset['deg_psnr'].min():.1f}, {subset['deg_psnr'].max():.1f}]"

            dp_mean = subset["delta_psnr"].mean()
            ds_mean = subset["delta_ssim"].mean()
            dl_mean = subset["delta_lpips"].mean()

            # Wilcoxon on the per-bin PSNR difference
            try:
                _, p_psnr = stats.wilcoxon(subset["full_psnr"], subset["nocn_psnr"])
            except ValueError:
                p_psnr = 1.0

            sig = "***" if p_psnr < 0.001 else "**" if p_psnr < 0.01 else "*" if p_psnr < 0.05 else "ns"

            print(f"  {bin_label:<10} {n:>4} {deg_range:>20} "
                  f"{dp_mean:>+8.3f} {ds_mean:>+8.4f} {dl_mean:>+8.4f} "
                  f"{p_psnr:>8.2e} {sig}")

            all_results.append({
                "dataset": dataset,
                "bin": bin_label,
                "n": n,
                "deg_psnr_min": subset["deg_psnr"].min(),
                "deg_psnr_max": subset["deg_psnr"].max(),
                "delta_psnr_mean": dp_mean,
                "delta_psnr_std": subset["delta_psnr"].std(),
                "delta_ssim_mean": ds_mean,
                "delta_ssim_std": subset["delta_ssim"].std(),
                "delta_lpips_mean": dl_mean,
                "delta_lpips_std": subset["delta_lpips"].std(),
                "p_psnr": p_psnr,
            })

        # Overall
        dp_all = merged["delta_psnr"].mean()
        dl_all = merged["delta_lpips"].mean()
        print(f"  {'Overall':<10} {len(merged):>4} {'':>20} "
              f"{dp_all:>+8.3f} {'':>8} {dl_all:>+8.4f}")

        # Trend test: is the ControlNet contribution correlated with severity?
        # Spearman correlation between deg_psnr and delta_psnr
        r_psnr, p_corr_psnr = stats.spearmanr(merged["deg_psnr"], merged["delta_psnr"])
        r_lpips, p_corr_lpips = stats.spearmanr(merged["deg_psnr"], merged["delta_lpips"])

        print(f"\n  Severity correlation (Spearman):")
        print(f"    ΔPSNR vs deg_severity: r={r_psnr:+.3f}, p={p_corr_psnr:.4f}")
        print(f"    ΔLPIPS vs deg_severity: r={r_lpips:+.3f}, p={p_corr_lpips:.4f}")
        print(f"    (positive r for ΔPSNR = CN helps MORE on mild images)")
        print(f"    (negative r for ΔLPIPS = CN helps perception MORE on mild images)")
        print()

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = os.path.join(args.output_dir, "controlnet_contribution_by_severity.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # LaTeX fragment
    print("\n" + "=" * 80)
    print("LATEX TABLE FRAGMENT")
    print("=" * 80)
    print(r"""
\begin{table}[t]
  \centering
  \caption[ControlNet contribution by degradation severity]{%
    ControlNet contribution ($\Delta$ = Ours $-$ Ours w/o CN) by degradation
    severity bin on DIV2K. Positive $\Delta$PSNR indicates the ControlNet
    improves fidelity; negative $\Delta$LPIPS indicates it improves perceptual quality.%
  }
  \label{tab:cn-contribution-severity}
  \small
  \begin{tabular}{@{} l S[table-format=1.3] S[table-format=1.4] S[table-format=-1.4] @{}}
    \toprule
    Severity bin & {$\Delta$PSNR (dB)} & {$\Delta$SSIM} & {$\Delta$LPIPS} \\
    \midrule""")

    for r in all_results:
        if r["dataset"] == "DIV2K":
            print(f"    {r['bin']:<10} & {r['delta_psnr_mean']:+.3f} "
                  f"& {r['delta_ssim_mean']:+.4f} & {r['delta_lpips_mean']:+.4f} \\\\")

    print(r"""    \bottomrule
  \end{tabular}
\end{table}
""")


if __name__ == "__main__":
    main()
