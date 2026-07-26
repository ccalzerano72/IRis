"""
Analyze both conditioning pathway contributions as a function of degradation severity.

Decomposes the dual-conditioning architecture into:
- ControlNet contribution: Ours(8ch+CN) - Ours_w/o_CN(8ch only)
- Latent concat contribution: Ours(8ch+CN) - 4ch+CN(CN only)

Images are binned into terciles by degraded-input PSNR (proxy for severity).

Usage:
    python script/conditioning_contribution_by_severity.py --base_dir comparison
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats


# Configuration
MODEL_FULL = "hybrid_002_re_015000_prediction_0_s05_e01_ddim"       # 8ch + CN
MODEL_NO_CN = "hybrid_re_ablation_cn_frozen_015000_prediction_0_s05_e01_ddim"    # Frozen-CN (clean ablation)
MODEL_4CH_CN = "controlnet_4ch_004_re_015000_prediction_0_s05_e01_ddim"  # 4ch + CN
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

    print("=" * 90)
    print("DUAL CONDITIONING: CONTRIBUTION BY DEGRADATION SEVERITY")
    print("=" * 90)
    print()
    print("ControlNet contribution  = Ours(8ch+CN) - 8ch_only       [what CN adds to latent concat]")
    print("Latent concat contribution = Ours(8ch+CN) - 4ch+CN       [what 8ch adds to ControlNet]")
    print()
    print("Positive ΔPSNR = pathway helps fidelity")
    print("Negative ΔLPIPS = pathway helps perception")
    print()

    for dataset in DATASETS:
        df_full = load_csv(args.base_dir, dataset, MODEL_FULL)
        df_nocn = load_csv(args.base_dir, dataset, MODEL_NO_CN)
        df_4ch_cn = load_csv(args.base_dir, dataset, MODEL_4CH_CN)
        df_deg = load_csv(args.base_dir, dataset, MODEL_DEGRADED)

        if any(d is None for d in [df_full, df_nocn, df_4ch_cn, df_deg]):
            print(f"  Skipping {dataset}: missing data")
            continue

        # Merge all on filename
        merged = df_deg[["filename", "psnr"]].rename(columns={"psnr": "deg_psnr"})
        for df, prefix in [(df_full, "full"), (df_nocn, "nocn"), (df_4ch_cn, "4chcn")]:
            merged = pd.merge(
                merged,
                df[["filename", "psnr", "ssim", "lpips"]].rename(
                    columns={"psnr": f"{prefix}_psnr", "ssim": f"{prefix}_ssim",
                             "lpips": f"{prefix}_lpips"}),
                on="filename"
            )

        # Compute deltas
        # ControlNet contribution (what CN adds when you already have 8ch)
        merged["cn_delta_psnr"] = merged["full_psnr"] - merged["nocn_psnr"]
        merged["cn_delta_lpips"] = merged["full_lpips"] - merged["nocn_lpips"]
        merged["cn_delta_ssim"] = merged["full_ssim"] - merged["nocn_ssim"]

        # Latent concat contribution (what 8ch adds when you already have CN)
        merged["lat_delta_psnr"] = merged["full_psnr"] - merged["4chcn_psnr"]
        merged["lat_delta_lpips"] = merged["full_lpips"] - merged["4chcn_lpips"]
        merged["lat_delta_ssim"] = merged["full_ssim"] - merged["4chcn_ssim"]

        # Bin by degraded PSNR
        merged["bin"] = pd.qcut(merged["deg_psnr"], N_BINS, labels=BIN_LABELS)

        print(f"{'═' * 90}")
        print(f"  {dataset} (N={len(merged)})")
        print(f"{'═' * 90}")
        print()
        print(f"  {'':10} {'── ControlNet contrib ──':>30}   {'── Latent concat contrib ──':>30}")
        print(f"  {'Bin':<10} {'ΔPSNR':>8} {'ΔLPIPS':>8} {'ΔSSIM':>8}   {'ΔPSNR':>8} {'ΔLPIPS':>8} {'ΔSSIM':>8}")
        print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8}   {'─'*8} {'─'*8} {'─'*8}")

        for bin_label in BIN_LABELS:
            subset = merged[merged["bin"] == bin_label]

            cn_dp = subset["cn_delta_psnr"].mean()
            cn_dl = subset["cn_delta_lpips"].mean()
            cn_ds = subset["cn_delta_ssim"].mean()

            lat_dp = subset["lat_delta_psnr"].mean()
            lat_dl = subset["lat_delta_lpips"].mean()
            lat_ds = subset["lat_delta_ssim"].mean()

            print(f"  {bin_label:<10} {cn_dp:>+8.3f} {cn_dl:>+8.4f} {cn_ds:>+8.4f}"
                  f"   {lat_dp:>+8.3f} {lat_dl:>+8.4f} {lat_ds:>+8.4f}")

            all_results.append({
                "dataset": dataset,
                "bin": bin_label,
                "n": len(subset),
                "cn_delta_psnr": cn_dp,
                "cn_delta_lpips": cn_dl,
                "cn_delta_ssim": cn_ds,
                "lat_delta_psnr": lat_dp,
                "lat_delta_lpips": lat_dl,
                "lat_delta_ssim": lat_ds,
            })

        # Overall
        print(f"  {'Overall':<10} {merged['cn_delta_psnr'].mean():>+8.3f} "
              f"{merged['cn_delta_lpips'].mean():>+8.4f} {merged['cn_delta_ssim'].mean():>+8.4f}"
              f"   {merged['lat_delta_psnr'].mean():>+8.3f} "
              f"{merged['lat_delta_lpips'].mean():>+8.4f} {merged['lat_delta_ssim'].mean():>+8.4f}")

        # Spearman correlations with severity
        print(f"\n  Spearman correlation with degradation severity (deg_psnr):")
        for name, col in [("CN ΔPSNR", "cn_delta_psnr"), ("CN ΔLPIPS", "cn_delta_lpips"),
                          ("Lat ΔPSNR", "lat_delta_psnr"), ("Lat ΔLPIPS", "lat_delta_lpips")]:
            r, p = stats.spearmanr(merged["deg_psnr"], merged[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {name:<12}: r={r:+.3f}, p={p:.4f} {sig}")

        print()

    # Save
    results_df = pd.DataFrame(all_results)
    out_path = os.path.join(args.output_dir, "conditioning_contribution_by_severity.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # LaTeX
    print("\n" + "=" * 90)
    print("LATEX TABLE")
    print("=" * 90)
    print(r"""
\begin{table}[t]
  \centering
  \caption[Conditioning pathway contributions by severity]{%
    Per-pathway contribution by degradation severity on DIV2K.
    ControlNet contribution = Ours $-$ Ours w/o CN (isolates what the ControlNet adds).
    Latent concat contribution = Ours $-$ 4ch+CN (isolates what the 8-channel input adds).
    Positive $\Delta$PSNR = improves fidelity; negative $\Delta$LPIPS = improves perception.%
  }
  \label{tab:pathway-contribution-severity}
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{} l S[table-format=+1.3] S[table-format=+1.4] S[table-format=+1.4]
                      S[table-format=+1.3] S[table-format=+1.4] S[table-format=+1.4] @{}}
    \toprule
    & \multicolumn{3}{c}{\textbf{ControlNet contribution}} & \multicolumn{3}{c}{\textbf{Latent concat contribution}} \\
    \cmidrule(lr){2-4} \cmidrule(lr){5-7}
    Severity & {$\Delta$PSNR} & {$\Delta$LPIPS} & {$\Delta$SSIM}
             & {$\Delta$PSNR} & {$\Delta$LPIPS} & {$\Delta$SSIM} \\
    \midrule""")

    for r in all_results:
        if r["dataset"] == "DIV2K":
            print(f"    {r['bin']:<10} & {r['cn_delta_psnr']:+.3f} & {r['cn_delta_lpips']:+.4f} "
                  f"& {r['cn_delta_ssim']:+.4f} & {r['lat_delta_psnr']:+.3f} "
                  f"& {r['lat_delta_lpips']:+.4f} & {r['lat_delta_ssim']:+.4f} \\\\")

    print(r"""    \bottomrule
  \end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
