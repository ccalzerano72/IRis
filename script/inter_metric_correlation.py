"""
Inter-metric Spearman correlation analysis for thesis Section 5.4.

Computes:
1. Correlation matrix between all 9 metrics on IRis predictions (per-image)
2. Divergence analysis: when PSNR and LPIPS disagree, what do NR metrics do?
3. Cross-model: correlation between metric deltas (IRis - competitor)

Output:
- Console summary with key findings
- CSV: comparison/statistical_analysis/inter_metric_correlation.csv
- LaTeX table fragment: comparison/statistical_analysis/inter_metric_correlation.tex

Usage:
    python script/inter_metric_correlation.py
    python script/inter_metric_correlation.py --dataset Urban100
    python script/inter_metric_correlation.py --all
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "comparison")
OUTPUT_DIR = os.path.join(BASE_DIR, "statistical_analysis")

METRICS = ["psnr", "ssim", "lpips", "delta_e", "arniqa", "brisque", "niqe", "maniqa", "musiq"]

# Metric display names for LaTeX
METRIC_NAMES = {
    "psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS", "delta_e": r"$\Delta E$",
    "arniqa": "ARNIQA", "brisque": "BRISQUE", "niqe": "NIQE",
    "maniqa": "MANIQA", "musiq": "MUSIQ",
}

# Model CSV filenames
MODELS = {
    "IRis": "metrics_hybrid_002_re_015000_prediction_0_s05_e01_ddim.csv",
    "Restormer": "metrics_restormer_real_denoising.csv",
    "DiffBIR": "metrics_diffbir_steps25_strength1.0_up1_nocapt.csv",
    "HyPIR": "metrics_hypir_sd2_up1.csv",
    "RealESRGAN": "metrics_realesrgan_x4plus_s4.csv",
}

DATASETS = {
    "DIV2K": os.path.join(BASE_DIR, "DIV2K", "metrics"),
    "Urban100": os.path.join(BASE_DIR, "Urban100", "metrics"),
    "BSDS100": os.path.join(BASE_DIR, "BSDS100", "metrics"),
}


# ============================================================================
# Analysis functions
# ============================================================================

def load_model_data(dataset_path, model_file):
    """Load per-image metrics for a model."""
    fpath = os.path.join(dataset_path, model_file)
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    if "filename" in df.columns:
        df = df.set_index("filename")
    return df


def compute_correlation_matrix(df, metrics=METRICS):
    """Compute Spearman correlation matrix with p-values."""
    data = df[metrics].dropna()
    n = len(metrics)
    
    rho_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                rho_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                rho, pval = spearmanr(data[metrics[i]], data[metrics[j]])
                rho_matrix[i, j] = rho
                pval_matrix[i, j] = pval
    
    rho_df = pd.DataFrame(rho_matrix, index=metrics, columns=metrics)
    pval_df = pd.DataFrame(pval_matrix, index=metrics, columns=metrics)
    
    return rho_df, pval_df


def analyze_divergence(df, metrics=METRICS):
    """Analyze when PSNR and perceptual metrics diverge."""
    data = df[metrics].dropna()
    
    findings = []
    
    # PSNR vs LPIPS correlation
    rho_psnr_lpips, p = spearmanr(data["psnr"], data["lpips"])
    findings.append(f"  PSNR vs LPIPS: rho={rho_psnr_lpips:.3f} (p={p:.2e})")
    
    # PSNR vs ARNIQA
    rho_psnr_arniqa, p = spearmanr(data["psnr"], data["arniqa"])
    findings.append(f"  PSNR vs ARNIQA: rho={rho_psnr_arniqa:.3f} (p={p:.2e})")
    
    # LPIPS vs ARNIQA
    rho_lpips_arniqa, p = spearmanr(data["lpips"], data["arniqa"])
    findings.append(f"  LPIPS vs ARNIQA: rho={rho_lpips_arniqa:.3f} (p={p:.2e})")
    
    # FR metrics internal consistency
    rho_psnr_ssim, p = spearmanr(data["psnr"], data["ssim"])
    findings.append(f"  PSNR vs SSIM: rho={rho_psnr_ssim:.3f} (p={p:.2e})")
    
    rho_psnr_de, p = spearmanr(data["psnr"], data["delta_e"])
    findings.append(f"  PSNR vs Delta_E: rho={rho_psnr_de:.3f} (p={p:.2e})")
    
    # NR metrics internal consistency
    rho_arniqa_musiq, p = spearmanr(data["arniqa"], data["musiq"])
    findings.append(f"  ARNIQA vs MUSIQ: rho={rho_arniqa_musiq:.3f} (p={p:.2e})")
    
    rho_brisque_niqe, p = spearmanr(data["brisque"], data["niqe"])
    findings.append(f"  BRISQUE vs NIQE: rho={rho_brisque_niqe:.3f} (p={p:.2e})")
    
    # Statistical vs learned NR
    rho_niqe_arniqa, p = spearmanr(data["niqe"], data["arniqa"])
    findings.append(f"  NIQE vs ARNIQA: rho={rho_niqe_arniqa:.3f} (p={p:.2e})")
    
    return findings


def compute_delta_correlations(dataset_path, ours_file, competitor_file, competitor_name):
    """Compute correlations between per-image deltas across metrics."""
    ours = load_model_data(dataset_path, ours_file)
    comp = load_model_data(dataset_path, competitor_file)
    
    if ours is None or comp is None:
        return None
    
    # Align on common images
    common = ours.index.intersection(comp.index)
    if len(common) < 10:
        return None
    
    deltas = pd.DataFrame(index=common)
    for m in METRICS:
        if m in ours.columns and m in comp.columns:
            # For "lower is better" metrics, negate so positive = IRis better
            if m in ["lpips", "delta_e", "brisque", "niqe"]:
                deltas[f"d_{m}"] = comp.loc[common, m] - ours.loc[common, m]
            else:
                deltas[f"d_{m}"] = ours.loc[common, m] - comp.loc[common, m]
    
    # Key correlation: does PSNR advantage predict LPIPS advantage?
    if "d_psnr" in deltas and "d_lpips" in deltas:
        rho, p = spearmanr(deltas["d_psnr"], deltas["d_lpips"])
        return {
            "competitor": competitor_name,
            "n": len(common),
            "rho_dpsnr_dlpips": rho,
            "p_dpsnr_dlpips": p,
        }
    return None


def generate_latex_table(rho_df, pval_df, dataset_name):
    """Generate a LaTeX correlation matrix (upper triangle only)."""
    n = len(METRICS)
    lines = []
    lines.append(f"% Inter-metric Spearman correlation on {dataset_name} (IRis s5e1)")
    lines.append(f"% Values shown for |rho| > 0.3; bold for |rho| > 0.7")
    lines.append(r"\begin{tabular}{@{} l " + "r " * n + "@{}}")
    lines.append(r"  \toprule")
    
    # Header
    header = "  "
    for m in METRICS:
        header += f" & {METRIC_NAMES[m]}"
    header += r" \\"
    lines.append(header)
    lines.append(r"  \midrule")
    
    # Rows (upper triangle only)
    for i, m_row in enumerate(METRICS):
        row = f"  {METRIC_NAMES[m_row]}"
        for j, m_col in enumerate(METRICS):
            if j <= i:
                row += " & "  # empty below diagonal
            else:
                rho = rho_df.iloc[i, j]
                pval = pval_df.iloc[i, j]
                if abs(rho) < 0.3 or pval > 0.05:
                    row += " & {\\footnotesize ---}"
                elif abs(rho) > 0.7:
                    row += f" & \\textbf{{{rho:+.2f}}}"
                else:
                    row += f" & {rho:+.2f}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def run_analysis(dataset_name, dataset_path):
    """Run full analysis for one dataset."""
    print(f"\n{'='*70}")
    print(f"  INTER-METRIC CORRELATION ANALYSIS: {dataset_name}")
    print(f"{'='*70}")
    
    # Load IRis data
    ours = load_model_data(dataset_path, MODELS["IRis"])
    if ours is None:
        print(f"  ERROR: IRis metrics not found at {dataset_path}")
        return None
    
    print(f"\n  Images: {len(ours)}")
    
    # 1. Correlation matrix
    print(f"\n  --- Spearman Correlation Matrix (IRis predictions) ---")
    rho_df, pval_df = compute_correlation_matrix(ours)
    
    # Print compact upper triangle
    print(f"\n  {'':12s}", end="")
    for m in METRICS:
        print(f" {m[:6]:>6s}", end="")
    print()
    for i, m_row in enumerate(METRICS):
        print(f"  {m_row:12s}", end="")
        for j, m_col in enumerate(METRICS):
            if j <= i:
                print(f" {'':>6s}", end="")
            else:
                rho = rho_df.iloc[i, j]
                pval = pval_df.iloc[i, j]
                marker = "*" if pval < 0.01 else " " if pval < 0.05 else "n"
                print(f" {rho:+.2f}{marker}", end="")
        print()
    
    # 2. Key divergence findings
    print(f"\n  --- Key Relationships ---")
    findings = analyze_divergence(ours)
    for f in findings:
        print(f)
    
    # 3. Cross-model delta correlations
    print(f"\n  --- Delta Correlations (does PSNR advantage predict LPIPS advantage?) ---")
    for comp_name, comp_file in MODELS.items():
        if comp_name == "IRis":
            continue
        result = compute_delta_correlations(dataset_path, MODELS["IRis"], comp_file, comp_name)
        if result:
            print(f"  vs {result['competitor']:12s}: rho(dPSNR, dLPIPS)={result['rho_dpsnr_dlpips']:+.3f} "
                  f"(p={result['p_dpsnr_dlpips']:.2e}, n={result['n']})")
    
    # Generate LaTeX
    latex = generate_latex_table(rho_df, pval_df, dataset_name)
    
    return {
        "dataset": dataset_name,
        "rho": rho_df,
        "pval": pval_df,
        "latex": latex,
    }


def main():
    parser = argparse.ArgumentParser(description="Inter-metric Spearman correlation analysis")
    parser.add_argument("--dataset", type=str, default="DIV2K",
                        choices=list(DATASETS.keys()),
                        help="Dataset to analyze (default: DIV2K)")
    parser.add_argument("--all", action="store_true",
                        help="Run analysis on all datasets")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets_to_run = DATASETS if args.all else {args.dataset: DATASETS[args.dataset]}
    
    all_results = []
    for name, path in datasets_to_run.items():
        result = run_analysis(name, path)
        if result:
            all_results.append(result)
    
    # Save outputs
    if all_results:
        # Save correlation matrix CSV (first dataset)
        r = all_results[0]
        csv_path = os.path.join(args.output_dir, "inter_metric_correlation.csv")
        r["rho"].to_csv(csv_path)
        print(f"\n  Correlation matrix saved to: {csv_path}")
        
        # Save LaTeX fragments
        tex_path = os.path.join(args.output_dir, "inter_metric_correlation.tex")
        with open(tex_path, "w") as f:
            for r in all_results:
                f.write(f"\n% === {r['dataset']} ===\n")
                f.write(r["latex"])
                f.write("\n\n")
        print(f"  LaTeX table saved to: {tex_path}")
    
    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
