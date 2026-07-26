#!/usr/bin/env python3
"""
ARNIQA Feature Space Analysis: Degradation Type Clustering

Preliminary analysis to determine whether ARNIQA's internal representations
(Layer 4 global features, 2048-dim) separate by degradation TYPE or only
by degradation SEVERITY.

This answers the question: "If we use ARNIQA Stage 1 global features as
conditioning for the U-Net, will the signal carry degradation-type information
that the U-Net can leverage?"

Usage:
    python script/restoration/analyze_arniqa_degradation_clusters.py \
        --input_dir /path/to/clean/images \
        --output_dir output/arniqa_cluster_analysis \
        --num_images 50 \
        --degradation_types whitenoise,jpeg,gaublur

Output:
    - cluster_analysis.png: 4-panel figure (PCA, t-SNE, cosine sim, stats)
    - embeddings.csv: raw embeddings with metadata for further analysis
    - analysis_report.txt: numerical summary and verdict
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ARNIQA.degradation import ImageDistorter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ImageNet normalization constants
# Verified from src/ARNIQA/model.py lines 23-24
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_arniqa_encoder(device: torch.device):
    """
    Load ARNIQA encoder (ResNet-50 backbone) from torch.hub.
    
    Pattern copied from src/ARNIQA/model.py ArniqaEncoder.__init__ (line 120):
        arniqa_model = torch.hub.load(
            repo_or_dir="miccunifi/ARNIQA",
            model="ARNIQA",
            regressor_dataset="kadid10k",
        )
        self.encoder = arniqa_model.encoder
    
    Returns:
        encoder.model (nn.Sequential): The ResNet-50 sequential model
        global_pool (nn.AdaptiveAvgPool2d): Global average pooling layer
    """
    logger.info("Loading ARNIQA model from torch.hub...")
    arniqa_model = torch.hub.load(
        repo_or_dir="miccunifi/ARNIQA",
        model="ARNIQA",
        regressor_dataset="kadid10k",
    )
    encoder_model = arniqa_model.encoder.model
    encoder_model.eval()
    encoder_model.to(device)
    
    global_pool = nn.AdaptiveAvgPool2d(1).to(device)
    
    logger.info("ARNIQA encoder loaded successfully")
    return encoder_model, global_pool


def normalize_input(rgb_01: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Apply ImageNet normalization to input tensor.
    
    Pattern copied from src/ARNIQA/model.py ArniqaEncoder._normalize_input (lines 237-243):
        rgb_01 = (rgb_in + 1.0) / 2.0
        rgb_norm = (rgb_01 - self.mean) / self.std
    
    Here input is already in [0, 1] range (from ImageDistorter), so we skip
    the [-1,1] -> [0,1] conversion.
    
    Args:
        rgb_01: Image tensor in [0, 1] range, shape [B, 3, H, W]
        device: Target device
    
    Returns:
        ImageNet-normalized tensor
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (rgb_01 - mean) / std


@torch.no_grad()
def extract_global_features(
    encoder_model: nn.Sequential,
    global_pool: nn.AdaptiveAvgPool2d,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Extract Layer 4 global features (2048-dim) from a single image.
    
    Pattern copied from src/ARNIQA/model.py ArniqaEncoder.forward_features (lines 270-295):
        x = model[0](x)  # conv1
        x = model[1](x)  # bn1
        x = model[2](x)  # relu
        x = model[3](x)  # maxpool
        x = model[4](x)  # layer1
        x = model[5](x)  # layer2
        layer3_out = model[6](x)  # [B, 1024, H/16, W/16]
        layer4_out = model[7](layer3_out)  # [B, 2048, H/32, W/32]
        global_feat = self.global_pool(layer4_out)  # [B, 2048, 1, 1]
        global_feat = global_feat.flatten(1)  # [B, 2048]
    
    Args:
        encoder_model: ARNIQA ResNet-50 sequential model
        global_pool: AdaptiveAvgPool2d(1)
        image_tensor: Image in [0, 1] range, shape [3, H, W] (single image)
        device: Target device
    
    Returns:
        numpy array of shape [2048]
    """
    # Add batch dimension and normalize
    x = image_tensor.unsqueeze(0).to(device)
    x = normalize_input(x, device)
    
    # Forward through ResNet-50 layers (same order as ArniqaEncoder.forward_features)
    x = encoder_model[0](x)  # conv1
    x = encoder_model[1](x)  # bn1
    x = encoder_model[2](x)  # relu
    x = encoder_model[3](x)  # maxpool
    x = encoder_model[4](x)  # layer1
    x = encoder_model[5](x)  # layer2
    x = encoder_model[6](x)  # layer3
    x = encoder_model[7](x)  # layer4 -> [B, 2048, H/32, W/32]
    
    # Global average pooling
    x = global_pool(x)       # [B, 2048, 1, 1]
    x = x.flatten(1)         # [B, 2048]
    
    return x.cpu().numpy()[0]


def load_clean_images(input_dir: str, num_images: int) -> list:
    """
    Load clean images from directory.
    
    Args:
        input_dir: Path to directory with clean images
        num_images: Maximum number of images to load
    
    Returns:
        List of (filename, tensor [3, H, W] in [0, 1]) tuples
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])
    
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    # Limit to num_images
    image_files = image_files[:num_images]
    logger.info(f"Loading {len(image_files)} clean images from {input_dir}")
    
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = torch.from_numpy(np.array(img)).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            images.append((img_path.name, tensor))
        except Exception as e:
            logger.warning(f"Failed to load {img_path.name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(images)} images")
    return images


def compute_cosine_similarity_matrix(embeddings: np.ndarray, labels: list) -> np.ndarray:
    """
    Compute mean cosine similarity between each pair of degradation classes.
    
    Args:
        embeddings: [N, 2048] feature matrix
        labels: list of N class labels (strings)
    
    Returns:
        Symmetric matrix of shape [num_classes, num_classes] with mean cosine similarities
    """
    # L2 normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    sim_matrix = np.zeros((n_classes, n_classes))
    
    for i, label_i in enumerate(unique_labels):
        mask_i = np.array([l == label_i for l in labels])
        feats_i = normed[mask_i]
        for j, label_j in enumerate(unique_labels):
            mask_j = np.array([l == label_j for l in labels])
            feats_j = normed[mask_j]
            # Mean cosine similarity between all pairs
            cos_sim = feats_i @ feats_j.T
            sim_matrix[i, j] = cos_sim.mean()
    
    return sim_matrix, unique_labels


def create_analysis_figure(
    embeddings: np.ndarray,
    labels_type: list,
    labels_level: list,
    sim_matrix: np.ndarray,
    class_names: list,
    silhouette_type: float,
    silhouette_level: float,
    output_path: str,
):
    """
    Create 4-panel analysis figure.
    
    Panels:
    1. PCA 2D projection colored by degradation type
    2. t-SNE 2D projection colored by degradation type
    3. Cosine similarity heatmap between classes
    4. Summary statistics text panel
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("ARNIQA Layer 4 Global Features: Degradation Type Clustering Analysis", 
                 fontsize=14, fontweight='bold')
    
    # Color map for degradation types
    unique_types = sorted(set(labels_type))
    colors_map = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    type_to_color = {t: colors_map[i] for i, t in enumerate(unique_types)}
    point_colors = [type_to_color[t] for t in labels_type]
    
    # Marker map for severity levels
    level_markers = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'v', -1: '*'}
    
    # --- Panel 1: PCA ---
    ax1 = axes[0, 0]
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)
    
    for deg_type in unique_types:
        mask = np.array([t == deg_type for t in labels_type])
        levels_for_type = np.array(labels_level)[mask]
        for level in sorted(set(levels_for_type)):
            level_mask = levels_for_type == level
            marker = level_markers.get(level, 'o')
            ax1.scatter(
                pca_coords[mask][level_mask, 0],
                pca_coords[mask][level_mask, 1],
                c=[type_to_color[deg_type]],
                marker=marker,
                s=40,
                alpha=0.7,
                label=f"{deg_type} L{level}" if level >= 0 else f"{deg_type}",
            )
    
    ax1.set_title(f"PCA (var explained: {pca.explained_variance_ratio_[:2].sum():.1%})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: t-SNE ---
    ax2 = axes[0, 1]
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    tsne_coords = tsne.fit_transform(embeddings)
    
    for deg_type in unique_types:
        mask = np.array([t == deg_type for t in labels_type])
        levels_for_type = np.array(labels_level)[mask]
        for level in sorted(set(levels_for_type)):
            level_mask = levels_for_type == level
            marker = level_markers.get(level, 'o')
            ax2.scatter(
                tsne_coords[mask][level_mask, 0],
                tsne_coords[mask][level_mask, 1],
                c=[type_to_color[deg_type]],
                marker=marker,
                s=40,
                alpha=0.7,
            )
    
    ax2.set_title("t-SNE")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: Cosine Similarity Heatmap ---
    ax3 = axes[1, 0]
    im = ax3.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.5, vmax=1.0, aspect='auto')
    ax3.set_xticks(range(len(class_names)))
    ax3.set_yticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.set_yticklabels(class_names)
    ax3.set_title("Mean Cosine Similarity Between Classes")
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax3.text(j, i, f"{sim_matrix[i, j]:.3f}", ha='center', va='center', fontsize=8)
    
    fig.colorbar(im, ax=ax3, shrink=0.8)
    
    # --- Panel 4: Summary Statistics ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Compute intra-class vs inter-class similarity
    intra_sims = [sim_matrix[i, i] for i in range(len(class_names))]
    inter_sims = [sim_matrix[i, j] for i in range(len(class_names)) 
                  for j in range(len(class_names)) if i != j]
    
    mean_intra = np.mean(intra_sims)
    mean_inter = np.mean(inter_sims)
    separation = mean_intra - mean_inter
    
    summary_text = (
        f"CLUSTERING ANALYSIS SUMMARY\n"
        f"{'='*40}\n\n"
        f"Total samples: {len(embeddings)}\n"
        f"Degradation types: {', '.join(unique_types)}\n"
        f"Severity levels per type: 5 (0-4)\n\n"
        f"SILHOUETTE SCORES (range: -1 to 1)\n"
        f"  By degradation type: {silhouette_type:.4f}\n"
        f"  By severity level:   {silhouette_level:.4f}\n\n"
        f"COSINE SIMILARITY\n"
        f"  Mean intra-class: {mean_intra:.4f}\n"
        f"  Mean inter-class: {mean_inter:.4f}\n"
        f"  Separation (intra - inter): {separation:.4f}\n\n"
        f"{'='*40}\n"
        f"VERDICT:\n"
    )
    
    if silhouette_type > 0.3 and separation > 0.05:
        verdict = "STRONG type separation.\nARNIQA features cluster by degradation type.\nConditioning approach is VIABLE."
    elif silhouette_type > 0.1 and separation > 0.02:
        verdict = "WEAK type separation.\nSome clustering by type, but noisy.\nConditioning may have LIMITED benefit."
    else:
        verdict = "NO meaningful type separation.\nARNIQA features do NOT discriminate types.\nConditioning will likely be IGNORED by U-Net."
    
    summary_text += verdict
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend for degradation types (separate from level markers)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=type_to_color[t],
                   markersize=10, label=t)
        for t in unique_types
    ]
    # Add level markers to legend
    for level, marker in sorted(level_markers.items()):
        if level >= 0:
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                           markersize=8, label=f"Level {level}")
            )
        else:
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                           markersize=10, label="Clean (original)")
            )
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=min(len(legend_elements), 6),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure saved to {output_path}")


def load_arniqa_scorer(device: torch.device):
    """
    Load ARNIQA quality scorer for computing scalar quality scores.
    
    Pattern copied from script/restoration/eval/04_calculate_metrics.py lines 79-85:
        from torchmetrics.image.arniqa import ARNIQA
        self.arniqa_model = ARNIQA(
            regressor_dataset='kadid10k',
            normalize=True,
            reduction='none'
        )
        self.arniqa_model = self.arniqa_model.to(self.device)
    
    Using kadid10k to match the conditioning encoder
    (verified from src/util/metric.py line 558).
    
    Returns:
        ARNIQA torchmetrics model ready for inference
    """
    logger.info("Loading ARNIQA scorer (torchmetrics)...")
    from torchmetrics.image.arniqa import ARNIQA
    scorer = ARNIQA(
        regressor_dataset='kadid10k',
        normalize=True,
        reduction='none'
    )
    scorer = scorer.to(device)
    scorer.eval()
    logger.info("ARNIQA scorer loaded successfully")
    return scorer


@torch.no_grad()
def compute_arniqa_score(
    scorer,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute scalar ARNIQA quality score for a single image.
    
    Pattern from script/restoration/eval/04_calculate_metrics.py lines 222-226:
        img_tensor = self._pil_to_tensor_01(img_pil).to(self.device)
        with torch.no_grad():
            score = self.arniqa_model(img_tensor)
        return round(float(score.item()), 4)
    
    Args:
        scorer: torchmetrics ARNIQA model
        image_tensor: Image in [0, 1] range, shape [3, H, W]
        device: Target device
    
    Returns:
        Scalar quality score (higher = better quality)
    """
    x = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
    score = scorer(x)
    return float(score.item())


def create_score_vs_level_figure(
    score_data: dict,
    output_path: str,
):
    """
    Create figure showing ARNIQA score vs degradation level.
    
    Args:
        score_data: dict mapping degradation_type -> {level -> [list of scores]}
                    Also contains "clean" -> {-1 -> [list of scores]}
        output_path: Where to save the figure
    """
    from scipy import stats
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ARNIQA Quality Score vs Degradation Level", 
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    deg_types = [k for k in score_data.keys() if k != "clean"]
    
    # --- Panel 1: Score vs Level (lines with error bars) ---
    ax1 = axes[0]
    
    # Plot clean baseline as horizontal band
    if "clean" in score_data and -1 in score_data["clean"]:
        clean_scores = score_data["clean"][-1]
        clean_mean = np.mean(clean_scores)
        clean_std = np.std(clean_scores)
        ax1.axhspan(clean_mean - clean_std, clean_mean + clean_std, 
                     alpha=0.15, color='green', label=f"clean (mean={clean_mean:.3f})")
        ax1.axhline(clean_mean, color='green', linestyle='--', alpha=0.5)
    
    correlations = {}
    
    for i, deg_type in enumerate(deg_types):
        levels = sorted([l for l in score_data[deg_type].keys() if l >= 0])
        means = []
        stds = []
        all_levels_flat = []
        all_scores_flat = []
        
        for level in levels:
            scores = score_data[deg_type][level]
            means.append(np.mean(scores))
            stds.append(np.std(scores))
            all_levels_flat.extend([level] * len(scores))
            all_scores_flat.extend(scores)
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot mean line with error bars
        ax1.errorbar(levels, means, yerr=stds, marker='o', color=colors[i],
                     label=f"{deg_type}", linewidth=2, capsize=4, markersize=6)
        
        # Scatter individual points (faded)
        for level in levels:
            scores = score_data[deg_type][level]
            ax1.scatter([level] * len(scores), scores, color=colors[i], 
                       alpha=0.15, s=15, zorder=1)
        
        # Compute Pearson and Spearman correlation
        if len(all_levels_flat) > 2:
            pearson_r, pearson_p = stats.pearsonr(all_levels_flat, all_scores_flat)
            spearman_r, spearman_p = stats.spearmanr(all_levels_flat, all_scores_flat)
            correlations[deg_type] = {
                'pearson_r': pearson_r, 'pearson_p': pearson_p,
                'spearman_r': spearman_r, 'spearman_p': spearman_p,
            }
    
    ax1.set_xlabel("Degradation Level (0=mild, 4=severe)")
    ax1.set_ylabel("ARNIQA Quality Score")
    ax1.set_xticks(range(5))
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Mean Score ± Std per Level")
    
    # --- Panel 2: Correlation summary ---
    ax2 = axes[1]
    ax2.axis('off')
    
    summary = "SCORE vs LEVEL CORRELATION\n"
    summary += "=" * 45 + "\n\n"
    
    for deg_type, corr in correlations.items():
        summary += f"{deg_type}:\n"
        summary += f"  Pearson r  = {corr['pearson_r']:+.4f}  (p={corr['pearson_p']:.2e})\n"
        summary += f"  Spearman ρ = {corr['spearman_r']:+.4f}  (p={corr['spearman_p']:.2e})\n\n"
    
    summary += "=" * 45 + "\n"
    summary += "INTERPRETATION:\n"
    
    # Check if all correlations are negative and significant
    all_negative = all(c['pearson_r'] < -0.3 for c in correlations.values())
    all_significant = all(c['pearson_p'] < 0.01 for c in correlations.values())
    
    if all_negative and all_significant:
        summary += "STRONG negative correlation for all types.\n"
        summary += "Higher degradation → lower ARNIQA score.\n"
        summary += "→ ARNIQA encodes BOTH type AND severity.\n"
        summary += "→ Conditioning carries intensity information."
    elif any(c['pearson_r'] < -0.3 and c['pearson_p'] < 0.01 for c in correlations.values()):
        summary += "MIXED results: some types show correlation,\n"
        summary += "others don't.\n"
        summary += "→ ARNIQA is sensitive to some degradations\n"
        summary += "  but blind to others."
    else:
        summary += "WEAK/NO correlation.\n"
        summary += "ARNIQA score does NOT track severity.\n"
        summary += "→ Conditioning carries TYPE info only,\n"
        summary += "  NOT intensity."
    
    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Score vs level figure saved to {output_path}")
    
    return correlations


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ARNIQA feature clustering by degradation type"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing clean images (PNG/JPG)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/arniqa_cluster_analysis",
        help="Output directory for figures and data"
    )
    parser.add_argument(
        "--num_images", type=int, default=50,
        help="Maximum number of clean images to use (default: 50)"
    )
    parser.add_argument(
        "--degradation_types", type=str, default="whitenoise,jpeg,gaublur",
        help="Comma-separated degradation types to test (default: whitenoise,jpeg,gaublur)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    args = parser.parse_args()
    
    device = torch.device(args.device)
    degradation_types = [d.strip() for d in args.degradation_types.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Device: {device}")
    logger.info(f"Degradation types to analyze: {degradation_types}")
    
    # --- Step 1: Load ARNIQA encoder ---
    encoder_model, global_pool = load_arniqa_encoder(device)
    
    # --- Step 1b: Load ARNIQA scorer (for scalar quality scores) ---
    arniqa_scorer = load_arniqa_scorer(device)
    
    # --- Step 2: Load clean images ---
    clean_images = load_clean_images(args.input_dir, args.num_images)
    
    # --- Step 3: Initialize degradation system ---
    # Using ImageDistorter from src/ARNIQA/degradation.py (verified at line 127)
    distorter = ImageDistorter()
    
    # Validate requested degradation types
    available_degradations = list(distorter.distortion_functions.keys())
    for deg_type in degradation_types:
        if deg_type not in available_degradations:
            raise ValueError(
                f"Degradation '{deg_type}' not available. "
                f"Available: {available_degradations}"
            )
    
    # --- Step 4: Extract features and scores ---
    all_embeddings = []
    all_labels_type = []   # degradation type (e.g., "whitenoise", "jpeg", "clean")
    all_labels_level = []  # severity level (0-4, or -1 for clean)
    all_filenames = []
    # Score data: {deg_type: {level: [scores]}}
    score_data = {}
    
    total_samples = len(clean_images) * (1 + len(degradation_types) * 5)
    logger.info(f"Extracting features and scores for {total_samples} samples "
                f"({len(clean_images)} clean + "
                f"{len(clean_images) * len(degradation_types) * 5} degraded)...")
    
    processed = 0
    
    for img_name, img_tensor in clean_images:
        # Extract features and score for clean image
        feat = extract_global_features(encoder_model, global_pool, img_tensor, device)
        score = compute_arniqa_score(arniqa_scorer, img_tensor, device)
        all_embeddings.append(feat)
        all_labels_type.append("clean")
        all_labels_level.append(-1)
        all_filenames.append(img_name)
        score_data.setdefault("clean", {}).setdefault(-1, []).append(score)
        processed += 1
        
        # Extract features and scores for each degradation type and level
        for deg_type in degradation_types:
            for level in range(5):  # Levels 0-4
                try:
                    # apply_distortion_to_tensor verified at degradation.py line 254
                    degraded = distorter.apply_distortion_to_tensor(
                        img_tensor, deg_type, level
                    )
                    feat = extract_global_features(
                        encoder_model, global_pool, degraded, device
                    )
                    score = compute_arniqa_score(arniqa_scorer, degraded, device)
                    all_embeddings.append(feat)
                    all_labels_type.append(deg_type)
                    all_labels_level.append(level)
                    all_filenames.append(f"{img_name}.{deg_type}.{level}")
                    score_data.setdefault(deg_type, {}).setdefault(level, []).append(score)
                    processed += 1
                except Exception as e:
                    logger.warning(f"Failed {img_name} {deg_type} L{level}: {e}")
                    continue
        
        if processed % 50 == 0:
            logger.info(f"Progress: {processed}/{total_samples} samples processed")
    
    embeddings = np.array(all_embeddings)
    logger.info(f"Feature extraction complete: {embeddings.shape}")
    
    # --- Step 5: Compute metrics ---
    
    # Silhouette score by degradation type
    silhouette_type = silhouette_score(embeddings, all_labels_type, metric='cosine')
    logger.info(f"Silhouette score (by type): {silhouette_type:.4f}")
    
    # Silhouette score by severity level (excluding clean)
    degraded_mask = np.array([l != "clean" for l in all_labels_type])
    if degraded_mask.sum() > 0:
        silhouette_level = silhouette_score(
            embeddings[degraded_mask],
            np.array(all_labels_level)[degraded_mask],
            metric='cosine'
        )
        logger.info(f"Silhouette score (by level): {silhouette_level:.4f}")
    else:
        silhouette_level = 0.0
    
    # Cosine similarity matrix
    sim_matrix, class_names = compute_cosine_similarity_matrix(
        embeddings, all_labels_type
    )
    
    # --- Step 6: Create visualization ---
    figure_path = str(output_dir / "cluster_analysis.png")
    create_analysis_figure(
        embeddings=embeddings,
        labels_type=all_labels_type,
        labels_level=all_labels_level,
        sim_matrix=sim_matrix,
        class_names=class_names,
        silhouette_type=silhouette_type,
        silhouette_level=silhouette_level,
        output_path=figure_path,
    )
    
    # --- Step 7: Score vs Level analysis ---
    score_figure_path = str(output_dir / "score_vs_level.png")
    correlations = create_score_vs_level_figure(
        score_data=score_data,
        output_path=score_figure_path,
    )
    
    # --- Step 8: Save embeddings CSV ---
    csv_path = output_dir / "embeddings.csv"
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["filename", "degradation_type", "level"] + [f"feat_{i}" for i in range(2048)]
        writer.writerow(header)
        for i in range(len(all_embeddings)):
            row = [all_filenames[i], all_labels_type[i], all_labels_level[i]] + \
                  all_embeddings[i].tolist()
            writer.writerow(row)
    logger.info(f"Embeddings saved to {csv_path}")
    
    # --- Step 9: Save text report ---
    report_path = output_dir / "analysis_report.txt"
    
    # Compute intra/inter class stats
    intra_sims = [sim_matrix[i, i] for i in range(len(class_names))]
    inter_sims = [sim_matrix[i, j] for i in range(len(class_names))
                  for j in range(len(class_names)) if i != j]
    mean_intra = np.mean(intra_sims)
    mean_inter = np.mean(inter_sims)
    separation = mean_intra - mean_inter
    
    with open(report_path, 'w') as f:
        f.write("ARNIQA Degradation Type Clustering Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Number of clean images: {len(clean_images)}\n")
        f.write(f"Degradation types: {degradation_types}\n")
        f.write(f"Total samples: {len(embeddings)}\n")
        f.write(f"Feature dimension: {embeddings.shape[1]}\n\n")
        
        f.write("SILHOUETTE SCORES\n")
        f.write(f"  By degradation type: {silhouette_type:.4f}\n")
        f.write(f"  By severity level:   {silhouette_level:.4f}\n\n")
        
        f.write("COSINE SIMILARITY MATRIX\n")
        f.write(f"  Classes: {class_names}\n")
        for i, name_i in enumerate(class_names):
            for j, name_j in enumerate(class_names):
                f.write(f"  {name_i} vs {name_j}: {sim_matrix[i, j]:.4f}\n")
        f.write(f"\n  Mean intra-class: {mean_intra:.4f}\n")
        f.write(f"  Mean inter-class: {mean_inter:.4f}\n")
        f.write(f"  Separation: {separation:.4f}\n\n")
        
        f.write("SCORE vs LEVEL CORRELATION\n")
        for deg_type, corr in correlations.items():
            f.write(f"  {deg_type}:\n")
            f.write(f"    Pearson r  = {corr['pearson_r']:+.4f}  (p={corr['pearson_p']:.2e})\n")
            f.write(f"    Spearman ρ = {corr['spearman_r']:+.4f}  (p={corr['spearman_p']:.2e})\n")
        f.write("\n")
        
        f.write("VERDICT\n")
        if silhouette_type > 0.3 and separation > 0.05:
            f.write("STRONG type separation. Conditioning approach is VIABLE.\n")
        elif silhouette_type > 0.1 and separation > 0.02:
            f.write("WEAK type separation. Conditioning may have LIMITED benefit.\n")
        else:
            f.write("NO meaningful type separation. Conditioning will likely be IGNORED.\n")
    
    logger.info(f"Report saved to {report_path}")
    
    # --- Print summary to console ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Silhouette (by type):  {silhouette_type:.4f}")
    print(f"Silhouette (by level): {silhouette_level:.4f}")
    print(f"Cosine sim separation: {separation:.4f}")
    print(f"Mean intra-class sim:  {mean_intra:.4f}")
    print(f"Mean inter-class sim:  {mean_inter:.4f}")
    print("=" * 60)
    
    print("\nSCORE vs LEVEL CORRELATION:")
    for deg_type, corr in correlations.items():
        print(f"  {deg_type}: Pearson r={corr['pearson_r']:+.4f} (p={corr['pearson_p']:.2e}), "
              f"Spearman ρ={corr['spearman_r']:+.4f} (p={corr['spearman_p']:.2e})")
    print("=" * 60)
    
    if silhouette_type > 0.3 and separation > 0.05:
        print("VERDICT: STRONG type separation -> conditioning VIABLE")
    elif silhouette_type > 0.1 and separation > 0.02:
        print("VERDICT: WEAK type separation -> LIMITED benefit expected")
    else:
        print("VERDICT: NO type separation -> conditioning likely USELESS")
    
    print(f"\nOutputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
