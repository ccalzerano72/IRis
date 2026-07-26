#!/usr/bin/env python3
"""
Comprehensive Metrics Calculation for Restoration Model Evaluation
Calculates PSNR, SSIM, LPIPS, Delta-E, ARNIQA, BRISQUE, NIQE, MANIQA, MUSIQ

Based on webserver_restoration/utils/metrics.py pattern

Usage:
    python script/restoration/eval/04_calculate_metrics.py \
        --clean_dir path/to/clean \
        --restored_dir path/to/restored \
        --output_dir output/eval/metrics \
        --model_name marigold
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Metrics imports
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab, deltaE_ciede2000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


class MetricsCalculator:
    """
    Comprehensive metrics calculator for image quality assessment.
    
    Pattern based on webserver_restoration/utils/metrics.py
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize all metric models"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.lpips_model = None
        self.arniqa_model = None
        self.brisque_model = None
        self.niqe_model = None
        self.maniqa_model = None
        self.musiq_model = None
        
        self._initialize_lpips()
        self._initialize_arniqa()
        self._initialize_pyiqa_models()
    
    def _initialize_lpips(self):
        """Initialize LPIPS model - pattern from webserver_restoration/utils/metrics.py lines 35-41"""
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex')
            self.lpips_model = self.lpips_model.to(self.device)
            logger.info("✓ LPIPS model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LPIPS: {str(e)}")
            self.lpips_model = None
    
    def _initialize_arniqa(self):
        """Initialize ARNIQA model - pattern from webserver_restoration/utils/metrics.py lines 43-53"""
        try:
            from torchmetrics.image.arniqa import ARNIQA
            self.arniqa_model = ARNIQA(
                regressor_dataset='koniq10k',
                normalize=True,
                reduction='none'
            )
            self.arniqa_model = self.arniqa_model.to(self.device)
            logger.info("✓ ARNIQA model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ARNIQA: {str(e)}")
            self.arniqa_model = None
    
    def _initialize_pyiqa_models(self):
        """Initialize pyiqa models (BRISQUE, NIQE, MANIQA, MUSIQ)"""
        try:
            import pyiqa
            
            # BRISQUE - pattern from webserver_restoration/utils/metrics.py lines 55-61
            try:
                self.brisque_model = pyiqa.create_metric('brisque', device=self.device)
                logger.info("✓ BRISQUE model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize BRISQUE: {str(e)}")
                self.brisque_model = None
            
            # NIQE - pattern from webserver_restoration/utils/metrics.py lines 63-69
            try:
                self.niqe_model = pyiqa.create_metric('niqe', device=self.device)
                logger.info("✓ NIQE model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NIQE: {str(e)}")
                self.niqe_model = None
            
            # MANIQA
            try:
                self.maniqa_model = pyiqa.create_metric('maniqa', device=self.device)
                logger.info("✓ MANIQA model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MANIQA: {str(e)}")
                self.maniqa_model = None
            
            # MUSIQ
            try:
                self.musiq_model = pyiqa.create_metric('musiq', device=self.device)
                logger.info("✓ MUSIQ model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MUSIQ: {str(e)}")
                self.musiq_model = None
                
        except ImportError as e:
            logger.warning(f"pyiqa not available: {str(e)}")

    
    # -------------------- Full-Reference Metrics --------------------
    
    def calculate_psnr(self, clean_np: np.ndarray, restored_np: np.ndarray) -> Optional[float]:
        """
        Calculate PSNR - pattern from webserver_restoration/utils/metrics.py lines 71-80
        
        Args:
            clean_np, restored_np: numpy arrays in [0, 255] range, shape (H, W, C)
        Returns:
            PSNR value in dB (higher is better)
        """
        try:
            psnr = peak_signal_noise_ratio(clean_np, restored_np, data_range=255)
            return round(float(psnr), 4)
        except Exception as e:
            logger.error(f"Error calculating PSNR: {str(e)}")
            return None
    
    def calculate_ssim(self, clean_np: np.ndarray, restored_np: np.ndarray) -> Optional[float]:
        """
        Calculate SSIM (Structural Similarity Index)
        
        Args:
            clean_np, restored_np: numpy arrays in [0, 255] range, shape (H, W, C)
        Returns:
            SSIM value in [0, 1] (higher is better)
        """
        try:
            # SSIM with Wang et al. 2004 standard parameters
            ssim = structural_similarity(
                clean_np, restored_np, 
                data_range=255, 
                channel_axis=2,  # RGB channel is last axis
                gaussian_weights=True,
                sigma=1.5,
                # win_size auto-computed to 11 with gaussian_weights=True, sigma=1.5
            )
            return round(float(ssim), 4)
        except Exception as e:
            logger.error(f"Error calculating SSIM: {str(e)}")
            return None
    
    def calculate_lpips(self, clean_pil: Image.Image, restored_pil: Image.Image) -> Optional[float]:
        """
        Calculate LPIPS - pattern from webserver_restoration/utils/metrics.py lines 82-107
        
        Args:
            clean_pil, restored_pil: PIL Images
        Returns:
            LPIPS value (lower is better, 0 = identical)
        """
        if self.lpips_model is None:
            return None
        
        try:
            # Ensure same size
            if clean_pil.size != restored_pil.size:
                restored_pil = restored_pil.resize(clean_pil.size, Image.Resampling.LANCZOS)
            
            # Convert to tensor [-1, 1] range for LPIPS
            clean_tensor = self._pil_to_lpips_tensor(clean_pil)
            restored_tensor = self._pil_to_lpips_tensor(restored_pil)
            
            clean_tensor = clean_tensor.to(self.device)
            restored_tensor = restored_tensor.to(self.device)
            
            with torch.no_grad():
                lpips_value = self.lpips_model(clean_tensor, restored_tensor)
            
            return round(float(lpips_value.item()), 4)
            
        except Exception as e:
            logger.error(f"Error calculating LPIPS: {str(e)}")
            return None
    
    def calculate_delta_e(self, clean_np: np.ndarray, restored_np: np.ndarray) -> Optional[float]:
        """
        Calculate Delta-E (CIEDE2000) color difference.
        
        Args:
            clean_np, restored_np: numpy arrays in [0, 255] range, shape (H, W, C), RGB
        Returns:
            Mean Delta-E value (lower is better, 0 = identical)
        """
        try:
            # Convert RGB [0, 255] to Lab color space
            clean_lab = rgb2lab(clean_np.astype(np.float64) / 255.0)
            restored_lab = rgb2lab(restored_np.astype(np.float64) / 255.0)
            
            # Compute per-pixel CIEDE2000 and return mean
            delta_e = deltaE_ciede2000(clean_lab, restored_lab)
            return round(float(delta_e.mean()), 4)
        except Exception as e:
            logger.error(f"Error calculating Delta-E: {str(e)}")
            return None
    
    # -------------------- No-Reference Metrics --------------------
    
    def calculate_arniqa(self, img_pil: Image.Image) -> Optional[float]:
        """
        Calculate ARNIQA - pattern from webserver_restoration/utils/metrics.py lines 109-128
        
        Args:
            img_pil: PIL Image
        Returns:
            ARNIQA score in [0, 1] range (higher is better)
        """
        if self.arniqa_model is None:
            return None
        
        try:
            img_tensor = self._pil_to_tensor_01(img_pil)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                score = self.arniqa_model(img_tensor)
            
            return round(float(score.item()), 4)
            
        except Exception as e:
            logger.error(f"Error calculating ARNIQA: {str(e)}")
            return None
    
    def calculate_brisque(self, img_pil: Image.Image) -> Optional[float]:
        """
        Calculate BRISQUE - pattern from webserver_restoration/utils/metrics.py lines 130-149
        
        Args:
            img_pil: PIL Image
        Returns:
            BRISQUE score (lower is better, typically 0-100)
        """
        if self.brisque_model is None:
            return None
        
        try:
            img_tensor = self._pil_to_tensor_01(img_pil)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                score = self.brisque_model(img_tensor)
            
            return round(float(score.item()), 4)
            
        except Exception as e:
            logger.error(f"Error calculating BRISQUE: {str(e)}")
            return None
    
    def calculate_niqe(self, img_pil: Image.Image) -> Optional[float]:
        """
        Calculate NIQE - pattern from webserver_restoration/utils/metrics.py lines 151-170
        
        Args:
            img_pil: PIL Image
        Returns:
            NIQE score (lower is better, typically 0-10)
        """
        if self.niqe_model is None:
            return None
        
        try:
            img_tensor = self._pil_to_tensor_01(img_pil)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                score = self.niqe_model(img_tensor)
            
            return round(float(score.item()), 4)
            
        except Exception as e:
            logger.error(f"Error calculating NIQE: {str(e)}")
            return None
    
    def calculate_maniqa(self, img_pil: Image.Image) -> Optional[float]:
        """
        Calculate MANIQA (Multi-dimension Attention Network for IQA)
        
        Args:
            img_pil: PIL Image
        Returns:
            MANIQA score (higher is better)
        """
        if self.maniqa_model is None:
            return None
        
        try:
            img_tensor = self._pil_to_tensor_01(img_pil)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                score = self.maniqa_model(img_tensor)
            
            return round(float(score.item()), 4)
            
        except Exception as e:
            logger.error(f"Error calculating MANIQA: {str(e)}")
            return None
    
    def calculate_musiq(self, img_pil: Image.Image) -> Optional[float]:
        """
        Calculate MUSIQ (Multi-scale Image Quality Transformer)
        
        Args:
            img_pil: PIL Image
        Returns:
            MUSIQ score (higher is better)
        """
        if self.musiq_model is None:
            return None
        
        try:
            img_tensor = self._pil_to_tensor_01(img_pil)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                score = self.musiq_model(img_tensor)
            
            return round(float(score.item()), 4)
            
        except Exception as e:
            logger.error(f"Error calculating MUSIQ: {str(e)}")
            return None

    
    # -------------------- Helper Methods --------------------
    
    def _pil_to_lpips_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to tensor for LPIPS: [-1, 1] range, (1, 3, H, W)
        Pattern from webserver_restoration/utils/metrics.py lines 280-284
        """
        np_img = np.array(pil_img).astype(np.float32)
        np_img = (np_img / 255.0) * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def _pil_to_tensor_01(self, pil_img: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to tensor: [0, 1] range, (1, 3, H, W)
        Pattern from webserver_restoration/utils/metrics.py lines 286-290
        """
        np_img = np.array(pil_img).astype(np.float32)
        np_img = np_img / 255.0  # [0, 255] -> [0, 1]
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
        return tensor
    
    def calculate_all_metrics(
        self, 
        clean_pil: Image.Image, 
        restored_pil: Image.Image
    ) -> Dict[str, Optional[float]]:
        """
        Calculate all metrics for a single image pair.
        
        Args:
            clean_pil: Clean/ground truth PIL Image
            restored_pil: Restored PIL Image
            
        Returns:
            Dictionary with all metric values
        """
        # Ensure same size
        if clean_pil.size != restored_pil.size:
            restored_pil = restored_pil.resize(clean_pil.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy for PSNR/SSIM
        clean_np = np.array(clean_pil)
        restored_np = np.array(restored_pil)
        
        metrics = {
            # Full-reference metrics
            'psnr': self.calculate_psnr(clean_np, restored_np),
            'ssim': self.calculate_ssim(clean_np, restored_np),
            'lpips': self.calculate_lpips(clean_pil, restored_pil),
            'delta_e': self.calculate_delta_e(clean_np, restored_np),
            # No-reference metrics (on restored image)
            'arniqa': self.calculate_arniqa(restored_pil),
            'brisque': self.calculate_brisque(restored_pil),
            'niqe': self.calculate_niqe(restored_pil),
            'maniqa': self.calculate_maniqa(restored_pil),
            'musiq': self.calculate_musiq(restored_pil),
        }
        
        return metrics


# -------------------- File Matching --------------------

def find_matching_files(clean_dir: str, restored_dir: str) -> List[Tuple[Path, Path, str]]:
    """
    Find matching clean and restored image files.
    
    Returns:
        List of (clean_path, restored_path, filename) tuples
    """
    clean_path = Path(clean_dir)
    restored_path = Path(restored_dir)
    
    # Get all clean files
    clean_files = {}
    for f in clean_path.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Store with various possible stems
            stem = f.stem
            clean_stem = stem.replace('_gt', '').replace('_clean', '').replace('_original', '')
            clean_files[clean_stem] = f
            clean_files[stem] = f  # Also store original stem
    
    # Find matching restored files
    matches = []
    for f in restored_path.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            stem = f.stem
            # Try various naming patterns
            clean_stem = stem.replace('_restored', '').replace('_pred', '').replace('_output', '')
            clean_stem = clean_stem.replace('_diffbir', '').replace('_marigold', '')
            
            # Try to find match
            if clean_stem in clean_files:
                matches.append((clean_files[clean_stem], f, clean_stem))
            elif stem in clean_files:
                matches.append((clean_files[stem], f, stem))
    
    return sorted(matches, key=lambda x: x[2])



# -------------------- Statistics Calculation --------------------

def calculate_statistics(df: pd.DataFrame, metric_columns: List[str]) -> Dict:
    """
    Calculate comprehensive statistics for all metrics.
    
    Args:
        df: DataFrame with per-image metrics
        metric_columns: List of metric column names
        
    Returns:
        Dictionary with statistics for each metric
    """
    stats = {}
    
    for metric in metric_columns:
        if metric not in df.columns:
            continue
        
        values = df[metric].dropna()
        if len(values) == 0:
            continue
        
        # Determine if higher or lower is better
        higher_is_better = metric in ['psnr', 'ssim', 'arniqa', 'maniqa', 'musiq']
        
        # Find best/worst
        if higher_is_better:
            best_idx = values.idxmax()
            worst_idx = values.idxmin()
        else:
            best_idx = values.idxmin()
            worst_idx = values.idxmax()
        
        stats[metric] = {
            'mean': round(float(values.mean()), 4),
            'std': round(float(values.std()), 4),
            'median': round(float(values.median()), 4),
            'min': round(float(values.min()), 4),
            'max': round(float(values.max()), 4),
            'count': int(len(values)),
            'higher_is_better': higher_is_better,
            'best_value': round(float(values.loc[best_idx]), 4),
            'best_image': df.loc[best_idx, 'filename'],
            'worst_value': round(float(values.loc[worst_idx]), 4),
            'worst_image': df.loc[worst_idx, 'filename'],
        }
    
    return stats


def write_summary_file(stats: Dict, model_name: str, output_path: Path, num_images: int):
    """
    Write human-readable summary file with statistics.
    """
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"METRICS SUMMARY - {model_name.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of images: {num_images}\n")
        f.write("=" * 70 + "\n\n")
        
        # Group metrics by type
        fr_metrics = ['psnr', 'ssim', 'lpips', 'delta_e']
        nr_metrics = ['arniqa', 'brisque', 'niqe', 'maniqa', 'musiq']
        
        f.write("FULL-REFERENCE METRICS (require clean image)\n")
        f.write("-" * 50 + "\n")
        for metric in fr_metrics:
            if metric in stats:
                _write_metric_stats(f, metric, stats[metric])
        
        f.write("\nNO-REFERENCE METRICS (quality of restored image)\n")
        f.write("-" * 50 + "\n")
        for metric in nr_metrics:
            if metric in stats:
                _write_metric_stats(f, metric, stats[metric])
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("METRIC INTERPRETATION\n")
        f.write("-" * 50 + "\n")
        f.write("PSNR:    Higher is better (dB), typical range: 20-40\n")
        f.write("SSIM:    Higher is better [0-1], 1 = identical\n")
        f.write("LPIPS:   Lower is better [0-1], 0 = identical\n")
        f.write("Delta-E: Lower is better (CIEDE2000), 0 = identical\n")
        f.write("ARNIQA:  Higher is better [0-1]\n")
        f.write("BRISQUE: Lower is better, typical range: 0-100\n")
        f.write("NIQE:    Lower is better, typical range: 0-10\n")
        f.write("MANIQA:  Higher is better\n")
        f.write("MUSIQ:   Higher is better\n")
        f.write("=" * 70 + "\n")


def _write_metric_stats(f, metric: str, s: Dict):
    """Write statistics for a single metric."""
    direction = "↑" if s['higher_is_better'] else "↓"
    f.write(f"\n{metric.upper()} {direction}\n")
    f.write(f"  Mean:   {s['mean']:.4f} ± {s['std']:.4f}\n")
    f.write(f"  Median: {s['median']:.4f}\n")
    f.write(f"  Range:  [{s['min']:.4f}, {s['max']:.4f}]\n")
    f.write(f"  Best:   {s['best_value']:.4f} ({s['best_image']})\n")
    f.write(f"  Worst:  {s['worst_value']:.4f} ({s['worst_image']})\n")



# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate comprehensive metrics for restoration evaluation"
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        required=True,
        help="Directory containing clean/ground truth images"
    )
    parser.add_argument(
        "--restored_dir",
        type=str,
        required=True,
        help="Directory containing restored images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for metrics results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="Name to identify the model (used in output filenames)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    logger.info(f"Searching for matching files...")
    logger.info(f"  Clean dir: {args.clean_dir}")
    logger.info(f"  Restored dir: {args.restored_dir}")
    
    matches = find_matching_files(args.clean_dir, args.restored_dir)
    
    if not matches:
        logger.error("No matching image pairs found!")
        logger.error("Make sure clean and restored images have matching filenames")
        return
    
    logger.info(f"Found {len(matches)} matching image pairs")
    
    # Initialize metrics calculator
    logger.info("Initializing metrics calculator...")
    calculator = MetricsCalculator(device=args.device)
    
    # Calculate metrics for each image pair
    results = []
    metric_columns = ['psnr', 'ssim', 'lpips', 'delta_e', 'arniqa', 'brisque', 'niqe', 'maniqa', 'musiq']
    
    logger.info("Calculating metrics...")
    for clean_path, restored_path, filename in tqdm(matches, desc="Processing images"):
        try:
            # Load images
            clean_pil = Image.open(clean_path).convert('RGB')
            restored_pil = Image.open(restored_path).convert('RGB')
            
            # Calculate all metrics
            metrics = calculator.calculate_all_metrics(clean_pil, restored_pil)
            metrics['filename'] = filename
            metrics['clean_path'] = str(clean_path)
            metrics['restored_path'] = str(restored_path)
            
            results.append(metrics)
            
        except Exception as e:
            logger.warning(f"Error processing {filename}: {str(e)}")
            continue
    
    if not results:
        logger.error("No results generated!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['filename'] + metric_columns + ['clean_path', 'restored_path']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Save per-image CSV
    csv_path = output_dir / f"metrics_{args.model_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Per-image metrics saved to: {csv_path}")
    
    # Calculate statistics
    stats = calculate_statistics(df, metric_columns)
    
    # Save summary text file
    summary_path = output_dir / f"summary_{args.model_name}.txt"
    write_summary_file(stats, args.model_name, summary_path, len(results))
    logger.info(f"Summary saved to: {summary_path}")
    
    # Print summary to console
    print("\n" + "=" * 50)
    print(f"RESULTS SUMMARY - {args.model_name.upper()}")
    print("=" * 50)
    for metric in metric_columns:
        if metric in stats:
            s = stats[metric]
            direction = "↑" if s['higher_is_better'] else "↓"
            print(f"{metric.upper():8} {direction}: {s['mean']:.4f} ± {s['std']:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
