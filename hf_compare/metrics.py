# Lightweight IQA module for the IRis-compare Space.
# Adapted from script/restoration/eval/04_calculate_metrics.py (thesis stack).
#
# FR (needs clean reference): PSNR, SSIM (scikit-image), LPIPS-alex (torch).
# NR (reference-free): NIQE, MUSIQ (pyiqa).
# Every metric initializes lazily and degrades to None on failure.

import logging

import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger(__name__)

HIGHER_BETTER = {"PSNR ↑", "SSIM ↑", "MUSIQ ↑"}
LOWER_BETTER = {"LPIPS ↓", "NIQE ↓"}
BEST_BG = "#cfe8cf"  # light green


def style_best(df: pd.DataFrame):
    """Highlight the best value of each metric column (green, bold).

    Respects metric direction (↑ higher-better, ↓ lower-better) and
    ignores missing (None/NaN) values.
    """

    def _highlight(col):
        vals = pd.to_numeric(col, errors="coerce")
        if vals.isna().all():
            return [""] * len(col)
        best_idx = vals.idxmax() if col.name in HIGHER_BETTER else vals.idxmin()
        return [
            f"background-color: {BEST_BG}; font-weight: bold" if i == best_idx else ""
            for i in col.index
        ]

    return df.style.apply(_highlight, axis=0)


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


class LightMetrics:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lpips_model = None
        self.niqe_model = None
        self.musiq_model = None
        self._init_lpips()
        self._init_pyiqa()

    def _init_lpips(self):
        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
            logger.info("LPIPS initialized")
        except Exception as e:
            logger.warning(f"LPIPS unavailable: {e}")
            self.lpips_model = None

    def _init_pyiqa(self):
        try:
            import pyiqa

            try:
                self.niqe_model = pyiqa.create_metric("niqe", device=self.device)
                logger.info("NIQE initialized")
            except Exception as e:
                logger.warning(f"NIQE unavailable: {e}")
            try:
                self.musiq_model = pyiqa.create_metric("musiq", device=self.device)
                logger.info("MUSIQ initialized")
            except Exception as e:
                logger.warning(f"MUSIQ unavailable: {e}")
        except Exception as e:
            logger.warning(f"pyiqa unavailable: {e}")

    @staticmethod
    def psnr(clean: Image.Image, restored: Image.Image):
        try:
            from skimage.metrics import peak_signal_noise_ratio

            return round(float(peak_signal_noise_ratio(_to_array(clean), _to_array(restored), data_range=1.0)), 2)
        except Exception as e:
            logger.warning(f"PSNR failed: {e}")
            return None

    @staticmethod
    def ssim(clean: Image.Image, restored: Image.Image):
        try:
            from skimage.metrics import structural_similarity

            return round(
                float(structural_similarity(_to_array(clean), _to_array(restored), channel_axis=2, data_range=1.0)),
                4,
            )
        except Exception as e:
            logger.warning(f"SSIM failed: {e}")
            return None

    def lpips(self, clean: Image.Image, restored: Image.Image):
        if self.lpips_model is None:
            return None
        try:
            c = torch.from_numpy(_to_array(clean)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            r = torch.from_numpy(_to_array(restored)).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            with torch.no_grad():
                v = self.lpips_model(c.to(self.device), r.to(self.device))
            return round(float(v.item()), 4)
        except Exception as e:
            logger.warning(f"LPIPS failed: {e}")
            return None

    def _pyiqa_score(self, model, img: Image.Image):
        if model is None:
            return None
        try:
            arr = _to_array(img)
            ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                v = model(ten)
            return round(float(v.item()), 4)
        except Exception as e:
            logger.warning(f"pyiqa metric failed: {e}")
            return None

    def niqe(self, img: Image.Image):
        return self._pyiqa_score(self.niqe_model, img)

    def musiq(self, img: Image.Image):
        return self._pyiqa_score(self.musiq_model, img)

    def full_reference(self, clean: Image.Image, restored: Image.Image) -> dict:
        if restored.size != clean.size:
            restored = restored.resize(clean.size, Image.LANCZOS)
        return {
            "PSNR ↑": self.psnr(clean, restored),
            "SSIM ↑": self.ssim(clean, restored),
            "LPIPS ↓": self.lpips(clean, restored),
        }

    def no_reference(self, img: Image.Image) -> dict:
        return {
            "NIQE ↓": self.niqe(img),
            "MUSIQ ↑": self.musiq(img),
        }
