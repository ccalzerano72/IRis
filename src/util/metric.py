# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from skimage.color import rgb2lab, deltaE_ciede2000
    SKIMAGE_COLOR_AVAILABLE = True
except ImportError:
    SKIMAGE_COLOR_AVAILABLE = False


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# -------------------- Depth Metrics --------------------


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# Adapted from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss


# -------------------- Normals Metrics --------------------


def compute_cosine_error(pred_norm, gt_norm, masked=False):
    if len(pred_norm.shape) == 4:
        pred_norm = pred_norm.squeeze(0)
    if len(gt_norm.shape) == 4:
        gt_norm = gt_norm.squeeze(0)

    # shape must be [3,H,W]
    assert (gt_norm.shape[0] == 3) and (
        pred_norm.shape[0] == 3
    ), "Channel dim should be the first dimension!"
    # mask out the zero vectors, otherwise torch.cosine_similarity computes 90° as error
    if masked:
        ch, h, w = gt_norm.shape

        mask = torch.norm(gt_norm, dim=0) > 0

        pred_norm = pred_norm[:, mask.view(h, w)]
        gt_norm = gt_norm[:, mask.view(h, w)]

    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=0)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi  # (H, W)

    return (
        pred_error.view(-1).detach().cpu().numpy()
    )  # flatten so can directly input to compute_normal_metrics()


def mean_angular_error(cosine_error):
    return round(np.average(cosine_error), 4)


def median_angular_error(cosine_error):
    return round(np.median(cosine_error), 4)


def rmse_angular_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(np.sqrt(np.sum(cosine_error * cosine_error) / num_pixels), 4)


def sub5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 5) / num_pixels), 4)


def sub7_5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 7.5) / num_pixels), 4)


def sub11_25_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 11.25) / num_pixels), 4)


def sub22_5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 22.5) / num_pixels), 4)


def sub30_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 30) / num_pixels), 4)


# -------------------- IID Metrics --------------------


def compute_iid_metric(pred, gt, target_name, metric_name, metric, valid_mask=None):
    # Shading and residual are up-to-scale. We first scale-align them to the gt
    # and map them to the range [0,1] for metric computation
    if target_name == "shading" or target_name == "residual":
        alignment_scale = compute_alignment_scale(pred, gt, valid_mask)
        pred = alignment_scale * pred
        # map to [0,1]
        pred, gt = quantile_map(pred, gt, valid_mask)

    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    if valid_mask is not None:
        if len(valid_mask.shape) == 3:
            valid_mask = valid_mask.unsqueeze(0)
        if metric_name == "psnr":
            return metric(pred[valid_mask], gt[valid_mask]).item()
        # for SSIM and LPIPs set the invalid pixels to zero
        else:
            invalid_mask = ~valid_mask
            pred[invalid_mask] = 0
            gt[invalid_mask] = 0

    return metric(pred, gt).item()


# compute least-squares alignment scale to align shading/residual prediction to gt
def compute_alignment_scale(pred, gt, valid_mask=None):
    pred = pred.squeeze()
    gt = gt.squeeze()
    assert pred.shape[0] == 3 and gt.shape[0] == 3, "First dim should be channel dim"

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        pred = pred[valid_mask]
        gt = gt[valid_mask]

    A_flattened = pred.view(-1, 1)
    b_flattened = gt.view(-1, 1)
    # Solve the least squares problem
    x, residuals, rank, s = torch.linalg.lstsq(A_flattened.float(), b_flattened.float())
    return x


def quantile_map(pred, gt, valid_mask=None):
    pred = pred.squeeze()
    gt = gt.squeeze()
    assert gt.shape[0] == 3, "channel dim must be first dim"

    percentile = 90
    brightness_nth_percentile_desired = 0.8
    brightness = 0.3 * gt[0, :, :] + 0.59 * gt[1, :, :] + 0.11 * gt[2, :, :]

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        brightness = brightness[valid_mask[0]]
    else:
        brightness = brightness.flatten()

    eps = 0.0001

    brightness_nth_percentile_current = torch.quantile(brightness, percentile / 100.0)

    if brightness_nth_percentile_current < eps:
        scale = 0
    else:
        scale = float(
            brightness_nth_percentile_desired / brightness_nth_percentile_current
        )

    # Apply scaling to ground truth and prediction
    gt_mapped = torch.clamp(scale * gt, 0, 1).unsqueeze(0)  # [1,3,H,W]
    pred_mapped = torch.clamp(scale * pred, 0, 1).unsqueeze(0)  # [1,3,H,W]

    return pred_mapped, gt_mapped


# -------------------- Restoration Metrics --------------------


def psnr(pred, target, valid_mask=None):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.
    
    Args:
        pred (torch.Tensor): Predicted image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        target (torch.Tensor): Target image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        valid_mask (torch.Tensor, optional): Valid mask for pixels to consider
        
    Returns:
        float: PSNR value in dB
    """
    # Ensure tensors have batch dimension
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)
    
    # Apply mask if provided
    if valid_mask is not None:
        if len(valid_mask.shape) == 3:
            valid_mask = valid_mask.unsqueeze(0)
        # Set invalid pixels to same value to avoid affecting MSE
        pred = pred.clone()
        target = target.clone()
        pred[~valid_mask] = 0
        target[~valid_mask] = 0
    
    # Calculate MSE
    mse = F.mse_loss(pred, target, reduction='mean')
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR (assuming data range [0, 1])
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_value.item()


def ssim(pred, target, valid_mask=None):
    """
    Calculate Structural Similarity Index Measure (SSIM) between predicted and target images.
    
    Args:
        pred (torch.Tensor): Predicted image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        target (torch.Tensor): Target image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        valid_mask (torch.Tensor, optional): Valid mask for pixels to consider
        
    Returns:
        float: SSIM value in range [0, 1]
    """
    # Ensure tensors have batch dimension
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)
    
    # Apply mask if provided (set invalid pixels to 0)
    if valid_mask is not None:
        if len(valid_mask.shape) == 3:
            valid_mask = valid_mask.unsqueeze(0)
        pred = pred.clone()
        target = target.clone()
        pred[~valid_mask] = 0
        target[~valid_mask] = 0
    
    # Use torchmetrics SSIM - move to same device as input
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    ssim_value = ssim_metric(pred, target)
    return ssim_value.item()


def lpips_metric(pred, target, valid_mask=None, net='alex'):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS) between predicted and target images.
    
    Args:
        pred (torch.Tensor): Predicted image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        target (torch.Tensor): Target image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        valid_mask (torch.Tensor, optional): Valid mask for pixels to consider
        net (str): Network to use ('alex', 'vgg', 'squeeze')
        
    Returns:
        float: LPIPS value (lower is better)
    """
    if not LPIPS_AVAILABLE:
        raise ImportError("LPIPS not available. Install with: pip install lpips")
    
    # Ensure tensors have batch dimension
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)
    
    # Convert from [0, 1] to [-1, 1] for LPIPS
    pred_lpips = pred * 2.0 - 1.0
    target_lpips = target * 2.0 - 1.0
    
    # Apply mask if provided (set invalid pixels to 0)
    if valid_mask is not None:
        if len(valid_mask.shape) == 3:
            valid_mask = valid_mask.unsqueeze(0)
        pred_lpips = pred_lpips.clone()
        target_lpips = target_lpips.clone()
        pred_lpips[~valid_mask] = 0
        target_lpips[~valid_mask] = 0
    
    # Initialize LPIPS model (cached per device)
    device_key = f'_lpips_model_{pred.device}'
    if not hasattr(lpips_metric, device_key):
        model = lpips.LPIPS(net=net)
        model = model.to(pred.device)
        setattr(lpips_metric, device_key, model)
    
    lpips_model = getattr(lpips_metric, device_key)
    
    # Calculate LPIPS
    lpips_value = lpips_model(pred_lpips, target_lpips)
    return lpips_value.mean().item()


# Alias for consistency with other metrics
def lpips_alex(pred, target, valid_mask=None):
    """LPIPS with AlexNet backbone"""
    return lpips_metric(pred, target, valid_mask, net='alex')


def lpips_vgg(pred, target, valid_mask=None):
    """LPIPS with VGG backbone"""
    return lpips_metric(pred, target, valid_mask, net='vgg')


def l1_metric(pred, target, valid_mask=None):
    """
    Calculate L1 (Mean Absolute Error) between predicted and target images.
    
    Args:
        pred (torch.Tensor): Predicted image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        target (torch.Tensor): Target image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        valid_mask (torch.Tensor, optional): Valid mask for pixels to consider
        
    Returns:
        float: L1 (MAE) value (lower is better)
    """
    # Ensure tensors have batch dimension
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)
    
    # Apply mask if provided
    if valid_mask is not None:
        if len(valid_mask.shape) == 3:
            valid_mask = valid_mask.unsqueeze(0)
        # Calculate L1 only on valid pixels
        diff = torch.abs(pred - target)
        diff[~valid_mask] = 0
        n = valid_mask.sum()
        l1_value = diff.sum() / n if n > 0 else torch.tensor(0.0)
    else:
        # Calculate L1 on all pixels
        l1_value = F.l1_loss(pred, target, reduction='mean')
    
    return l1_value.item()


# -------------------- No-Reference Quality Metrics --------------------

# Global ARNIQA model cache (using torchmetrics implementation)
_ARNIQA_METRIC_CACHE = {}

def arniqa_score(pred, target=None, valid_mask=None):
    """
    Calculate ARNIQA quality score (no-reference metric).
    Higher score = better perceived quality.
    
    ARNIQA (Agnolucci et al., WACV 2024) learns a distortion manifold
    for image quality assessment without requiring a reference image.
    
    Uses torchmetrics.image.arniqa.ARNIQA which handles multi-scale
    processing internally (full-scale + half-scale).
    
    Args:
        pred (torch.Tensor): Predicted/restored image tensor [B, C, H, W] or [C, H, W] in range [0, 1]
        target (torch.Tensor, optional): Ignored (no-reference metric). Included for API consistency.
        valid_mask (torch.Tensor, optional): Ignored for ARNIQA.
        
    Returns:
        float: ARNIQA quality score (higher is better, typically in range [0, 1])
    """
    global _ARNIQA_METRIC_CACHE
    
    # Ensure tensor has batch dimension
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    
    device = pred.device
    device_key = str(device)
    
    # Load ARNIQA model from torchmetrics (cached per device)
    # Pattern from webserver_restoration/utils/metrics.py lines 46-58
    if device_key not in _ARNIQA_METRIC_CACHE:
        try:
            from torchmetrics.image.arniqa import ARNIQA
            arniqa_model = ARNIQA(
                regressor_dataset='kadid10k',  # Same as conditioning encoder
                normalize=True,
                reduction='none'
            )
            arniqa_model = arniqa_model.to(device)
            arniqa_model.eval()
            _ARNIQA_METRIC_CACHE[device_key] = arniqa_model
        except ImportError:
            import logging
            logging.warning("torchmetrics.image.arniqa not available. Install with: pip install torchmetrics[image]")
            return 0.0
        except Exception as e:
            import logging
            logging.warning(f"Failed to load ARNIQA model: {e}. Returning 0.0")
            return 0.0
    
    arniqa_model = _ARNIQA_METRIC_CACHE[device_key]
    
    # ARNIQA expects input in [0, 1] range
    # torchmetrics ARNIQA handles normalization and multi-scale internally
    # Use float16 to reduce memory: torchmetrics internally calls model.to(dtype=img.dtype)
    # so passing float16 input keeps the ResNet encoder in float16 (~50% less memory)
    with torch.no_grad():
        score = arniqa_model(pred.half())
    
    # Return mean score if batch
    return score.mean().item()

def delta_e(pred, target, valid_mask=None):
    """
    Compute mean CIEDE2000 color difference in L*a*b* color space.

    Args:
        pred (torch.Tensor): Predicted image tensor [B, 3, H, W] or [3, H, W] in range [0, 1]
        target (torch.Tensor): Target image tensor [B, 3, H, W] or [3, H, W] in range [0, 1]
        valid_mask (torch.Tensor, optional): Valid mask for pixels to consider

    Returns:
        float: Mean Delta E (CIEDE2000) value. Lower is better.
    """
    if not SKIMAGE_COLOR_AVAILABLE:
        raise ImportError(
            "scikit-image color module not available. "
            "Install with: pip install scikit-image"
        )

    # Ensure tensors have batch dimension
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(target.shape) == 3:
        target = target.unsqueeze(0)

    # Apply mask if provided
    if valid_mask is not None:
        if len(valid_mask.shape) == 3:
            valid_mask = valid_mask.unsqueeze(0)

    batch_size = pred.shape[0]
    delta_e_values = []

    for i in range(batch_size):
        # Convert from [3, H, W] to [H, W, 3] numpy in [0, 1]
        pred_np = pred[i].detach().cpu().permute(1, 2, 0).numpy()
        target_np = target[i].detach().cpu().permute(1, 2, 0).numpy()

        # Clip to [0, 1] to ensure valid input for rgb2lab
        pred_np = np.clip(pred_np, 0.0, 1.0)
        target_np = np.clip(target_np, 0.0, 1.0)

        # Convert RGB [0, 1] to L*a*b*
        pred_lab = rgb2lab(pred_np)
        target_lab = rgb2lab(target_np)

        # Compute per-pixel CIEDE2000
        de_map = deltaE_ciede2000(target_lab, pred_lab)

        # Apply mask if provided
        if valid_mask is not None:
            # valid_mask shape: [B, C, H, W] or [B, 1, H, W]
            mask_np = valid_mask[i, 0].detach().cpu().numpy().astype(bool)
            if mask_np.sum() > 0:
                delta_e_values.append(de_map[mask_np].mean())
            else:
                delta_e_values.append(0.0)
        else:
            delta_e_values.append(de_map.mean())

    return float(np.mean(delta_e_values))

