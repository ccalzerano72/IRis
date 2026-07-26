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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import logging

# Try to import pytorch-msssim, fallback to torchmetrics functional
try:
    from pytorch_msssim import ms_ssim as ms_ssim_func
    MSSSIM_AVAILABLE = 'pytorch_msssim'
except ImportError:
    try:
        from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as ms_ssim_func
        MSSSIM_AVAILABLE = 'torchmetrics_functional'
    except ImportError:
        MSSSIM_AVAILABLE = None


class DINOPerceptualLoss(nn.Module):
    """DINO-based perceptual loss for image restoration.
    
    Uses frozen DINOv2/v3 features to compute perceptual similarity.
    Modern alternative to VGG-based LPIPS — self-supervised features trained on 1.7B images.
    
    Reference: https://na-vae.github.io/dino_perceptual/
    
    Args:
        model_size (str): DINO model size ('S', 'B', 'L', 'H', 'G'). Default 'B' (ViT-Base, 86M params).
        version (str): DINO version ('v2' or 'v3'). Default 'v3'.
        target_size (int): Resize images before feature extraction. Default 256 (memory efficient).
        layers (str): Which transformer layers to use. Default 'all'.
        device (str): Device for computation.
    """
    
    def __init__(self, model_size='B', version='v3', target_size=256, layers='all', device='cuda'):
        super(DINOPerceptualLoss, self).__init__()
        
        try:
            from dino_perceptual import DINOPerceptual
        except ImportError:
            raise ImportError(
                "DINO perceptual loss requires 'dino-perceptual' package.\n"
                "Install with: pip install dino-perceptual"
            )
        
        self.dino_loss = DINOPerceptual(
            model_size=model_size,
            version=version,
            target_size=target_size,
            layers=layers,
        ).to(device).eval()
        
        # Freeze all parameters
        for param in self.dino_loss.parameters():
            param.requires_grad = False
        
        self.target_size = target_size
        logging.info(
            f"DINO perceptual loss initialized: DINOv{version[-1]}-{model_size}, "
            f"target_size={target_size}, layers={layers}"
        )
    
    def forward(self, pred, target):
        """Compute DINO perceptual loss.
        
        Args:
            pred (torch.Tensor): Predicted images [B, C, H, W] in [-1, 1]
            target (torch.Tensor): Target images [B, C, H, W] in [-1, 1]
            
        Returns:
            torch.Tensor: Scalar DINO perceptual loss
        """
        # DINOPerceptual expects [-1, 1] input and returns per-sample loss [B]
        return self.dino_loss(pred, target).mean()


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for image restoration.
    
    Uses pretrained VGG19 features to compute perceptual similarity.
    """
    
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg.to(device)
        
        # Define layer mapping
        self.layer_name_mapping = {
            'relu1_2': 3,   # After ReLU 1_2
            'relu2_2': 8,   # After ReLU 2_2  
            'relu3_3': 17,  # After ReLU 3_3
            'relu4_3': 26,  # After ReLU 4_3
            'relu5_3': 35   # After ReLU 5_3
        }
        
        self.target_layers = [self.layer_name_mapping[layer] for layer in layers]
        
        # Normalization for VGG (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def extract_features(self, x):
        """Extract features from target layers."""
        # Input should be in [0,1] range
        # Apply ImageNet normalization
        x_norm = self.normalize(x)
        
        features = []
        for i, layer in enumerate(self.vgg):
            x_norm = layer(x_norm)
            if i in self.target_layers:
                features.append(x_norm)
                
        return features
        
    def forward(self, pred, target):
        """Compute perceptual loss with automatic downsampling for memory efficiency."""
        # Downsample to 256×256 if images are larger (saves ~75% memory)
        # VGG perceptual loss doesn't need full resolution
        _, _, h, w = pred.shape
        if h > 256 or w > 256:
            pred = F.interpolate(pred, size=(256, 256), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)
        
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            # Normalize by number of elements to keep scale similar to MSE
            loss += F.mse_loss(pred_feat, target_feat)
            
        # Average across layers and normalize to [0,1] range like MSE
        loss = loss / len(pred_features)
        
        # Scale down to match MSE magnitude (VGG features are ~10× larger)
        loss = loss / 10.0
        
        return loss


class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM Loss for image restoration.
    
    Computes 1 - MS-SSIM to use as a loss (lower is better).
    MS-SSIM is more robust than single-scale SSIM and correlates better
    with human perception.
    
    Uses functional (stateless) implementation to avoid memory accumulation.
    """
    
    def __init__(self, data_range=1.0, device='cuda'):
        super(MSSSIMLoss, self).__init__()
        self.data_range = data_range
        
        if MSSSIM_AVAILABLE is None:
            raise ImportError(
                "MS-SSIM loss requires either 'pytorch-msssim' or 'torchmetrics' package.\n"
                "Install with: pip install pytorch-msssim"
            )
        
    def forward(self, pred, target):
        """Compute MS-SSIM loss.
        
        Args:
            pred (torch.Tensor): Predicted images [B, C, H, W] in [0, 1]
            target (torch.Tensor): Target images [B, C, H, W] in [0, 1]
            
        Returns:
            torch.Tensor: MS-SSIM loss (1 - MS-SSIM)
        """
        # Use functional (stateless) implementation
        if MSSSIM_AVAILABLE == 'pytorch_msssim':
            # pytorch-msssim: fast and memory-efficient
            ms_ssim_value = ms_ssim_func(pred, target, data_range=self.data_range, size_average=True)
        else:
            # torchmetrics functional: also stateless
            ms_ssim_value = ms_ssim_func(pred, target, data_range=self.data_range)
        
        # MS-SSIM returns value in [0, 1] where 1 is perfect
        # We return 1 - MS-SSIM so that minimizing the loss improves quality
        return 1.0 - ms_ssim_value


class CombinedRestorationLoss(nn.Module):
    """Combined loss for image restoration: MSE + L1 + Perceptual + MS-SSIM + LPIPS + DINO + E-LatentLPIPS.
    
    Flexible loss function that supports multiple components:
    - MSE: Pixel-wise L2 accuracy (optional)
    - L1: Pixel-wise L1 accuracy (optional)
    - VGG Perceptual: Feature-based perceptual similarity (optional)
    - MS-SSIM: Multi-scale structural similarity (optional)
    - LPIPS: Learned perceptual similarity in RGB space (optional)
    - DINO: DINOv2/v3-based perceptual similarity in RGB space (optional)
    - E-LatentLPIPS: Perceptual similarity in latent space (optional)
    
    Args:
        mse_weight (float): Weight for MSE loss (0 to disable)
        l1_weight (float): Weight for L1 loss (0 to disable)
        perceptual_weight (float): Weight for VGG perceptual loss (0 to disable)
        ms_ssim_weight (float): Weight for MS-SSIM loss (0 to disable)
        lpips_weight (float): Weight for LPIPS loss (0 to disable)
        dino_weight (float): Weight for DINO perceptual loss (0 to disable)
        dino_model_size (str): DINO model size ('S', 'B', 'L', 'H', 'G')
        dino_version (str): DINO version ('v2' or 'v3')
        dino_target_size (int): Resize images before DINO feature extraction
        dino_layers (str): Which DINO transformer layers to use
        elatentlpips_weight (float): Weight for E-LatentLPIPS loss (0 to disable)
        perceptual_layers (list): VGG layers for perceptual loss
        lpips_net (str): LPIPS network ('alex', 'vgg', 'squeeze')
        elatentlpips_encoder (str): E-LatentLPIPS encoder ('sd15', 'sd21', 'sdxl', 'sd3', 'flux')
        elatentlpips_augment (str): E-LatentLPIPS augmentation ('b', 'bg', 'bgc', 'bgco')
        device (str): Device for computation
    """
    
    def __init__(self, 
                 mse_weight=1.0,
                 l1_weight=0.0,
                 perceptual_weight=0.1, 
                 ms_ssim_weight=0.0,
                 lpips_weight=0.05,
                 dino_weight=0.0,
                 dino_model_size='B',
                 dino_version='v3',
                 dino_target_size=256,
                 dino_layers='all',
                 elatentlpips_weight=0.0,
                 perceptual_layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                 lpips_net='alex',
                 elatentlpips_encoder='sd21',
                 elatentlpips_augment='bg',
                 device='cuda'):
        super(CombinedRestorationLoss, self).__init__()
        
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.lpips_weight = lpips_weight
        self.dino_weight = dino_weight
        self.elatentlpips_weight = elatentlpips_weight
        
        # Initialize loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # VGG Perceptual Loss (optional)
        if perceptual_weight > 0:
            self.perceptual_loss = VGGPerceptualLoss(layers=perceptual_layers, device=device)
        else:
            self.perceptual_loss = None
        
        # MS-SSIM Loss (optional)
        if ms_ssim_weight > 0:
            self.ms_ssim_loss = MSSSIMLoss(data_range=1.0, device=device)
        else:
            self.ms_ssim_loss = None
            
        # LPIPS Loss (optional)
        if lpips_weight > 0:
            import lpips
            self.lpips_loss = lpips.LPIPS(net=lpips_net).to(device)
            # Freeze LPIPS parameters
            for param in self.lpips_loss.parameters():
                param.requires_grad = False
        else:
            self.lpips_loss = None
        
        # DINO Perceptual Loss (optional)
        if dino_weight > 0:
            self.dino_loss = DINOPerceptualLoss(
                model_size=dino_model_size,
                version=dino_version,
                target_size=dino_target_size,
                layers=dino_layers,
                device=device,
            )
        else:
            self.dino_loss = None
        
        # E-LatentLPIPS Loss (optional)
        if elatentlpips_weight > 0:
            try:
                from elatentlpips import ELatentLPIPS
                self.elatentlpips_loss = ELatentLPIPS(
                    encoder=elatentlpips_encoder,
                    augment=elatentlpips_augment
                ).to(device).eval()
                # Freeze E-LatentLPIPS parameters
                for param in self.elatentlpips_loss.parameters():
                    param.requires_grad = False
                logging.info(f"E-LatentLPIPS initialized: encoder={elatentlpips_encoder}, augment={elatentlpips_augment}")
            except ImportError:
                logging.warning("E-LatentLPIPS not available. Install with: pip install elatentlpips")
                self.elatentlpips_loss = None
        else:
            self.elatentlpips_loss = None
            
    def forward(self, pred, target):
        """Compute combined loss.
        
        Args:
            pred (torch.Tensor): Predicted images [B, C, H, W] in [0, 1]
            target (torch.Tensor): Target images [B, C, H, W] in [0, 1]
            
        Returns:
            tuple: (total_loss, loss_dict) where loss_dict contains individual components
        """
        losses = {}
        total_loss = 0.0
        
        # MSE Loss
        if self.mse_weight > 0:
            mse_loss = self.mse_loss(pred, target)
            losses['mse'] = mse_loss
            total_loss += self.mse_weight * mse_loss
        
        # L1 Loss
        if self.l1_weight > 0:
            l1_loss = self.l1_loss(pred, target)
            losses['l1'] = l1_loss
            total_loss += self.l1_weight * l1_loss
            
        # VGG Perceptual Loss
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual_loss
            total_loss += self.perceptual_weight * perceptual_loss
        
        # MS-SSIM Loss
        if self.ms_ssim_loss is not None and self.ms_ssim_weight > 0:
            ms_ssim_loss = self.ms_ssim_loss(pred, target)
            losses['ms_ssim'] = ms_ssim_loss
            total_loss += self.ms_ssim_weight * ms_ssim_loss
            
        # LPIPS Loss
        if self.lpips_loss is not None and self.lpips_weight > 0:
            # Downsample to 256×256 if images are larger (saves ~75% memory)
            # LPIPS doesn't need full resolution
            _, _, h, w = pred.shape
            if h > 25600 or w > 25600:
                pred_down = F.interpolate(pred, size=(256, 256), mode='bilinear', align_corners=False)
                target_down = F.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)
            else:
                pred_down = pred
                target_down = target
            
            # LPIPS expects inputs in [-1, 1]
            pred_lpips = pred_down * 2.0 - 1.0
            target_lpips = target_down * 2.0 - 1.0
            lpips_loss = self.lpips_loss(pred_lpips, target_lpips).mean()
            losses['lpips'] = lpips_loss
            total_loss += self.lpips_weight * lpips_loss
            
        # DINO Perceptual Loss
        if self.dino_loss is not None and self.dino_weight > 0:
            # DINO expects [-1, 1]; pred/target are in [0, 1]
            pred_dino = pred * 2.0 - 1.0
            target_dino = target * 2.0 - 1.0
            dino_loss = self.dino_loss(pred_dino, target_dino)
            losses['dino'] = dino_loss
            total_loss += self.dino_weight * dino_loss
            
        losses['total'] = total_loss
        return total_loss, losses
    
    def compute_elatentlpips(self, pred_latent, target_latent):
        """Compute E-LatentLPIPS loss on latent representations.
        
        Args:
            pred_latent (torch.Tensor): Predicted latents [B, 4, h, w]
            target_latent (torch.Tensor): Target latents [B, 4, h, w]
            
        Returns:
            torch.Tensor: E-LatentLPIPS loss value
        """
        if self.elatentlpips_loss is None or self.elatentlpips_weight == 0:
            return torch.tensor(0.0, device=pred_latent.device)
        
        # E-LatentLPIPS expects normalized latents
        # Set normalize=True to apply VAE scaling_factor and shift_factor
        elatentlpips_loss = self.elatentlpips_loss(
            pred_latent, 
            target_latent, 
            normalize=True
        ).mean()
        
        return elatentlpips_loss


class LatentGradientLoss(nn.Module):
    """SNR-weighted gradient loss in latent space to preserve sharp edges and reduce blurriness.
    
    Applies Sobel filters (horizontal + vertical) to each latent channel,
    then computes L1 between the gradient maps of predicted and target latents.
    
    This penalizes the loss of high-frequency content (edges, textures) that
    L2/L1 pixel-wise losses tend to smooth out.
    
    SNR Weighting (curriculum-aware sharpness):
        When sample_weights (alpha_t) are provided, the loss is weighted per-sample:
            grad_loss_i = alpha_t[i] * L1(sobel(pred_i), sobel(target_i))
        
        This creates a curriculum effect:
        - High t (alpha_t ≈ 0): x_0 prediction is unreliable → gradient loss ≈ 0
        - Low t (alpha_t ≈ 1): x_0 prediction is accurate → full gradient loss
        
        The model first learns global structure (via noise prediction loss) and
        the gradient loss progressively "takes over" at low timesteps to sculpt
        fine details and sharpness.
        
        Without this weighting, high-t samples produce noisy/exploding gradients
        that destabilize training (especially ARNIQA adapter weights).
    
    Zero trainable parameters — uses fixed Sobel kernels as buffers.
    Works directly on 4-channel latent tensors [B, 4, h, w].
    """
    
    def __init__(self):
        super(LatentGradientLoss, self).__init__()
        
        # Sobel kernels for horizontal and vertical gradients
        # Shape: [1, 1, 3, 3] — will be applied per-channel via groups=C
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        
        # Register as buffers (not parameters — no gradients on kernels)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _compute_gradients(self, x):
        """Apply Sobel filters to each channel independently.
        
        Args:
            x: Latent tensor [B, C, H, W]
            
        Returns:
            Tuple of (grad_x, grad_y), each [B, C, H, W]
        """
        channels = x.shape[1]
        
        # Expand kernels to match number of channels: [C, 1, 3, 3]
        kernel_x = self.sobel_x.expand(channels, -1, -1, -1)
        kernel_y = self.sobel_y.expand(channels, -1, -1, -1)
        
        # Apply per-channel convolution (groups=C)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
        
        return grad_x, grad_y
    
    def forward(self, pred, target, sample_weights=None):
        """Compute gradient loss between predicted and target latents.
        
        Args:
            pred: Predicted (denoised) latent [B, 4, h, w]
            target: Target (clean) latent [B, 4, h, w]
            sample_weights: Optional per-sample weights [B] (e.g., alpha_t for SNR weighting).
                When provided, computes weighted per-sample loss.
                When None, computes simple mean over batch.
            
        Returns:
            Scalar gradient loss (L1 on gradient maps, optionally SNR-weighted)
        """
        pred_gx, pred_gy = self._compute_gradients(pred)
        target_gx, target_gy = self._compute_gradients(target)
        
        if sample_weights is not None:
            # Per-sample loss: compute L1 per sample, then weight by alpha_t
            # F.l1_loss with reduction='none' → [B, C, H, W]
            # Mean over (C, H, W) → [B]
            loss_x = F.l1_loss(pred_gx, target_gx, reduction='none').mean(dim=(1, 2, 3))  # [B]
            loss_y = F.l1_loss(pred_gy, target_gy, reduction='none').mean(dim=(1, 2, 3))  # [B]
            per_sample_loss = loss_x + loss_y  # [B]
            
            # Weight by sample_weights (alpha_t) and average over batch
            weighted_loss = (per_sample_loss * sample_weights).mean()
            return weighted_loss
        else:
            # Simple mean over entire batch (original behavior)
            loss_x = F.l1_loss(pred_gx, target_gx)
            loss_y = F.l1_loss(pred_gy, target_gy)
            return loss_x + loss_y

    @torch.no_grad()
    def visualize_gradients(self, pred, target):
        """Generate gradient magnitude maps for visualization.
        
        Computes Sobel gradient magnitude for predicted and target latents,
        averages across latent channels, and returns normalized grayscale maps.
        
        Args:
            pred: Predicted (denoised) latent [B, 4, h, w]
            target: Target (clean) latent [B, 4, h, w]
            
        Returns:
            Dict with numpy arrays (first sample, [h, w] in [0, 1]):
            - 'grad_target': Gradient magnitude of target (clean) latent
            - 'grad_pred': Gradient magnitude of predicted latent
            - 'grad_diff': Absolute difference between target and pred gradients
        """
        import numpy as np
        
        # Compute gradients for both
        pred_gx, pred_gy = self._compute_gradients(pred)
        target_gx, target_gy = self._compute_gradients(target)
        
        # Gradient magnitude: sqrt(gx² + gy²), averaged across channels
        pred_mag = torch.sqrt(pred_gx ** 2 + pred_gy ** 2).mean(dim=1, keepdim=True)  # [B, 1, h, w]
        target_mag = torch.sqrt(target_gx ** 2 + target_gy ** 2).mean(dim=1, keepdim=True)  # [B, 1, h, w]
        
        # Absolute difference
        diff_mag = torch.abs(target_mag - pred_mag)  # [B, 1, h, w]
        
        # Take first sample, squeeze to [h, w], normalize to [0, 1]
        def _normalize(t):
            t = t[0, 0]  # [h, w]
            t_min, t_max = t.min(), t.max()
            if t_max - t_min > 1e-8:
                t = (t - t_min) / (t_max - t_min)
            else:
                t = torch.zeros_like(t)
            return t.cpu().numpy()
        
        return {
            'grad_target': _normalize(target_mag),
            'grad_pred': _normalize(pred_mag),
            'grad_diff': _normalize(diff_mag),
        }


def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        criterion = torch.nn.MSELoss(**kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    elif "combined_restoration_loss" == loss_name:
        criterion = CombinedRestorationLoss(**kwargs)
    else:
        raise NotImplementedError

    return criterion


class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(depth_gt)

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(depth_gt)
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss
