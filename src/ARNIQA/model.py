"""
ARNIQA Quality-Aware Conditioning Module

This module provides the ARNIQA encoder and adapters for extracting
quality-aware features to condition the restoration U-Net.

Stage 1: Global-only conditioning (2048-dim global feature → 1024-dim embedding)
Stage 2: Global + Spatial conditioning (Layer 3 spatial features → 576 tokens + 1 global token)

Reference:
    Agnolucci et al., "ARNIQA: Learning Distortion Manifold for Image Quality Assessment", WACV 2024
"""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# ImageNet normalization constants (ARNIQA uses ImageNet-pretrained ResNet-50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Positional Embedding Utilities
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    Generate 2D sinusoidal positional embeddings.
    
    Following the ViT/MAE approach: use sin/cos functions at different frequencies
    for x and y coordinates separately, then concatenate.
    
    Args:
        embed_dim: Embedding dimension (must be divisible by 4)
        grid_size: Spatial grid size (e.g., 24 for 24×24 = 576 tokens)
    
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] numpy array
    """
    assert embed_dim % 4 == 0, f"embed_dim must be divisible by 4, got {embed_dim}"
    
    # Create grid coordinates
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # [grid_size, grid_size] each
    
    # Flatten to [grid_size*grid_size]
    grid_h = grid_h.flatten()
    grid_w = grid_w.flatten()
    
    # Compute positional embeddings for each dimension
    # Each coordinate gets embed_dim/2 dimensions (half for sin, half for cos)
    half_dim = embed_dim // 2
    quarter_dim = embed_dim // 4
    
    # Frequency bands (following transformer convention)
    omega = np.arange(quarter_dim, dtype=np.float32)
    omega = 1.0 / (10000 ** (omega / quarter_dim))
    
    # Compute sin/cos for height
    out_h = np.outer(grid_h, omega)  # [grid_size*grid_size, quarter_dim]
    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)  # [N, half_dim]
    
    # Compute sin/cos for width
    out_w = np.outer(grid_w, omega)  # [grid_size*grid_size, quarter_dim]
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)  # [N, half_dim]
    
    # Concatenate height and width embeddings
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # [N, embed_dim]
    
    return pos_embed


class ArniqaEncoder(nn.Module):
    """
    ARNIQA Encoder: ResNet-50 backbone for quality feature extraction.
    
    Extracts features at multiple levels:
    - Global: 2048-dim vector (after global average pooling of layer4)
    - Layer 3: [B, 1024, H/16, W/16] spatial features (for Stage 2)
    - Layer 4: [B, 2048, H/32, W/32] spatial features (for Stage 3+)
    
    The encoder is loaded with ARNIQA pretrained weights from torch.hub.
    It should be kept FROZEN during training to preserve quality representations.
    
    ARNIQA's encoder uses nn.Sequential with indexed layers:
    - model[0-3]: conv1, bn1, relu, maxpool
    - model[4]: layer1 (256 channels)
    - model[5]: layer2 (512 channels)
    - model[6]: layer3 (1024 channels) ← Used for Stage 2 spatial features
    - model[7]: layer4 (2048 channels) ← Used for global features
    - model[8]: avgpool
    
    Args:
        stage: Conditioning stage (1=global only, 2=global+spatial)
    """
    
    # Expected channel counts for ResNet-50 layers (safety validation)
    EXPECTED_LAYER3_CHANNELS = 1024
    EXPECTED_LAYER4_CHANNELS = 2048
    
    def __init__(self, stage: int = 1):
        super().__init__()
        
        self.stage = stage
        self._first_forward = True  # For debug logging
        
        logging.info(f"Loading ARNIQA encoder for Stage {stage}")
        
        try:
            arniqa_model = torch.hub.load(
                repo_or_dir="miccunifi/ARNIQA",
                model="ARNIQA",
                regressor_dataset="kadid10k",
            )
            
            # Extract the encoder (ResNet-50 backbone wrapped in Sequential)
            self.encoder = arniqa_model.encoder
            
            # Validate layer structure for Stage 2+
            if stage >= 2:
                self._validate_layer_structure()
            
            logging.info("ARNIQA encoder loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load ARNIQA from torch.hub: {e}")
            raise RuntimeError(
                f"ARNIQA encoder loading failed. Stage {stage} requires ARNIQA weights. "
                "Ensure internet connection and torch.hub access."
            ) from e
        
        # Global average pooling for global features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Register ImageNet normalization as buffers (for device handling)
        self.register_buffer(
            'mean', 
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', 
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        
        # Freeze encoder by default
        self.freeze()
    
    def _validate_layer_structure(self):
        """
        Validate that ARNIQA encoder has expected layer structure.
        
        Checks:
        - model[6] (Layer 3) outputs 1024 channels
        - model[7] (Layer 4) outputs 2048 channels
        
        Raises RuntimeError if structure doesn't match expectations.
        """
        model = self.encoder.model
        
        # Check model is Sequential with enough layers
        if not hasattr(model, '__getitem__') or len(model) < 8:
            raise RuntimeError(
                f"ARNIQA encoder.model has unexpected structure. "
                f"Expected Sequential with 9 layers, got {type(model)} with {len(model) if hasattr(model, '__len__') else '?'} layers."
            )
        
        # Validate Layer 3 (model[6]) - should output 1024 channels
        # Check the last conv in the last bottleneck block
        layer3 = model[6]
        if hasattr(layer3, '__getitem__') and len(layer3) > 0:
            last_block = layer3[-1]
            if hasattr(last_block, 'conv3'):
                l3_channels = last_block.conv3.out_channels
                if l3_channels != self.EXPECTED_LAYER3_CHANNELS:
                    raise RuntimeError(
                        f"ARNIQA Layer 3 has {l3_channels} output channels, "
                        f"expected {self.EXPECTED_LAYER3_CHANNELS}. "
                        "Backbone structure may have changed."
                    )
        
        # Validate Layer 4 (model[7]) - should output 2048 channels
        layer4 = model[7]
        if hasattr(layer4, '__getitem__') and len(layer4) > 0:
            last_block = layer4[-1]
            if hasattr(last_block, 'conv3'):
                l4_channels = last_block.conv3.out_channels
                if l4_channels != self.EXPECTED_LAYER4_CHANNELS:
                    raise RuntimeError(
                        f"ARNIQA Layer 4 has {l4_channels} output channels, "
                        f"expected {self.EXPECTED_LAYER4_CHANNELS}. "
                        "Backbone structure may have changed."
                    )
        
        logging.info(
            f"ARNIQA layer structure validated: "
            f"Layer3={self.EXPECTED_LAYER3_CHANNELS}ch, "
            f"Layer4={self.EXPECTED_LAYER4_CHANNELS}ch"
        )
    
    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        logging.info("ARNIQA encoder frozen")
    
    def unfreeze(self):
        """Unfreeze encoder parameters (not recommended)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logging.warning("ARNIQA encoder unfrozen - this may degrade quality representations")
    
    def train(self, mode: bool = True):
        """Override train to keep encoder in eval mode."""
        # Keep encoder in eval mode even during training
        # This ensures BatchNorm uses running stats, not batch stats
        super().train(mode)
        self.encoder.eval()
        return self
    
    def _normalize_input(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Normalize input from Marigold format to ARNIQA format.
        
        Args:
            rgb_in: RGB image in [-1, 1] range (Marigold convention)
        
        Returns:
            Normalized image for ARNIQA (ImageNet normalization)
        """
        # Convert from [-1, 1] to [0, 1]
        rgb_01 = (rgb_in + 1.0) / 2.0
        
        # Apply ImageNet normalization
        rgb_norm = (rgb_01 - self.mean) / self.std
        
        return rgb_norm
    
    def forward_features(self, rgb_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features at multiple levels in a single forward pass.
        
        For Stage 1: Only global features are computed (via full forward)
        For Stage 2+: Intercept Layer 3 output mid-way, then continue to Layer 4
        
        This single-pass approach is efficient: the image passes through
        the backbone once, and we capture intermediate outputs as needed.
        
        Args:
            rgb_in: RGB image in [-1, 1] range (Marigold convention)
                   Shape: [B, 3, H, W]
        
        Returns:
            Dictionary with:
            - 'global': [B, 2048] global feature vector
            - 'layer3': [B, 1024, H/16, W/16] spatial features (None if stage=1)
        """
        # Normalize input
        x = self._normalize_input(rgb_in)
        
        model = self.encoder.model
        
        # Forward through early layers (conv1 -> maxpool -> layer1 -> layer2)
        x = model[0](x)  # conv1
        x = model[1](x)  # bn1
        x = model[2](x)  # relu
        x = model[3](x)  # maxpool
        x = model[4](x)  # layer1
        x = model[5](x)  # layer2
        
        # Layer 3: capture output for Stage 2 spatial features
        layer3_out = model[6](x)  # [B, 1024, H/16, W/16]
        
        # Layer 4: continue for global features
        layer4_out = model[7](layer3_out)  # [B, 2048, H/32, W/32]
        
        # Global average pooling
        global_feat = self.global_pool(layer4_out)  # [B, 2048, 1, 1]
        global_feat = global_feat.flatten(1)  # [B, 2048]
        
        # Debug logging on first forward (Stage 2)
        if self._first_forward and self.stage >= 2:
            logging.info(
                f"ARNIQA forward_features shapes: "
                f"layer3={list(layer3_out.shape)}, "
                f"layer4={list(layer4_out.shape)}, "
                f"global={list(global_feat.shape)}"
            )
            self._first_forward = False
        
        return {
            'global': global_feat,
            'layer3': layer3_out if self.stage >= 2 else None,
        }
    
    def forward(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only global features (for Stage 1 compatibility).
        
        Args:
            rgb_in: RGB image in [-1, 1] range
        
        Returns:
            Global feature vector [B, 2048]
        """
        features = self.forward_features(rgb_in)
        return features['global']


# =============================================================================
# Spatial Adapter (Stage 2)
# =============================================================================

class ArniqaSpatialAdapter(nn.Module):
    """
    Spatial Adapter for Stage 2: Projects Layer 3 spatial features to tokens.
    
    This adapter processes the spatial feature map from ARNIQA's Layer 3
    and converts it into a sequence of tokens for cross-attention.
    
    Pipeline:
    1. Adaptive pooling to fixed spatial size (default 24×24)
    2. 1×1 Conv2d projection to output_dim channels
    3. Flatten to sequence: [B, C, H, W] → [B, H*W, C]
    4. Add 2D sinusoidal positional embeddings
    5. Final projection with LayerNorm
    
    Output format: [B, spatial_size², output_dim] (e.g., [B, 576, 1024] for 24×24)
    
    Key design choices:
    - Conv2d 1×1 for projection (efficient on 2D data, preserves locality)
    - Positional embeddings as buffer (computed once, not per forward)
    - Zero initialization for final layer (start in "unconditional mode")
    - LayerNorm at output for stable cross-attention
    
    Args:
        input_channels: Input channels from Layer 3 (1024 for ResNet-50)
        output_dim: Output dimension (1024 for SD v2 cross-attention)
        spatial_size: Target spatial size after pooling (default: 24 → 576 tokens)
    """
    
    def __init__(
        self,
        input_channels: int = 1024,
        output_dim: int = 1024,
        spatial_size: int = 24,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size  # 576 for 24×24
        
        # Adaptive pooling to fixed spatial size
        self.spatial_pool = nn.AdaptiveAvgPool2d(spatial_size)
        
        # 1×1 Conv projection (more efficient than Linear on 2D data)
        # Projects from input_channels to output_dim
        self.conv_proj = nn.Conv2d(input_channels, output_dim, kernel_size=1, bias=True)
        
        # Final projection with LayerNorm (applied after flattening)
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Positional embeddings: computed once, registered as buffer
        # Shape: [1, num_tokens, output_dim] for easy broadcasting
        pos_embed_np = get_2d_sincos_pos_embed(output_dim, spatial_size)
        pos_embed = torch.from_numpy(pos_embed_np).float().unsqueeze(0)  # [1, N, D]
        self.register_buffer('pos_embed', pos_embed, persistent=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights with zero-init for final layers.
        
        CRITICAL: Both conv_proj and final_proj must output zeros at init
        to ensure the spatial adapter starts in "unconditional mode".
        Combined with the global adapter's zero-init, the total ARNIQA
        conditioning signal is zero at the start of training.
        """
        # Save current RNG state
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
        # Deterministic initialization
        torch.manual_seed(43)  # Different seed from global adapter
        if torch.cuda.is_available():
            torch.cuda.manual_seed(43)
        
        # CRITICAL: Zero-init for conv projection
        # This ensures spatial features contribute nothing at init
        nn.init.zeros_(self.conv_proj.weight)
        nn.init.zeros_(self.conv_proj.bias)
        
        # CRITICAL: Zero-init for final Linear layer (index 0 in final_proj)
        nn.init.zeros_(self.final_proj[0].weight)
        nn.init.zeros_(self.final_proj[0].bias)
        
        # Restore RNG state
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        
        logging.info(
            f"ArniqaSpatialAdapter initialized with zero-init "
            f"(spatial_size={self.spatial_size}, num_tokens={self.num_tokens})"
        )
    
    def forward(self, layer3_features: torch.Tensor) -> torch.Tensor:
        """
        Project Layer 3 spatial features to cross-attention tokens.
        
        Args:
            layer3_features: [B, 1024, H/16, W/16] spatial features from ARNIQA
        
        Returns:
            Spatial tokens [B, num_tokens, output_dim] ready for cross-attention
        """
        batch_size = layer3_features.shape[0]
        
        # 1. Adaptive pooling to fixed spatial size
        # [B, 1024, H/16, W/16] → [B, 1024, spatial_size, spatial_size]
        x = self.spatial_pool(layer3_features)
        
        # 2. 1×1 Conv projection
        # [B, 1024, S, S] → [B, output_dim, S, S]
        x = self.conv_proj(x)
        
        # 3. Flatten to sequence: [B, C, H, W] → [B, H*W, C]
        # Permute: [B, C, H, W] → [B, H, W, C] → [B, H*W, C]
        x = x.permute(0, 2, 3, 1)  # [B, S, S, C]
        x = x.reshape(batch_size, self.num_tokens, self.output_dim)  # [B, N, C]
        
        # 4. Add positional embeddings (broadcast across batch)
        x = x + self.pos_embed  # [B, N, C] + [1, N, C] → [B, N, C]
        
        # 5. Final projection with LayerNorm
        x = self.final_proj(x)  # [B, N, C]
        
        return x


# =============================================================================
# Global Adapter (Stage 1 and 2)
# =============================================================================


class ArniqaGlobalAdapter(nn.Module):
    """
    Global Adapter: Projects 2048-dim ARNIQA features to cross-attention tokens.
    
    This adapter maps the global quality features to the dimension expected
    by the U-Net cross-attention (matching CLIP text encoder output dimension).
    
    Output format: [B, num_tokens, 1024] - distinct quality tokens per sample.
    
    When num_tokens > 1, the final Linear projects to output_dim * num_tokens,
    then reshapes to [B, num_tokens, output_dim]. Each token gets its own
    learned projection from the 2048-dim feature, allowing specialization
    (e.g., one token for degradation type, another for severity).
    
    In Stage 2, always outputs [B, 1, 1024] regardless of num_tokens setting
    (the spatial adapter provides additional tokens).
    
    Key design choices:
    - LayerNorm per token at output (ensures stable magnitude for cross-attention)
    - Small-scale Xavier initialization for final Linear layer (non-zero signal
      from step 0, forcing the U-Net to account for conditioning immediately)
    - GELU activation (matches transformer conventions)
    
    Args:
        input_dim: Input feature dimension (2048 for ResNet-50)
        output_dim: Output dimension per token (1024 for SD v2 cross-attention)
        hidden_dim: Hidden layer dimension (default: 1024)
        dropout: Dropout probability (default: 0.0)
        num_tokens: Number of distinct output tokens (default: 4)
        init_scale: Xavier gain for final Linear layer (default: 0.01)
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 1024,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
        num_tokens: int = 4,
        init_scale: float = 0.01,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.init_scale = init_scale
        
        # MLP: input_dim → hidden_dim → (output_dim * num_tokens)
        # Final Linear projects to all tokens at once, then reshape
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim * num_tokens),
        )
        
        # LayerNorm applied per-token after reshape
        # Ensures stable magnitude for cross-attention
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights with small-scale Xavier for final Linear layer.
        
        Unlike zero-init (used in previous experiments), this provides a
        non-zero conditioning signal from step 0, forcing the U-Net to
        account for the varying ARNIQA input from the start of training.
        
        The small scale (default gain=0.01) keeps the initial perturbation
        small enough to avoid destabilizing pretrained U-Net weights, while
        being large enough to produce a meaningful gradient signal.
        
        Uses isolated RNG state to ensure deterministic initialization
        without affecting the global random state.
        """
        # Save current RNG state to restore after initialization
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
        # Set deterministic seed for adapter initialization only
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Standard initialization for first layer
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        
        # Small-scale Xavier initialization for final Linear layer (index 3)
        # This provides non-zero conditioning from step 0
        nn.init.xavier_uniform_(self.mlp[3].weight, gain=self.init_scale)
        nn.init.zeros_(self.mlp[3].bias)
        
        # Restore RNG state to not affect rest of training
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        
        logging.info(
            f"ArniqaGlobalAdapter initialized with small-scale Xavier "
            f"(gain={self.init_scale}, seed=42, num_tokens={self.num_tokens})"
        )
    
    def forward(
        self, 
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project global features to cross-attention format.
        
        Args:
            global_features: [B, 2048] global quality features from ARNIQA
        
        Returns:
            Quality tokens [B, num_tokens, output_dim] ready for cross-attention.
            Each token is a distinct learned projection from the global features.
        """
        batch_size = global_features.shape[0]
        
        # Project: [B, 2048] → [B, output_dim * num_tokens]
        projected = self.mlp(global_features)
        
        # Reshape to distinct tokens: [B, output_dim * num_tokens] → [B, num_tokens, output_dim]
        projected = projected.view(batch_size, self.num_tokens, self.output_dim)
        
        # Apply LayerNorm per token for stable cross-attention magnitude
        projected = self.layer_norm(projected)
        
        return projected


class ArniqaConditioner(nn.Module):
    """
    Complete ARNIQA conditioning module combining encoder and adapters.
    
    This is the main interface for ARNIQA quality-aware conditioning.
    It handles:
    - Feature extraction (encoder, frozen)
    - Feature projection (adapters, trainable)
    - Conditioning dropout (for CFG-like training)
    
    Stage 1: Global-only conditioning
        - Output: [B, num_tokens, 1024] (default num_tokens=1)
        - Only global_adapter is used
    
    Stage 2: Global + Spatial conditioning (Joint Optimization)
        - Output: [B, 1 + spatial_size², 1024] = [B, 577, 1024] for spatial_size=24
        - Both global_adapter and spatial_adapter are used
        - Global token is always 1 (num_tokens ignored in Stage 2)
        - Spatial tokens: spatial_size² (576 for 24×24)
    
    CRITICAL: Both adapters use zero-init for their final layers.
    This ensures the combined output is zero at initialization,
    allowing the model to start in "unconditional mode" and gradually
    learn to use the quality features.
    
    Args:
        output_dim: Output dimension for cross-attention (default: 1024)
        conditioning_dropout: Probability of dropping conditioning (default: 0.1)
        stage: Conditioning stage (1=global only, 2=global+spatial)
        spatial_size: Spatial grid size for Stage 2 (default: 24 → 576 tokens)
        num_tokens: Number of global tokens for Stage 1 only (default: 1)
    """
    
    def __init__(
        self,
        output_dim: int = 1024,
        conditioning_dropout: float = 0.1,
        stage: int = 1,
        spatial_size: int = 24,
        num_tokens: int = 1,
    ):
        super().__init__()
        
        self.conditioning_dropout = conditioning_dropout
        self.output_dim = output_dim
        self.stage = stage
        self.spatial_size = spatial_size
        self.num_tokens = num_tokens
        self._first_forward = True  # For debug logging
        
        # Encoder (frozen) - stage determines whether Layer 3 is extracted
        self.encoder = ArniqaEncoder(stage=stage)
        
        # Global adapter (trainable) - always present
        # In Stage 2, always outputs [B, 1, 1024] regardless of num_tokens
        self.global_adapter = ArniqaGlobalAdapter(
            input_dim=2048,
            output_dim=output_dim,
            num_tokens=1 if stage >= 2 else num_tokens,  # Stage 2: always 1 token
        )
        
        # Spatial adapter (trainable) - only for Stage 2+
        self.spatial_adapter = None
        if stage >= 2:
            self.spatial_adapter = ArniqaSpatialAdapter(
                input_channels=1024,  # Layer 3 output channels
                output_dim=output_dim,
                spatial_size=spatial_size,
            )
            
            # Calculate total tokens for logging
            total_tokens = 1 + (spatial_size * spatial_size)  # global + spatial
            logging.info(
                f"ArniqaConditioner Stage {stage}: "
                f"global=1 token, spatial={spatial_size}×{spatial_size}={spatial_size**2} tokens, "
                f"total={total_tokens} tokens"
            )
        else:
            logging.info(
                f"ArniqaConditioner Stage {stage}: "
                f"global={num_tokens} token(s)"
            )
    
    def forward(
        self,
        rgb_in: torch.Tensor,
        apply_dropout: bool = True,
    ) -> torch.Tensor:
        """
        Extract and project ARNIQA features for U-Net conditioning.
        
        Args:
            rgb_in: RGB image in [-1, 1] range, shape [B, 3, H, W]
            apply_dropout: Whether to apply conditioning dropout (training only)
        
        Returns:
            Quality conditioning tokens:
            - Stage 1: [B, num_tokens, output_dim]
            - Stage 2: [B, 1 + spatial_size², output_dim]
        """
        batch_size = rgb_in.shape[0]
        device = rgb_in.device
        
        # Extract features (encoder is frozen, no gradients)
        with torch.no_grad():
            features = self.encoder.forward_features(rgb_in)
            global_feat = features['global']  # [B, 2048]
            layer3_feat = features.get('layer3')  # [B, 1024, H/16, W/16] or None
        
        # Project global features: [B, 2048] → [B, 1, 1024] (or [B, num_tokens, 1024] for Stage 1)
        global_tokens = self.global_adapter(global_feat)
        
        # Stage 2: Add spatial tokens
        if self.stage >= 2 and self.spatial_adapter is not None and layer3_feat is not None:
            # Project spatial features: [B, 1024, H/16, W/16] → [B, spatial_size², 1024]
            spatial_tokens = self.spatial_adapter(layer3_feat)
            
            # Concatenate: [B, 1, 1024] + [B, 576, 1024] → [B, 577, 1024]
            quality_tokens = torch.cat([global_tokens, spatial_tokens], dim=1)
            
            # Debug logging on first forward
            if self._first_forward:
                logging.info(
                    f"ARNIQA Stage 2 output shape: {list(quality_tokens.shape)} "
                    f"(global={list(global_tokens.shape)}, spatial={list(spatial_tokens.shape)})"
                )
                self._first_forward = False
        else:
            quality_tokens = global_tokens
            
            # Debug logging on first forward (Stage 1)
            if self._first_forward:
                logging.info(f"ARNIQA Stage 1 output shape: {list(quality_tokens.shape)}")
                self._first_forward = False
        
        # Conditioning dropout (during training)
        # CRITICAL: Apply to ENTIRE block (global + spatial together)
        # This simulates unconditional generation for CFG-like training
        if self.training and apply_dropout and self.conditioning_dropout > 0:
            # Create dropout mask: 1.0 to keep, 0.0 to drop
            # Shape: [B, 1, 1] broadcasts to [B, num_tokens, output_dim]
            dropout_mask = (
                torch.rand(batch_size, device=device) >= self.conditioning_dropout
            ).float().view(batch_size, 1, 1)
            
            quality_tokens = quality_tokens * dropout_mask
        
        return quality_tokens
    
    def get_trainable_parameters(self):
        """
        Return iterator over all trainable parameters (adapters, not encoder).
        
        For Stage 1: Only global_adapter parameters
        For Stage 2: Both global_adapter and spatial_adapter parameters
        """
        if self.stage >= 2 and self.spatial_adapter is not None:
            # Chain both adapters' parameters
            from itertools import chain
            return chain(
                self.global_adapter.parameters(),
                self.spatial_adapter.parameters()
            )
        else:
            return self.global_adapter.parameters()
    
    def get_parameter_groups(
        self, 
        global_adapter_lr: float = 1e-4,
        spatial_adapter_lr: float = 1e-4,
    ):
        """
        Get parameter groups for optimizer with separate learning rates.
        
        Args:
            global_adapter_lr: Learning rate for global adapter
            spatial_adapter_lr: Learning rate for spatial adapter (Stage 2 only)
        
        Returns:
            List of parameter group dicts for optimizer
        """
        groups = [
            {
                'params': list(self.global_adapter.parameters()),
                'lr': global_adapter_lr,
                'name': 'arniqa_global_adapter',
            }
        ]
        
        if self.stage >= 2 and self.spatial_adapter is not None:
            groups.append({
                'params': list(self.spatial_adapter.parameters()),
                'lr': spatial_adapter_lr,
                'name': 'arniqa_spatial_adapter',
            })
        
        return groups
    
    def get_adapter_grad_norms(self) -> Dict[str, float]:
        """
        Compute gradient norms for each adapter (for monitoring learning).
        
        Returns:
            Dictionary with gradient norms:
            - 'global_adapter': L2 norm of global adapter gradients
            - 'spatial_adapter': L2 norm of spatial adapter gradients (Stage 2 only)
        """
        norms = {}
        
        # Global adapter gradient norm
        global_grad_norm_sq = 0.0
        for p in self.global_adapter.parameters():
            if p.grad is not None:
                global_grad_norm_sq += p.grad.data.norm(2).item() ** 2
        norms['global_adapter'] = global_grad_norm_sq ** 0.5
        
        # Spatial adapter gradient norm (Stage 2 only)
        if self.stage >= 2 and self.spatial_adapter is not None:
            spatial_grad_norm_sq = 0.0
            for p in self.spatial_adapter.parameters():
                if p.grad is not None:
                    spatial_grad_norm_sq += p.grad.data.norm(2).item() ** 2
            norms['spatial_adapter'] = spatial_grad_norm_sq ** 0.5
        
        return norms
    
    def get_adapter_weight_norms(self) -> Dict[str, float]:
        """
        Compute weight norms for each adapter (for monitoring).
        
        Returns:
            Dictionary with weight norms:
            - 'global_adapter': L2 norm of global adapter weights
            - 'spatial_adapter': L2 norm of spatial adapter weights (Stage 2 only)
        """
        norms = {}
        
        # Global adapter weight norm
        global_weight_norm_sq = 0.0
        for p in self.global_adapter.parameters():
            global_weight_norm_sq += p.data.norm(2).item() ** 2
        norms['global_adapter'] = global_weight_norm_sq ** 0.5
        
        # Spatial adapter weight norm (Stage 2 only)
        if self.stage >= 2 and self.spatial_adapter is not None:
            spatial_weight_norm_sq = 0.0
            for p in self.spatial_adapter.parameters():
                spatial_weight_norm_sq += p.data.norm(2).item() ** 2
            norms['spatial_adapter'] = spatial_weight_norm_sq ** 0.5
        
        return norms
