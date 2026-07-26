"""
Configuration utilities for handling variable interpolation in YAML configs.

This module provides utilities to resolve variable references in OmegaConf configs
when native interpolation is not supported or needs custom handling.
"""

import re
from omegaconf import OmegaConf, DictConfig
from typing import Any, Union
from src.util.config_util import recursive_load_config


def resolve_config_variables(cfg: DictConfig) -> DictConfig:
    """
    Resolve variable interpolation in OmegaConf config.
    
    Handles ${variable} syntax by replacing with actual values from the config.
    This is a fallback for when OmegaConf's native interpolation doesn't work.
    
    Args:
        cfg: OmegaConf DictConfig object
        
    Returns:
        DictConfig with all variables resolved
        
    Example:
        >>> cfg = OmegaConf.create({
        ...     'dataset_name': 'div2k',
        ...     'dir': '${dataset_name}/train'
        ... })
        >>> resolved = resolve_config_variables(cfg)
        >>> print(resolved.dir)  # 'div2k/train'
    """
    # Try OmegaConf's native resolution first
    try:
        OmegaConf.resolve(cfg)
        return cfg
    except Exception:
        # If native resolution fails, use custom resolution
        pass
    
    # Convert to container for manipulation
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    
    # Recursively resolve variables
    resolved_dict = _resolve_dict(cfg_dict, cfg_dict)
    
    # Convert back to DictConfig
    return OmegaConf.create(resolved_dict)


def _resolve_dict(obj: Any, root: dict) -> Any:
    """
    Recursively resolve variables in nested dictionaries.
    
    Args:
        obj: Current object to resolve (dict, list, or primitive)
        root: Root dictionary for variable lookup
        
    Returns:
        Resolved object
    """
    if isinstance(obj, dict):
        return {k: _resolve_dict(v, root) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_dict(item, root) for item in obj]
    elif isinstance(obj, str):
        return _resolve_string(obj, root)
    else:
        return obj


def _resolve_string(s: str, root: dict) -> Union[str, Any]:
    """
    Resolve variable references in a string.
    
    Supports:
    - Simple variables: ${variable}
    - Nested variables: ${parent.child}
    - Full string replacement: "${variable}" -> actual value (preserves type)
    - Partial replacement: "prefix_${variable}_suffix" -> "prefix_value_suffix"
    
    Args:
        s: String potentially containing variable references
        root: Root dictionary for variable lookup
        
    Returns:
        Resolved string or value (if entire string is a variable reference)
    """
    # Pattern to match ${variable} or ${parent.child.grandchild}
    pattern = r'\$\{([^}]+)\}'
    
    # Check if entire string is a single variable reference
    full_match = re.fullmatch(pattern, s)
    if full_match:
        var_path = full_match.group(1)
        value = _get_nested_value(root, var_path)
        return value  # Return actual value (preserves type)
    
    # Otherwise, replace all variable references in the string
    def replace_var(match):
        var_path = match.group(1)
        value = _get_nested_value(root, var_path)
        return str(value)
    
    return re.sub(pattern, replace_var, s)


def _get_nested_value(d: dict, path: str) -> Any:
    """
    Get value from nested dictionary using dot notation.
    
    Args:
        d: Dictionary to search
        path: Dot-separated path (e.g., 'parent.child.grandchild')
        
    Returns:
        Value at the specified path
        
    Raises:
        KeyError: If path doesn't exist
    """
    keys = path.split('.')
    value = d
    
    for key in keys:
        if isinstance(value, dict):
            value = value[key]
        else:
            raise KeyError(f"Cannot access '{key}' in non-dict value at path '{path}'")
    
    return value


def load_config_with_interpolation(config_path: str) -> DictConfig:
    """
    Load YAML config and resolve all variable interpolations.
    
    This function:
    1. Recursively loads base_config files (using Marigold's recursive_load_config)
    2. Merges all configs
    3. Resolves variable interpolations (${variable})
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        DictConfig with all variables resolved
        
    Example:
        >>> cfg = load_config_with_interpolation('config/dataset_train.yaml')
        >>> print(cfg.dataset.train.dir)  # Variables resolved
    """
    # Step 1: Load config with base_config merging (Marigold's standard approach)
    cfg = recursive_load_config(config_path)
    
    # Step 2: Resolve variable interpolations
    return resolve_config_variables(cfg)
