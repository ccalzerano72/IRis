# Restoration Dataset Factory - Thesis Implementation
# 
# Factory for creating restoration datasets based on configuration
# Handles auto-detection and generation of missing data
# --------------------------------------------------------------------------

import os
import sys
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

from .base_restoration_dataset import DatasetMode
from .div2k_restoration_dataset import DIV2KRestorationDataset
from .kadis700k_restoration_dataset import Kadis700kRestorationDataset
from .laion_restoration_dataset import LAIONRestorationDataset


class RestorationDatasetFactory:
    """Factory for creating restoration datasets with auto-generation support"""
    
    # Registry of available dataset classes
    DATASET_REGISTRY = {
        'div2k': DIV2KRestorationDataset,
        'kadis700k': Kadis700kRestorationDataset,
        'laion': LAIONRestorationDataset,
    }
    
    @classmethod
    def create_dataset(cls, cfg, mode: DatasetMode, auto_generate: bool = True, base_data_dir: str = None):
        """
        Create restoration dataset from configuration.
        
        Args:
            cfg: Dataset configuration
            mode: Dataset mode (TRAIN, EVAL, RGB_ONLY)
            auto_generate: If True, generate missing data automatically
            
        Returns:
            Dataset instance
        """
        # Extract dataset name from config
        dataset_name = cls._extract_dataset_name(cfg)
        
        # Resolve paths if base_data_dir is provided
        if base_data_dir is not None:
            import copy
            cfg_resolved = copy.deepcopy(cfg)
            
            # Resolve clean dir
            if not os.path.isabs(cfg_resolved.dir):
                cfg_resolved.dir = os.path.join(base_data_dir, cfg_resolved.dir)
            
            # Resolve degraded dir (only if it exists - not needed for online/pipeline/sr modes)
            if hasattr(cfg_resolved.degradation_config, 'degraded_dir') and cfg_resolved.degradation_config.degraded_dir:
                if not os.path.isabs(cfg_resolved.degradation_config.degraded_dir):
                    cfg_resolved.degradation_config.degraded_dir = os.path.join(base_data_dir, cfg_resolved.degradation_config.degraded_dir)
            
            cfg = cfg_resolved
        
        # Check if data exists, generate if needed
        if auto_generate:
            cls._ensure_data_exists(cfg, dataset_name)
        
        # Get dataset class
        dataset_class = cls.DATASET_REGISTRY.get(dataset_name)
        if dataset_class is None:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASET_REGISTRY.keys())}")
        
        # Create dataset instance
        return dataset_class.from_config(cfg, mode)
    
    @classmethod
    def _extract_dataset_name(cls, cfg):
        """Extract dataset name from configuration"""
        if hasattr(cfg, 'name'):
            # Extract from name like "restoration_div2k"
            name = cfg.name
            if name.startswith('restoration_'):
                return name.replace('restoration_', '')
            return name
        
        # Fallback: try to infer from paths
        if hasattr(cfg, 'dir'):
            dir_path = Path(cfg.dir)
            # Look for dataset name in path
            for part in dir_path.parts:
                if part in cls.DATASET_REGISTRY:
                    return part
        
        raise ValueError(f"Cannot determine dataset name from config: {cfg}")
    
    @classmethod
    def _ensure_data_exists(cls, cfg, dataset_name):
        """
        Ensure that clean images, degraded images, and file lists exist.
        Generate them if missing.
        """
        # Check degradation mode
        degradation_mode = cfg.degradation_config.get('mode', 'pre_generated')
        
        # Check what exists
        clean_dir = cfg.dir
        filenames_path = cfg.filenames
        
        clean_exists = os.path.exists(clean_dir) and len(os.listdir(clean_dir)) > 0
        filenames_exist = os.path.exists(filenames_path)
        
        print(f"Data status for {dataset_name} (mode={degradation_mode}):")
        print(f"  Clean dir ({clean_dir}): {'✓' if clean_exists else '✗'}")
        print(f"  File list ({filenames_path}): {'✓' if filenames_exist else '✗'}")
        
        # Check degraded dir only if mode=pre_generated
        if degradation_mode == 'pre_generated':
            degraded_dir = cfg.degradation_config.degraded_dir
            degraded_exists = os.path.exists(degraded_dir) and len(os.listdir(degraded_dir)) > 0
            print(f"  Degraded dir ({degraded_dir}): {'✓' if degraded_exists else '✗'}")
            
            # If everything exists, we're good
            if clean_exists and degraded_exists and filenames_exist:
                print("  → All data exists, proceeding with training")
                return
            
            # Something is missing
            print("  → Missing data detected")
            raise FileNotFoundError(
                f"Missing data for {dataset_name} restoration training (mode=pre_generated).\n"
                f"Please run the dataset preparation script first:\n"
                f"  python script/restoration/dataset_preprocess/prepare_dataset.py --source_dir /path/to/{dataset_name}\n"
                f"\nMissing components:\n"
                f"  - Clean images: {clean_dir} ({'exists' if clean_exists else 'missing'})\n"
                f"  - Degraded images: {degraded_dir} ({'exists' if degraded_exists else 'missing'})\n"
                f"  - File list: {filenames_path} ({'exists' if filenames_exist else 'missing'})"
            )
        
        elif degradation_mode == 'online':
            print(f"  Degraded images: will be generated on-the-fly during training")
            
            # For online mode, only need clean images and file list
            if clean_exists and filenames_exist:
                print("  → All required data exists, proceeding with training")
                return
            
            # Something is missing
            print("  → Missing data detected")
            raise FileNotFoundError(
                f"Missing data for {dataset_name} restoration training (mode=online).\n"
                f"Please run the dataset preparation script first:\n"
                f"  python script/restoration/dataset_preprocess/prepare_dataset.py --source_dir /path/to/{dataset_name}\n"
                f"\nMissing components:\n"
                f"  - Clean images: {clean_dir} ({'exists' if clean_exists else 'missing'})\n"
                f"  - File list: {filenames_path} ({'exists' if filenames_exist else 'missing'})\n"
                f"\nNote: Degraded images are NOT needed in online mode - they will be generated during training."
            )
        
        elif degradation_mode == 'pipeline':
            print(f"  Degraded images: will be generated using realistic pipeline during training")
            
            # For pipeline mode, only need clean images and file list
            if clean_exists and filenames_exist:
                print("  → All required data exists, proceeding with training")
                return
            
            # Something is missing
            print("  → Missing data detected")
            raise FileNotFoundError(
                f"Missing data for {dataset_name} restoration training (mode=pipeline).\n"
                f"Please run the dataset preparation script first:\n"
                f"  python script/restoration/dataset_preprocess/prepare_dataset.py --source_dir /path/to/{dataset_name}\n"
                f"\nMissing components:\n"
                f"  - Clean images: {clean_dir} ({'exists' if clean_exists else 'missing'})\n"
                f"  - File list: {filenames_path} ({'exists' if filenames_exist else 'missing'})\n"
                f"\nNote: Degraded images are NOT needed in pipeline mode - they will be generated using the realistic pipeline during training."
            )
        
        elif degradation_mode == 'sr':
            print(f"  Degraded images: will be generated using SR degradation (downscale-upscale) during training")
            
            # For SR mode, only need clean images and file list
            if clean_exists and filenames_exist:
                print("  → All required data exists, proceeding with training")
                return
            
            # Something is missing
            print("  → Missing data detected")
            raise FileNotFoundError(
                f"Missing data for {dataset_name} restoration training (mode=sr).\n"
                f"Please run the dataset preparation script first:\n"
                f"  python script/restoration/dataset_preprocess/prepare_dataset.py --source_dir /path/to/{dataset_name}\n"
                f"\nMissing components:\n"
                f"  - Clean images: {clean_dir} ({'exists' if clean_exists else 'missing'})\n"
                f"  - File list: {filenames_path} ({'exists' if filenames_exist else 'missing'})\n"
                f"\nNote: Degraded images are NOT needed in SR mode - they will be generated using downscale-upscale during training."
            )
        
        elif degradation_mode in ('diffbir_codeformer', 'diffbir_realesrgan'):
            print(f"  Degraded images: will be generated using DiffBIR {degradation_mode} pipeline during training")
            
            # For DiffBIR modes, only need clean images and file list
            if clean_exists and filenames_exist:
                print("  → All required data exists, proceeding with training")
                return
            
            # Something is missing
            print("  → Missing data detected")
            raise FileNotFoundError(
                f"Missing data for {dataset_name} restoration training (mode={degradation_mode}).\n"
                f"Please run the dataset preparation script first:\n"
                f"  python script/restoration/dataset_preprocess/prepare_dataset.py --source_dir /path/to/{dataset_name}\n"
                f"\nMissing components:\n"
                f"  - Clean images: {clean_dir} ({'exists' if clean_exists else 'missing'})\n"
                f"  - File list: {filenames_path} ({'exists' if filenames_exist else 'missing'})\n"
                f"\nNote: Degraded images are NOT needed in {degradation_mode} mode - they will be generated using DiffBIR's degradation pipeline during training."
            )
        
        else:
            raise ValueError(f"Unknown degradation mode: {degradation_mode}. Use 'pre_generated', 'online', 'pipeline', 'sr', 'diffbir_codeformer', or 'diffbir_realesrgan'")
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class):
        """Register a new dataset class"""
        cls.DATASET_REGISTRY[name] = dataset_class
    
    @classmethod
    def list_available_datasets(cls):
        """List all available dataset types"""
        return list(cls.DATASET_REGISTRY.keys())


def create_restoration_dataset(cfg, mode: DatasetMode, auto_generate: bool = True):
    """
    Convenience function to create restoration dataset.
    
    Args:
        cfg: Dataset configuration
        mode: Dataset mode
        auto_generate: Whether to auto-generate missing data
        
    Returns:
        Dataset instance
    """
    return RestorationDatasetFactory.create_dataset(cfg, mode, auto_generate)