# DIV2K Restoration Dataset - Thesis Implementation
# 
# Specialization of BaseRestorationDataset for DIV2K dataset
# --------------------------------------------------------------------------

import os
from .base_restoration_dataset import BaseRestorationDataset, DatasetMode


class DIV2KRestorationDataset(BaseRestorationDataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        clean_dir: str,
        degraded_dir: str,
        disp_name: str = "div2k_restoration",
        **kwargs,
    ) -> None:
        # DIV2K specific initialization
        super().__init__(
            mode=mode,
            filename_ls_path=filename_ls_path,
            clean_dir=clean_dir,
            degraded_dir=degraded_dir,
            disp_name=disp_name,
            **kwargs,
        )
        
        # DIV2K specific properties
        self.dataset_name = "div2k"
        
        # Verify DIV2K specific structure
        self._verify_div2k_structure()

    def _verify_div2k_structure(self):
        """Verify DIV2K specific directory structure"""
        # DIV2K images are typically named 0001.png, 0002.png, etc.
        # This is already handled by the base class, but we can add
        # DIV2K specific validation here if needed
        pass

    @classmethod
    def from_config(cls, cfg, mode: DatasetMode):
        """Create dataset from configuration"""
        return cls(
            mode=mode,
            filename_ls_path=cfg.filenames,
            clean_dir=cfg.dir,
            degraded_dir=cfg.degradation_config.degraded_dir,
            disp_name=cfg.disp_name,
            degradation_config=cfg.degradation_config,
            augmentation_args=getattr(cfg, 'augmentation_args', None),
            resize_to_hw=getattr(cfg, 'resize_to_hw', None),
        )