# LAION Restoration Dataset - Thesis Implementation
# 
# Specialization of BaseRestorationDataset for LAION dataset
# --------------------------------------------------------------------------

import os
from .base_restoration_dataset import BaseRestorationDataset, DatasetMode


class LAIONRestorationDataset(BaseRestorationDataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        clean_dir: str,
        degraded_dir: str,
        disp_name: str = "laion_restoration",
        **kwargs,
    ) -> None:
        # LAION specific initialization
        super().__init__(
            mode=mode,
            filename_ls_path=filename_ls_path,
            clean_dir=clean_dir,
            degraded_dir=degraded_dir,
            disp_name=disp_name,
            **kwargs,
        )
        
        # LAION specific properties
        self.dataset_name = "laion"

    @classmethod
    def from_config(cls, cfg, mode: DatasetMode):
        """Create dataset from configuration"""
        return cls(
            mode=mode,
            filename_ls_path=cfg.filenames,
            clean_dir=cfg.dir,
            degraded_dir=cfg.degradation_config.get('degraded_dir', ''),
            disp_name=cfg.disp_name,
            degradation_config=cfg.degradation_config,
            augmentation_args=getattr(cfg, 'augmentation_args', None),
            resize_to_hw=getattr(cfg, 'resize_to_hw', None),
        )
