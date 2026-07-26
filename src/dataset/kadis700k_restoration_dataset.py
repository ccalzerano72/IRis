# Kadis700k Restoration Dataset - Thesis Implementation
# 
# Specialization of BaseRestorationDataset for Kadis700k dataset
# --------------------------------------------------------------------------

import os
from .base_restoration_dataset import BaseRestorationDataset, DatasetMode


class Kadis700kRestorationDataset(BaseRestorationDataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        clean_dir: str,
        degraded_dir: str,
        disp_name: str = "kadis700k_restoration",
        **kwargs,
    ) -> None:
        # Kadis700k specific initialization
        super().__init__(
            mode=mode,
            filename_ls_path=filename_ls_path,
            clean_dir=clean_dir,
            degraded_dir=degraded_dir,
            disp_name=disp_name,
            **kwargs,
        )
        
        # Kadis700k specific properties
        self.dataset_name = "kadis700k"
        
        # Verify Kadis700k specific structure
        self._verify_kadis700k_structure()

    def _verify_kadis700k_structure(self):
        """Verify Kadis700k specific directory structure"""
        # Kadis700k has different naming conventions than DIV2K
        # This is handled by the base class, but we can add
        # Kadis700k specific validation here if needed
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