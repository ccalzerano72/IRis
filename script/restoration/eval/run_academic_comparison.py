#!/usr/bin/env python3
"""
Academic Comparison Script for Blind Image Restoration Models.

Runs multiple restoration models (DiffBIR, DFPIR, Restormer, Real-ESRGAN, StableSR)
with various parameter configurations across all datasets in comparison/datasets/.

Each model is invoked via its existing shell script (03_infer_diffbir.sh, etc.),
which handles inference + metrics calculation + per-image resume internally.

This script adds:
  - Multi-dataset iteration (auto-discovers datasets from comparison/datasets/)
  - Cross-model orchestration (runs all models × all datasets sequentially)
  - State persistence (JSON file) for interrupt/resume across crashes
  - Test mode (--test): limits to 2 images per run, saves to comparison/test_results/
  - Smart skip: if you edit the RUN_CONFIGS list, already-completed runs are skipped
    and only new/changed configurations are executed.

Usage:
    # Full run (all images, all datasets)
    python script/restoration/eval/run_academic_comparison.py

    # Test mode (2 images per run, saves to comparison/test_results/)
    python script/restoration/eval/run_academic_comparison.py --test

    # Specify GPU
    python script/restoration/eval/run_academic_comparison.py --gpu 1

Output structure:
    comparison/results/<DATASET>/<model_subdir>/restored/
    comparison/results/<DATASET>/metrics/summary_<model_name>.txt
    comparison/results/<DATASET>/metrics/metrics_<model_name>.csv

State file:
    comparison/comparison_state.json  (or comparison/test_comparison_state.json)
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# RUN CONFIGURATIONS — Edit this list to add/remove/change model runs.
#
# Each entry is a dict with:
#   "model"       : which shell script to call
#   "label"       : human-readable description (also used as unique run key)
#   "input_subdir": "degraded_1x" or "degraded_4x" (subfolder inside each dataset)
#   "args"        : dict of model-specific parameters passed to the shell script
#
# The "model" field maps to the shell script:
#   "diffbir"    -> script/restoration/eval/03_infer_diffbir.sh
#   "dfpir"      -> script/restoration/eval/05_infer_dfpir.sh
#   "restormer"  -> script/restoration/eval/06_infer_restormer.sh
#   "realesrgan" -> script/restoration/eval/07_infer_realesrgan.sh
#   "stablesr"   -> script/restoration/eval/08_infer_stablesr.sh
#   "hypir"      -> script/restoration/eval/09_infer_hypir.sh
#   "nafnet"     -> script/restoration/eval/11_infer_nafnet.sh
# =============================================================================

RUN_CONFIGS = [
    # --- DiffBIR: steps=25, strength=0.5 and 1.0 ---
    {
        "model": "diffbir",
        "label": "DiffBIR steps=25 strength=0.5",
        "input_subdir": "degraded_1x",
        "args": {"steps": 25, "strength": 0.5},
    },
    {
        "model": "diffbir",
        "label": "DiffBIR steps=25 strength=1.0",
        "input_subdir": "degraded_1x",
        "args": {"steps": 25, "strength": 1.0},
    },
    {
        "model": "diffbir",
        "label": "DiffBIR steps=25 strength=1.0",
        "input_subdir": "degraded_1x",
        "args": {"steps": 50, "strength": 1.0},
    },

    # --- DFPIR: noise15, noise25, noise50, blur, general ---
    {
        "model": "dfpir",
        "label": "DFPIR noise15",
        "input_subdir": "degraded_1x",
        "args": {"degradation": "noise15"},
    },
    {
        "model": "dfpir",
        "label": "DFPIR noise25",
        "input_subdir": "degraded_1x",
        "args": {"degradation": "noise25"},
    },
    {
        "model": "dfpir",
        "label": "DFPIR noise50",
        "input_subdir": "degraded_1x",
        "args": {"degradation": "noise50"},
    },
    {
        "model": "dfpir",
        "label": "DFPIR blur",
        "input_subdir": "degraded_1x",
        "args": {"degradation": "blur"},
    },
    {
        "model": "dfpir",
        "label": "DFPIR general",
        "input_subdir": "degraded_1x",
        "args": {"degradation": "general"},
    },

    # --- Restormer: tile=256, real_denoising and gaussian_color_denoising ---
    {
        "model": "restormer",
        "label": "Restormer Real_Denoising",
        "input_subdir": "degraded_1x",
        "args": {"task": "Real_Denoising", "tile": 256},
    },
    {
        "model": "restormer",
        "label": "Restormer Gaussian_Color_Denoising",
        "input_subdir": "degraded_1x",
        "args": {"task": "Gaussian_Color_Denoising", "tile": 256},
    },

    # --- Real-ESRGAN: outscale=4 (degraded_4x) and outscale=1 (degraded_1x), tile=256 ---
    {
        "model": "realesrgan",
        "label": "Real-ESRGAN x4plus scale=4",
        "input_subdir": "degraded_4x",
        "args": {"outscale": 4, "tile": 256},
    },
    {
        "model": "realesrgan",
        "label": "Real-ESRGAN x4plus scale=1",
        "input_subdir": "degraded_1x",
        "args": {"outscale": 1, "tile": 256},
    },

    # --- StableSR: ddim_steps=20, dec_w=0.0 and 0.5, upscale=4, colorfix=wavelet ---
    {
        "model": "stablesr",
        "label": "StableSR s20 w0.0",
        "input_subdir": "degraded_4x",
        "args": {"ddim_steps": 20, "dec_w": 0.0, "upscale": 4, "colorfix": "wavelet"},
    },
    {
        "model": "stablesr",
        "label": "StableSR s20 w0.5",
        "input_subdir": "degraded_4x",
        "args": {"ddim_steps": 20, "dec_w": 0.5, "upscale": 4, "colorfix": "wavelet"},
    },

    # --- HYPIR: single-step GAN (SD2.1 LoRA), upscale=1 and upscale=4 ---
    {
        "model": "hypir",
        "label": "HYPIR SD2 upscale=1",
        "input_subdir": "degraded_1x",
        "args": {"upscale": 1},
    },
    {
        "model": "hypir",
        "label": "HYPIR SD2 upscale=4",
        "input_subdir": "degraded_4x",
        "args": {"upscale": 4},
    },

    # --- NAFNet: SIDD width64 (regression CNN baseline, ECCV 2022) ---
    # {
    #    "model": "nafnet",
    #    "label": "NAFNet SIDD width64",
    #    "input_subdir": "degraded_1x",
    #    "args": {"width": 64},
    # },
]


# =============================================================================
# Marigold/Hybrid checkpoint configurations.
#
# Each entry: (checkpoint_path, steps_list, ensemble_list)
# The checkpoint type (marigold/hybrid/controlnet) is auto-detected at startup.
# Other parameters use defaults: resolution=0, scheduler=ddim, guidance_scale=1.0
# =============================================================================

MARIGOLD_CHECKPOINTS = [
    {
        "checkpoint": "checkpoints/001_re_015000/latest",
        "steps": [5, 10, 25],
        "ensembles": [1, 10],
    },
    {
        "checkpoint": "checkpoints/002_re_015000/latest",
        "steps": [5, 10, 25],
        "ensembles": [1, 10],
    },
    {
        "checkpoint": "checkpoints/004_re_015000/latest",
        "steps": [5, 10, 25],
        "ensembles": [1, 10],
    },
    {
        "checkpoint": "checkpoints/re_ablation_cn_frozen_015000/latest",
        "steps": [5, 10, 25],
        "ensembles": [1, 10],
    },
]


def build_marigold_run_configs() -> List[dict]:
    """
    Build RUN_CONFIGS entries for Marigold/Hybrid checkpoints.

    Auto-detects checkpoint type using detect_checkpoint_type() (copied from
    webserver_test/utils/batch_runner_unified.py) and builds entries that match
    the 02_infer_unified.sh output naming convention.
    """
    configs = []
    for ckpt_cfg in MARIGOLD_CHECKPOINTS:
        ckpt_path = ckpt_cfg["checkpoint"]
        ckpt_type = detect_checkpoint_type(ckpt_path)
        ckpt_parent = get_ckpt_parent_dir(ckpt_path)

        if ckpt_type == "unknown":
            logger.warning(
                f"Cannot detect checkpoint type for {ckpt_path}, skipping"
            )
            continue

        logger.info(
            f"Checkpoint {ckpt_parent}: detected type = {ckpt_type}"
        )

        for steps in ckpt_cfg["steps"]:
            for ensemble in ckpt_cfg["ensembles"]:
                label = (
                    f"[{ckpt_type}] {ckpt_parent} | ddim s{steps} e{ensemble} res0"
                )
                configs.append({
                    "model": "marigold",
                    "label": label,
                    "input_subdir": "degraded_1x",
                    "args": {
                        "checkpoint": ckpt_path,
                        "steps": steps,
                        "ensemble": ensemble,
                        "resolution": 0,
                        "scheduler": "ddim",
                        # Internal fields for model name builder
                        "_ckpt_type": ckpt_type,
                        "_ckpt_parent": ckpt_parent,
                    },
                })
    return configs


# =============================================================================
# Shell script paths (relative to project root)
# =============================================================================

SCRIPT_MAP = {
    "diffbir":    "script/restoration/eval/03_infer_diffbir.sh",
    "dfpir":      "script/restoration/eval/05_infer_dfpir.sh",
    "restormer":  "script/restoration/eval/06_infer_restormer.sh",
    "realesrgan": "script/restoration/eval/07_infer_realesrgan.sh",
    "stablesr":   "script/restoration/eval/08_infer_stablesr.sh",
    "hypir":      "script/restoration/eval/09_infer_hypir.sh",
    # "nafnet":     "script/restoration/eval/11_infer_nafnet.sh",
    "marigold":   "script/restoration/eval/02_infer_unified.sh",
}

# Model name patterns — must match what each shell script produces.
# Verified by reading MODEL_NAME= in each .sh file.
MODEL_NAME_BUILDERS = {
    # 03_infer_diffbir.sh: MODEL_NAME="diffbir_steps${steps}_strength${strength}_up${upscale}_nocapt"
    "diffbir": lambda args: f"diffbir_steps{args['steps']}_strength{args['strength']}_up1_nocapt",
    # 05_infer_dfpir.sh: MODEL_NAME="dfpir_${degradation}"
    "dfpir": lambda args: f"dfpir_{args['degradation']}",
    # 06_infer_restormer.sh: MODEL_NAME="restormer_${TASK_LOWER}"
    "restormer": lambda args: f"restormer_{args['task'].lower()}",
    # 07_infer_realesrgan.sh: MODEL_NAME="realesrgan_x4plus_s${outscale}"
    "realesrgan": lambda args: f"realesrgan_x4plus_s{args['outscale']}",
    # 08_infer_stablesr.sh: MODEL_NAME="stablesr_s${ddim_steps}_w${dec_w}"
    "stablesr": lambda args: f"stablesr_s{args['ddim_steps']}_w{args['dec_w']}",
    # 09_infer_hypir.sh: MODEL_NAME="hypir_sd2_up${upscale}"
    "hypir": lambda args: f"hypir_sd2_up{args['upscale']}",
    # 11_infer_nafnet.sh: MODEL_NAME="nafnet_sidd_width${width}"
    # "nafnet": lambda args: f"nafnet_sidd_width{args['width']}",
    # 02_infer_unified.sh: OUTPUT_SUBDIR="${CKPT_TYPE}_${CKPT_PARENT_DIR}_prediction_${processing_res}_s%02d_e%02d_${scheduler}${cfg_suffix}"
    "marigold": lambda args: (
        f"{args['_ckpt_type']}_{args['_ckpt_parent']}"
        f"_prediction_{args.get('resolution', 0)}"
        f"_s{args['steps']:02d}_e{args['ensemble']:02d}"
        f"_{args.get('scheduler', 'ddim')}"
    ),
}


def detect_checkpoint_type(ckpt_path: str) -> str:
    """
    Detect checkpoint type from its contents.

    Copied from webserver_test/utils/batch_runner_unified.py lines 44-82.
    Mirrors the detection logic in 02_infer_unified.sh:
      - hybrid_003_config.json present  -> "hybrid"
      - hybrid_config.json present      -> "hybrid"
      - controlnet/ dir, NO unet/       -> "controlnet"
      - unet/ dir, no hybrid configs    -> "marigold"
      - otherwise                       -> "unknown"
    """
    resolved = os.path.realpath(ckpt_path)
    p = Path(resolved)

    if not p.is_dir():
        return "unknown"

    # Check architecture_config.json first (most specific detection)
    arch_config = p / "architecture_config.json"
    if arch_config.is_file():
        try:
            import json
            with open(arch_config) as f:
                cfg = json.load(f)
            if cfg.get("architecture") == "controlnet_trainable_unet_4ch":
                return "controlnet_4ch"
        except (json.JSONDecodeError, KeyError):
            pass

    has_hybrid_003 = (p / "hybrid_003_config.json").is_file()
    has_hybrid_002 = (p / "hybrid_config.json").is_file()
    has_controlnet = (p / "controlnet").is_dir()
    has_unet = (p / "unet").is_dir()

    if has_hybrid_003 or has_hybrid_002:
        return "hybrid"
    elif has_controlnet and not has_unet:
        return "controlnet"
    elif has_unet:
        return "marigold"
    else:
        return "unknown"


def get_ckpt_parent_dir(ckpt_path: str) -> str:
    """
    Extract checkpoint parent directory name.

    Uses abspath (not realpath) to avoid resolving symlinks, matching
    02_infer_unified.sh: CKPT_PARENT_DIR=$(basename "$(cd "$(dirname "${ckpt}")" && pwd)")
    and RunConfig.get_ckpt_parent_dir() from batch_runner_unified.py lines 93-105.
    """
    abs_path = os.path.abspath(ckpt_path)
    return os.path.basename(os.path.dirname(abs_path))


def get_model_name(model: str, args: dict) -> str:
    """Build the model name string that the shell script will produce."""
    return MODEL_NAME_BUILDERS[model](args)


def get_summary_filename(model: str, args: dict) -> str:
    """Build the summary filename that the shell script will produce."""
    return f"summary_{get_model_name(model, args)}.txt"


def get_csv_filename(model: str, args: dict) -> str:
    """Build the CSV filename that the shell script will produce."""
    return f"metrics_{get_model_name(model, args)}.csv"


def make_run_key(dataset: str, label: str) -> str:
    """Unique key for a (dataset, run_config) pair."""
    return f"{dataset}|{label}"


# =============================================================================
# Command builders — one per model, matching the exact positional args
# verified from each shell script and the webserver_test batch runners.
# =============================================================================

def build_diffbir_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 03_infer_diffbir.sh.

    Verified from 03_infer_diffbir.sh positional args:
        $1 = subfolder
        $2 = input_dir
        $3 = strength
        $4 = steps
        $5 = upscale (fixed 1 for same-resolution comparison)
        $6 = task (fixed "denoise")
        $7 = clean_dir
        $8 = max_images ("" or number)
    """
    return [
        "bash", script_path,
        subfolder,
        input_dir,
        str(args["strength"]),
        str(args["steps"]),
        "1",        # upscale fixed at 1
        "denoise",  # task fixed at denoise
        clean_dir,
        str(max_images) if max_images > 0 else "",
    ]


def build_dfpir_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 05_infer_dfpir.sh.

    Verified from 05_infer_dfpir.sh positional args:
        $1 = subfolder
        $2 = input_dir
        $3 = clean_dir
        $4 = degradation
        $5 = checkpoint (empty = auto-detect)
        $6 = gpu
        $7 = max_images ("" or number)
    """
    return [
        "bash", script_path,
        subfolder,
        input_dir,
        clean_dir,
        args["degradation"],
        "",          # checkpoint: empty = auto-detect
        str(gpu),
        str(max_images) if max_images > 0 else "",
    ]


def build_restormer_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 06_infer_restormer.sh.

    Verified from 06_infer_restormer.sh positional args:
        $1 = subfolder
        $2 = input_dir
        $3 = clean_dir
        $4 = max_images ("" or number)
        $5 = task (Real_Denoising or Gaussian_Color_Denoising)
        $6 = gpu
        $7 = tile (256)
    """
    return [
        "bash", script_path,
        subfolder,
        input_dir,
        clean_dir,
        str(max_images) if max_images > 0 else "",
        args["task"],
        str(gpu),
        str(args.get("tile", 256)),
    ]


def build_realesrgan_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 07_infer_realesrgan.sh.

    Verified from 07_infer_realesrgan.sh positional args:
        $1 = subfolder
        $2 = input_dir
        $3 = clean_dir
        $4 = max_images ("" or number)
        $5 = outscale (default 1)
        $6 = gpu
        $7 = tile (256)
    """
    return [
        "bash", script_path,
        subfolder,
        input_dir,
        clean_dir,
        str(max_images) if max_images > 0 else "",
        str(args["outscale"]),
        str(gpu),
        str(args.get("tile", 256)),
    ]


def build_stablesr_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 08_infer_stablesr.sh.

    Verified from 08_infer_stablesr.sh positional args:
        $1 = subfolder
        $2 = input_dir
        $3 = clean_dir
        $4 = max_images ("" or number)
        $5 = ddim_steps (default 20)
        $6 = dec_w (default 0.0)
        $7 = upscale (default 4)
        $8 = colorfix (default wavelet)
        $9 = gpu
    """
    return [
        "bash", script_path,
        subfolder,
        input_dir,
        clean_dir,
        str(max_images) if max_images > 0 else "",
        str(args["ddim_steps"]),
        str(args["dec_w"]),
        str(args["upscale"]),
        args.get("colorfix", "wavelet"),
        str(gpu),
    ]


CMD_BUILDERS = {
    "diffbir":    build_diffbir_cmd,
    "dfpir":      build_dfpir_cmd,
    "restormer":  build_restormer_cmd,
    "realesrgan": build_realesrgan_cmd,
    "stablesr":   build_stablesr_cmd,
}


def build_hypir_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 09_infer_hypir.sh.

    Verified from 09_infer_hypir.sh positional args:
        $1 = subfolder
        $2 = input_dir
        $3 = clean_dir
        $4 = upscale (1 or 4)
        $5 = max_images ("" or number)
        $6 = gpu
    """
    return [
        "bash", script_path,
        subfolder,
        input_dir,
        clean_dir,
        str(args["upscale"]),
        str(max_images) if max_images > 0 else "",
        str(gpu),
    ]


# Register hypir command builder
CMD_BUILDERS["hypir"] = build_hypir_cmd


def build_nafnet_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 11_infer_nafnet.sh.

    Verified from 11_infer_nafnet.sh positional args:
        $1 = subfolder
        $2 = input_dir
        $3 = clean_dir
        $4 = max_images ("" or number)
        $5 = width (default 64)
        $6 = gpu
    """
    return [
        "bash", script_path,
        subfolder,
        input_dir,
        clean_dir,
        str(max_images) if max_images > 0 else "",
        str(args["width"]),
        str(gpu),
    ]


# Register nafnet command builder
# CMD_BUILDERS["nafnet"] = build_nafnet_cmd


def build_marigold_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 02_infer_unified.sh.

    Verified from 02_infer_unified.sh positional args:
        $1 = subfolder
        $2 = input_dir
        $3 = clean_dir
        $4 = processing_res (default 0)
        $5 = denoise_steps (default 10)
        $6 = ensemble_size (default 1)
        $7 = ckpt
        $8 = scheduler (default ddim)
        $9 = guidance_scale (fixed 1.0)
        $10 = use_cpu ("" or "cpu")
        $11 = max_images ("" or number)
    """
    return [
        "bash", script_path,
        subfolder,
        input_dir,
        clean_dir,
        str(args.get("resolution", 0)),
        str(args["steps"]),
        str(args["ensemble"]),
        args["checkpoint"],
        args.get("scheduler", "ddim"),
        "1.0",  # CFG fixed at 1.0
        "",     # use_cpu: empty = GPU
        str(max_images) if max_images > 0 else "",
    ]


# Register marigold command builder (after function definition)
CMD_BUILDERS["marigold"] = build_marigold_cmd


# =============================================================================
# Environment helpers
# =============================================================================

def build_clean_env() -> dict:
    """
    Build a clean environment for subprocess, removing venv contamination.

    Copied from webserver_test/utils/batch_runner_diffbir.py lines 400-428.
    When the script runs inside the Marigold .venv, VIRTUAL_ENV, PATH, and
    PYTHONPATH leak into child processes. This causes 'conda run -n diffbir'
    to pick up Marigold packages instead of the diffbir conda env's packages.
    """
    env = os.environ.copy()

    # Remove VIRTUAL_ENV so conda doesn't see an active venv
    env.pop("VIRTUAL_ENV", None)
    # Remove PYTHONHOME if set by venv
    env.pop("PYTHONHOME", None)
    # Remove PYTHONPATH entirely — conda run should set its own
    env.pop("PYTHONPATH", None)
    # Remove _OLD_VIRTUAL_PATH (set by venv activate)
    env.pop("_OLD_VIRTUAL_PATH", None)

    # Remove venv bin from PATH
    venv_bin = os.environ.get("VIRTUAL_ENV", "")
    if venv_bin:
        venv_bin_path = os.path.join(venv_bin, "bin")
        path_parts = env.get("PATH", "").split(os.pathsep)
        path_parts = [p for p in path_parts if p != venv_bin_path]
        env["PATH"] = os.pathsep.join(path_parts)

    return env


# =============================================================================
# Dataset discovery
# =============================================================================

def discover_datasets(datasets_dir: Path) -> List[str]:
    """
    Auto-discover datasets from comparison/datasets/.

    A valid dataset directory must contain at least 'clean/' and one of
    'degraded_1x/' or 'degraded_4x/'.

    Returns sorted list of dataset names.
    """
    datasets = []
    if not datasets_dir.is_dir():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return datasets

    for entry in sorted(datasets_dir.iterdir()):
        if not entry.is_dir():
            continue
        clean = entry / "clean"
        deg_1x = entry / "degraded_1x"
        deg_4x = entry / "degraded_4x"
        if clean.is_dir() and (deg_1x.is_dir() or deg_4x.is_dir()):
            datasets.append(entry.name)
        else:
            logger.warning(
                f"Skipping {entry.name}: missing clean/ or degraded_*/ subdirectory"
            )

    return datasets


# =============================================================================
# State management — persist progress to JSON for crash recovery
# =============================================================================

class ComparisonState:
    """
    Tracks which (dataset, run_config) pairs have been completed, failed, or
    are still pending. Persisted to a JSON file so the script can be interrupted
    and resumed.

    Completion is determined by checking whether the summary + CSV files exist
    in the metrics directory (same logic as each shell script).
    """

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.completed: List[dict] = []
        self.failed: List[dict] = []
        self.skipped: List[dict] = []
        self.batch_start_time: Optional[float] = None
        self._load()

    def _load(self):
        """Load state from disk if it exists."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            self.completed = data.get("completed", [])
            self.failed = data.get("failed", [])
            self.skipped = data.get("skipped", [])
            self.batch_start_time = data.get("batch_start_time")
            logger.info(
                f"Loaded state: {len(self.completed)} completed, "
                f"{len(self.failed)} failed, {len(self.skipped)} skipped"
            )
        except Exception as e:
            logger.warning(f"Failed to load state from {self.state_file}: {e}")

    def save(self):
        """Save state to disk atomically (write to tmp then rename)."""
        try:
            state_data = {
                "version": "1.0",
                "last_updated": time.time(),
                "batch_start_time": self.batch_start_time,
                "completed": self.completed,
                "failed": self.failed,
                "skipped": self.skipped,
            }
            tmp = self.state_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(state_data, f, indent=2)
            tmp.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def add_completed(self, dataset: str, label: str, model_name: str,
                      duration: float):
        self.completed.append({
            "run_key": make_run_key(dataset, label),
            "dataset": dataset,
            "label": label,
            "model_name": model_name,
            "duration": round(duration, 1),
            "timestamp": time.time(),
        })
        self.save()

    def add_failed(self, dataset: str, label: str, model_name: str,
                   error: str, return_code: Optional[int]):
        self.failed.append({
            "run_key": make_run_key(dataset, label),
            "dataset": dataset,
            "label": label,
            "model_name": model_name,
            "error": error,
            "return_code": return_code,
            "timestamp": time.time(),
        })
        self.save()

    def add_skipped(self, dataset: str, label: str, model_name: str):
        self.skipped.append({
            "run_key": make_run_key(dataset, label),
            "dataset": dataset,
            "label": label,
            "model_name": model_name,
        })

    def get_completed_keys(self) -> set:
        return {r["run_key"] for r in self.completed}

    def get_skipped_keys(self) -> set:
        return {r["run_key"] for r in self.skipped}


# =============================================================================
# Run completion check — mirrors the logic in each shell script
# =============================================================================

def is_run_completed(results_base: Path, dataset: str, model: str,
                     args: dict) -> bool:
    """
    Check if a run has already been completed by looking for the summary + CSV
    files in the metrics directory.

    This is the same check each shell script does:
        if [ -f "${SUMMARY_FILE}" ] && [ -f "${CSV_FILE}" ]; then
            echo "Metrics already exist ... Skipping entire run."
    """
    metrics_dir = results_base / dataset / "metrics"
    summary = metrics_dir / get_summary_filename(model, args)
    csv = metrics_dir / get_csv_filename(model, args)
    return summary.exists() and csv.exists()


# =============================================================================
# Single run execution
# =============================================================================

def execute_run(
    project_root: Path,
    results_base: Path,
    datasets_dir: Path,
    dataset: str,
    run_config: dict,
    gpu: int,
    max_images: int,
    test_mode: bool = False,
) -> tuple:
    """
    Execute a single (dataset, run_config) pair.

    Returns (success: bool, duration: float, error_msg: Optional[str],
             return_code: Optional[int])
    """
    model = run_config["model"]
    args = run_config["args"]
    input_subdir = run_config["input_subdir"]

    # Resolve paths
    script_path = str(project_root / SCRIPT_MAP[model])
    input_dir = str(datasets_dir / dataset / input_subdir)
    clean_dir = str(datasets_dir / dataset / "clean")

    # The subfolder is relative to output/ — the shell scripts build paths as:
    #   OUTPUT_DIR="output/${subfolder}/..."
    # We use a relative path so output/${subfolder} resolves to the right place:
    #   output/../comparison/results/<DATASET> = comparison/results/<DATASET>
    #   output/../comparison/test_results/<DATASET> = comparison/test_results/<DATASET>
    results_subdir = "test_results" if test_mode else "results"
    subfolder = f"../comparison/{results_subdir}/{dataset}"

    # Build command
    cmd = CMD_BUILDERS[model](
        script_path=script_path,
        subfolder=subfolder,
        input_dir=input_dir,
        clean_dir=clean_dir,
        args=args,
        gpu=gpu,
        max_images=max_images,
    )

    model_name = get_model_name(model, args)
    logger.info(f"  Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        # Use clean env for DiffBIR (conda), offline HF for marigold, normal for others
        if model == "diffbir":
            env = build_clean_env()
        elif model == "marigold":
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env = None

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            preexec_fn=os.setsid,
            env=env,
            text=True,
            bufsize=1,
        )

        # Stream output to logger
        output_lines = []
        for line in iter(process.stdout.readline, ""):
            line_stripped = line.rstrip()
            if line_stripped:
                logger.info(f"    | {line_stripped}")
            output_lines.append(line)
            # Keep only last 100 lines in memory
            if len(output_lines) > 200:
                output_lines = output_lines[-100:]

        process.stdout.close()
        return_code = process.wait(timeout=60)

        duration = time.time() - start_time

        if return_code == 0:
            return (True, duration, None, 0)
        else:
            error_msg = _detect_error("".join(output_lines), return_code)
            return (False, duration, error_msg, return_code)

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        return (False, duration, "Process timed out", None)

    except Exception as e:
        duration = time.time() - start_time
        return (False, duration, f"Exception: {str(e)}", None)


def _detect_error(output: str, return_code: int) -> str:
    """
    Detect common error types from subprocess output.
    Pattern from webserver_test/utils/batch_runner_diffbir.py lines 524-550.
    """
    oom_patterns = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "torch.cuda.OutOfMemoryError",
    ]
    for pattern in oom_patterns:
        if pattern in output:
            return f"GPU Out of Memory (exit code {return_code})"

    if "RuntimeError" in output:
        import re
        match = re.search(r"RuntimeError: (.+?)(?:\n|$)", output)
        if match:
            return f"RuntimeError: {match.group(1)[:200]}"

    if "FileNotFoundError" in output:
        return f"File not found error (exit code {return_code})"

    if "ERROR:" in output:
        import re
        match = re.search(r"ERROR: (.+?)(?:\n|$)", output)
        if match:
            return match.group(1)[:200]

    return f"Process exited with code {return_code}"


# =============================================================================
# Main orchestration
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format seconds into human-readable HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def main():
    parser = argparse.ArgumentParser(
        description="Academic comparison of blind image restoration models."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: 2 images per run, saves to comparison/test_results/",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index (default: 0)",
    )
    parser.add_argument(
        "--datasets-dir", type=str, default=None,
        help="Path to datasets directory (default: comparison/datasets/)",
    )
    args = parser.parse_args()

    # Resolve project root (this script is at script/restoration/eval/)
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Resolve paths
    if args.datasets_dir:
        datasets_dir = Path(args.datasets_dir).resolve()
    else:
        datasets_dir = (project_root / "comparison" / "datasets").resolve()

    if args.test:
        results_base = (project_root / "comparison" / "test_results").resolve()
        state_file = project_root / "comparison" / "test_comparison_state.json"
        max_images = 2
        mode_label = "TEST MODE"
    else:
        results_base = (project_root / "comparison" / "results").resolve()
        state_file = project_root / "comparison" / "comparison_state.json"
        max_images = 0  # 0 = all images
        mode_label = "FULL MODE"

    # Discover datasets
    datasets = discover_datasets(datasets_dir)
    if not datasets:
        logger.error("No valid datasets found. Exiting.")
        sys.exit(1)

    # Build the full run queue: all (dataset, run_config) pairs
    # Merge baseline model configs with auto-detected Marigold/Hybrid configs
    marigold_configs = build_marigold_run_configs()
    all_configs = RUN_CONFIGS + marigold_configs

    run_queue = []
    for dataset in datasets:
        for run_config in all_configs:
            # Check that the required input_subdir exists for this dataset
            input_subdir_path = datasets_dir / dataset / run_config["input_subdir"]
            if not input_subdir_path.is_dir():
                logger.warning(
                    f"Skipping {run_config['label']} on {dataset}: "
                    f"{run_config['input_subdir']}/ not found"
                )
                continue
            run_queue.append((dataset, run_config))

    total_runs = len(run_queue)

    # Load state
    state = ComparisonState(state_file)

    # Determine which runs to skip (already completed on disk)
    pending_runs = []
    for dataset, run_config in run_queue:
        model = run_config["model"]
        run_args = run_config["args"]
        label = run_config["label"]
        model_name = get_model_name(model, run_args)

        if is_run_completed(results_base, dataset, model, run_args):
            # Already done — record as skipped if not already tracked
            run_key = make_run_key(dataset, label)
            if run_key not in state.get_completed_keys() and \
               run_key not in state.get_skipped_keys():
                state.add_skipped(dataset, label, model_name)
        else:
            pending_runs.append((dataset, run_config))

    skipped_count = total_runs - len(pending_runs)

    # Print summary
    logger.info("=" * 70)
    logger.info(f"ACADEMIC COMPARISON — {mode_label}")
    logger.info("=" * 70)
    logger.info(f"Datasets:       {', '.join(datasets)}")
    logger.info(f"Run configs:    {len(all_configs)} ({len(RUN_CONFIGS)} baseline + {len(marigold_configs)} marigold/hybrid)")
    logger.info(f"Total runs:     {total_runs}")
    logger.info(f"Already done:   {skipped_count}")
    logger.info(f"Pending:        {len(pending_runs)}")
    logger.info(f"GPU:            {args.gpu}")
    if args.test:
        logger.info(f"Max images:     {max_images} (test mode)")
    logger.info(f"Results dir:    {results_base}")
    logger.info(f"State file:     {state_file}")
    logger.info("=" * 70)

    if not pending_runs:
        logger.info("All runs already completed. Nothing to do.")
        state.save()
        return

    # Print pending runs
    logger.info("Pending runs:")
    for i, (dataset, run_config) in enumerate(pending_runs, 1):
        logger.info(f"  {i:3d}. [{dataset}] {run_config['label']}")
    logger.info("")

    # Set up SIGINT handler for graceful shutdown
    interrupted = False

    def sigint_handler(signum, frame):
        nonlocal interrupted
        if interrupted:
            # Second Ctrl+C: force exit
            logger.warning("Force exit requested.")
            sys.exit(1)
        interrupted = True
        logger.warning(
            "Interrupt received. Will stop after current run completes. "
            "Press Ctrl+C again to force exit."
        )

    signal.signal(signal.SIGINT, sigint_handler)

    # Record batch start time
    if state.batch_start_time is None:
        state.batch_start_time = time.time()

    # Execute pending runs
    completed_count = 0
    failed_count = 0

    for i, (dataset, run_config) in enumerate(pending_runs, 1):
        if interrupted:
            logger.warning("Stopping due to interrupt.")
            break

        model = run_config["model"]
        run_args = run_config["args"]
        label = run_config["label"]
        model_name = get_model_name(model, run_args)

        # Check again in case it was completed by a parallel process or
        # a previous iteration in this same run
        if is_run_completed(results_base, dataset, model, run_args):
            logger.info(
                f"[{i}/{len(pending_runs)}] [{dataset}] {label} — "
                f"already completed, skipping"
            )
            state.add_skipped(dataset, label, model_name)
            continue

        # ETA calculation
        eta_str = ""
        if completed_count > 0:
            elapsed = time.time() - state.batch_start_time
            avg_per_run = elapsed / completed_count
            remaining = len(pending_runs) - i
            eta_seconds = avg_per_run * remaining
            eta_str = f" | ETA: {format_duration(eta_seconds)}"

        logger.info("")
        logger.info("-" * 70)
        logger.info(
            f"[{i}/{len(pending_runs)}] [{dataset}] {label}{eta_str}"
        )
        logger.info(f"  Model name: {model_name}")
        logger.info("-" * 70)

        success, duration, error_msg, return_code = execute_run(
            project_root=project_root,
            results_base=results_base,
            datasets_dir=datasets_dir,
            dataset=dataset,
            run_config=run_config,
            gpu=args.gpu,
            max_images=max_images,
            test_mode=args.test,
        )

        if success:
            completed_count += 1
            state.add_completed(dataset, label, model_name, duration)
            logger.info(
                f"  COMPLETED in {format_duration(duration)}"
            )
        else:
            failed_count += 1
            state.add_failed(dataset, label, model_name, error_msg, return_code)
            logger.error(
                f"  FAILED ({error_msg}) after {format_duration(duration)}"
            )

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Completed:  {completed_count}")
    logger.info(f"Failed:     {failed_count}")
    logger.info(f"Skipped:    {skipped_count}")
    if interrupted:
        remaining = len(pending_runs) - completed_count - failed_count - \
            (skipped_count - (total_runs - len(pending_runs)))
        logger.info(f"Remaining:  {max(0, remaining)} (interrupted)")
    total_elapsed = time.time() - state.batch_start_time
    logger.info(f"Total time: {format_duration(total_elapsed)}")
    logger.info(f"State saved to: {state_file}")
    logger.info("=" * 70)

    state.save()

    # -------------------------------------------------------------------------
    # Generate baseline metrics (clean + degraded) for each dataset.
    # Uses 05_calculate_baselines.py which checks for existing files and skips
    # if already computed, so this is safe to run every time.
    # -------------------------------------------------------------------------
    if not interrupted:
        logger.info("")
        logger.info("=" * 70)
        logger.info("GENERATING BASELINE METRICS")
        logger.info("=" * 70)

        baseline_script = str(
            project_root / "script" / "restoration" / "eval" / "05_calculate_baselines.py"
        )

        # Collect all dataset result directories
        baseline_args = []
        for dataset in datasets:
            dataset_results = results_base / dataset
            if dataset_results.is_dir():
                baseline_args.extend(["--results_dir", str(dataset_results)])

        if baseline_args:
            baseline_cmd = [
                sys.executable, baseline_script,
                "--device", f"cuda:{args.gpu}",
            ] + baseline_args

            logger.info(f"  Command: {' '.join(baseline_cmd)}")

            try:
                result = subprocess.run(
                    baseline_cmd,
                    cwd=str(project_root),
                    capture_output=False,
                    text=True,
                    timeout=14400,  # 4 hours max for all baselines
                )
                if result.returncode == 0:
                    logger.info("  Baseline metrics generated successfully")
                else:
                    logger.warning(
                        f"  Baseline metrics generation failed (exit code {result.returncode})"
                    )
            except subprocess.TimeoutExpired:
                logger.warning("  Baseline metrics generation timed out")
            except Exception as e:
                logger.warning(f"  Baseline metrics generation error: {e}")
        else:
            logger.info("  No dataset result directories found, skipping baselines")

        logger.info("=" * 70)

    # Exit with error code if any runs failed
    if failed_count > 0:
        sys.exit(1)
    if interrupted:
        sys.exit(2)


if __name__ == "__main__":
    main()
