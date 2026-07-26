#!/usr/bin/env python3
"""
Crops Comparison Script for Blind Image Restoration Models.

Runs multiple restoration models across all degradation types found under
a single input directory.  Unlike run_academic_comparison.py (which iterates
over *datasets*, each with its own clean/ and degraded_*/ subdirs), this script
iterates over *degradation subdirectories* that all share one clean/ reference.

Expected input layout
---------------------
<input_dir>/
    clean/                  <- reference images, shared by all degradations
    degraded/
        gaublur_crops_test/ <- degradation type 1
        jpeg_crops_test/    <- degradation type 2
        ...                 <- N more degradation types

Output layout
-------------
<output_dir>/
    <degradation_name>/
        <model_subdir>/restored/
        metrics/
            metrics_<model_name>.csv
            summary_<model_name>.txt
    crops_comparison_state.json   (interrupt/resume state)

Usage
-----
    # Full run (all images, all degradation types)
    python script/restoration/eval/run_crops_comparison.py

    # Test mode (2 images per run, separate state file)
    python script/restoration/eval/run_crops_comparison.py --test

    # Custom input directory
    python script/restoration/eval/run_crops_comparison.py \\
        --input_dir /path/to/comparison_crops

    # Specify GPU
    python script/restoration/eval/run_crops_comparison.py --gpu 1
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# RUN CONFIGURATIONS
#
# Copied verbatim from run_academic_comparison.py.
# The "input_subdir" field (degraded_1x / degraded_4x) is NOT used here:
# each degradation folder is the input directly.  The field is kept so this
# list can be shared with run_academic_comparison.py without modification.
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
        "label": "DiffBIR steps=50 strength=1.0",
        "input_subdir": "degraded_1x",
        "args": {"steps": 50, "strength": 1.0},
    },

    # --- DFPIR ---
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

    # --- Restormer ---
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

    # --- Real-ESRGAN (scale=1 only — crops are not upscaled) ---
    {
        "model": "realesrgan",
        "label": "Real-ESRGAN x4plus scale=1",
        "input_subdir": "degraded_1x",
        "args": {"outscale": 1, "tile": 256},
    },

    # --- HYPIR (upscale=1 only) ---
    {
        "model": "hypir",
        "label": "HYPIR SD2 upscale=1",
        "input_subdir": "degraded_1x",
        "args": {"upscale": 1},
    },
]


# =============================================================================
# Marigold / Hybrid checkpoint configurations
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
        "checkpoint": "checkpoints/009_re_015000/latest",
        "steps": [5, 10, 25],
        "ensembles": [1, 10],
    },
]


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
    "marigold":   "script/restoration/eval/02_infer_unified.sh",
}

# Model name builders — verified from each shell script's MODEL_NAME= definition.
MODEL_NAME_BUILDERS = {
    "diffbir":    lambda a: f"diffbir_steps{a['steps']}_strength{a['strength']}_up1_nocapt",
    "dfpir":      lambda a: f"dfpir_{a['degradation']}",
    "restormer":  lambda a: f"restormer_{a['task'].lower()}",
    "realesrgan": lambda a: f"realesrgan_x4plus_s{a['outscale']}",
    "stablesr":   lambda a: f"stablesr_s{a['ddim_steps']}_w{a['dec_w']}",
    "hypir":      lambda a: f"hypir_sd2_up{a['upscale']}",
    "marigold":   lambda a: (
        f"{a['_ckpt_type']}_{a['_ckpt_parent']}"
        f"_prediction_{a.get('resolution', 0)}"
        f"_s{a['steps']:02d}_e{a['ensemble']:02d}"
        f"_{a.get('scheduler', 'ddim')}"
    ),
}


# =============================================================================
# Checkpoint helpers (copied from run_academic_comparison.py)
# =============================================================================

def detect_checkpoint_type(ckpt_path: str) -> str:
    """
    Detect checkpoint type from its contents.
    Mirrors the detection logic in 02_infer_unified.sh.
    """
    resolved = os.path.realpath(ckpt_path)
    p = Path(resolved)

    if not p.is_dir():
        return "unknown"

    arch_config = p / "architecture_config.json"
    if arch_config.is_file():
        try:
            with open(arch_config) as f:
                cfg = json.load(f)
            if cfg.get("architecture") == "controlnet_trainable_unet_4ch":
                return "controlnet_4ch"
        except (json.JSONDecodeError, KeyError):
            pass

    has_hybrid_003 = (p / "hybrid_003_config.json").is_file()
    has_hybrid_002 = (p / "hybrid_config.json").is_file()
    has_controlnet = (p / "controlnet").is_dir()
    has_unet       = (p / "unet").is_dir()

    if has_hybrid_003 or has_hybrid_002:
        return "hybrid"
    elif has_controlnet and not has_unet:
        return "controlnet"
    elif has_unet:
        return "marigold"
    else:
        return "unknown"


def get_ckpt_parent_dir(ckpt_path: str) -> str:
    """Extract checkpoint parent directory name (without resolving symlinks)."""
    abs_path = os.path.abspath(ckpt_path)
    return os.path.basename(os.path.dirname(abs_path))


def build_marigold_run_configs() -> List[dict]:
    """Build run config entries for all Marigold/Hybrid checkpoints."""
    configs = []
    for ckpt_cfg in MARIGOLD_CHECKPOINTS:
        ckpt_path = ckpt_cfg["checkpoint"]
        ckpt_type = detect_checkpoint_type(ckpt_path)
        ckpt_parent = get_ckpt_parent_dir(ckpt_path)

        if ckpt_type == "unknown":
            logger.warning(f"Cannot detect checkpoint type for {ckpt_path}, skipping")
            continue

        logger.info(f"Checkpoint {ckpt_parent}: detected type = {ckpt_type}")

        for steps in ckpt_cfg["steps"]:
            for ensemble in ckpt_cfg["ensembles"]:
                label = f"[{ckpt_type}] {ckpt_parent} | ddim s{steps} e{ensemble} res0"
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
                        "_ckpt_type": ckpt_type,
                        "_ckpt_parent": ckpt_parent,
                    },
                })
    return configs


# =============================================================================
# Model name helpers
# =============================================================================

def get_model_name(model: str, args: dict) -> str:
    return MODEL_NAME_BUILDERS[model](args)


def get_summary_filename(model: str, args: dict) -> str:
    return f"summary_{get_model_name(model, args)}.txt"


def get_csv_filename(model: str, args: dict) -> str:
    return f"metrics_{get_model_name(model, args)}.csv"


def make_run_key(degradation: str, label: str) -> str:
    return f"{degradation}|{label}"


# =============================================================================
# Command builders — verified from each shell script's positional arg list
# =============================================================================

def build_diffbir_cmd(script_path, subfolder, input_dir, clean_dir,
                      args, gpu, max_images):
    """03_infer_diffbir.sh: $1=subfolder $2=input_dir $3=strength $4=steps
    $5=upscale $6=task $7=clean_dir $8=max_images"""
    return [
        "bash", script_path, subfolder, input_dir,
        str(args["strength"]), str(args["steps"]),
        "1", "denoise", clean_dir,
        str(max_images) if max_images > 0 else "",
    ]


def build_dfpir_cmd(script_path, subfolder, input_dir, clean_dir,
                    args, gpu, max_images):
    """05_infer_dfpir.sh: $1=subfolder $2=input_dir $3=clean_dir
    $4=degradation $5=checkpoint $6=gpu $7=max_images"""
    return [
        "bash", script_path, subfolder, input_dir, clean_dir,
        args["degradation"], "", str(gpu),
        str(max_images) if max_images > 0 else "",
    ]


def build_restormer_cmd(script_path, subfolder, input_dir, clean_dir,
                        args, gpu, max_images):
    """06_infer_restormer.sh: $1=subfolder $2=input_dir $3=clean_dir
    $4=max_images $5=task $6=gpu $7=tile"""
    return [
        "bash", script_path, subfolder, input_dir, clean_dir,
        str(max_images) if max_images > 0 else "",
        args["task"], str(gpu), str(args.get("tile", 256)),
    ]


def build_realesrgan_cmd(script_path, subfolder, input_dir, clean_dir,
                         args, gpu, max_images):
    """07_infer_realesrgan.sh: $1=subfolder $2=input_dir $3=clean_dir
    $4=max_images $5=outscale $6=gpu $7=tile"""
    return [
        "bash", script_path, subfolder, input_dir, clean_dir,
        str(max_images) if max_images > 0 else "",
        str(args["outscale"]), str(gpu), str(args.get("tile", 256)),
    ]


def build_stablesr_cmd(script_path, subfolder, input_dir, clean_dir,
                       args, gpu, max_images):
    """08_infer_stablesr.sh: $1=subfolder $2=input_dir $3=clean_dir
    $4=max_images $5=ddim_steps $6=dec_w $7=upscale $8=colorfix $9=gpu"""
    return [
        "bash", script_path, subfolder, input_dir, clean_dir,
        str(max_images) if max_images > 0 else "",
        str(args["ddim_steps"]), str(args["dec_w"]),
        str(args["upscale"]), args.get("colorfix", "wavelet"), str(gpu),
    ]


def build_hypir_cmd(script_path, subfolder, input_dir, clean_dir,
                    args, gpu, max_images):
    """09_infer_hypir.sh: $1=subfolder $2=input_dir $3=clean_dir
    $4=upscale $5=max_images $6=gpu"""
    return [
        "bash", script_path, subfolder, input_dir, clean_dir,
        str(args["upscale"]),
        str(max_images) if max_images > 0 else "",
        str(gpu),
    ]


def build_marigold_cmd(script_path, subfolder, input_dir, clean_dir,
                       args, gpu, max_images):
    """02_infer_unified.sh: $1=subfolder $2=input_dir $3=clean_dir
    $4=processing_res $5=denoise_steps $6=ensemble_size $7=ckpt
    $8=scheduler $9=guidance_scale $10=use_cpu $11=max_images"""
    return [
        "bash", script_path, subfolder, input_dir, clean_dir,
        str(args.get("resolution", 0)), str(args["steps"]),
        str(args["ensemble"]), args["checkpoint"],
        args.get("scheduler", "ddim"), "1.0", "",
        str(max_images) if max_images > 0 else "",
    ]


CMD_BUILDERS = {
    "diffbir":    build_diffbir_cmd,
    "dfpir":      build_dfpir_cmd,
    "restormer":  build_restormer_cmd,
    "realesrgan": build_realesrgan_cmd,
    "stablesr":   build_stablesr_cmd,
    "hypir":      build_hypir_cmd,
    "marigold":   build_marigold_cmd,
}


# =============================================================================
# Environment helpers
# =============================================================================

def build_clean_env() -> dict:
    """
    Remove venv contamination so 'conda run -n diffbir' sees its own env.
    Copied from run_academic_comparison.py / batch_runner_diffbir.py.
    """
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("_OLD_VIRTUAL_PATH", None)
    venv_bin = os.environ.get("VIRTUAL_ENV", "")
    if venv_bin:
        venv_bin_path = os.path.join(venv_bin, "bin")
        parts = [p for p in env.get("PATH", "").split(os.pathsep)
                 if p != venv_bin_path]
        env["PATH"] = os.pathsep.join(parts)
    return env


# =============================================================================
# Degradation discovery
# =============================================================================

def discover_degradations(degraded_dir: Path) -> List[str]:
    """
    Return sorted list of degradation subdirectory names found under
    <input_dir>/degraded/.  A valid entry is any directory that contains
    at least one image file (png/jpg/jpeg).
    """
    degradations = []
    if not degraded_dir.is_dir():
        logger.error(f"Degraded directory not found: {degraded_dir}")
        return degradations

    for entry in sorted(degraded_dir.iterdir()):
        if not entry.is_dir():
            continue
        images = (
            list(entry.glob("*.png"))
            + list(entry.glob("*.jpg"))
            + list(entry.glob("*.jpeg"))
        )
        if images:
            degradations.append(entry.name)
        else:
            logger.warning(f"Skipping {entry.name}: no images found")

    return degradations


# =============================================================================
# State management
# =============================================================================

class ComparisonState:
    """
    Tracks completed/failed/skipped (degradation, run_config) pairs.
    Persisted to JSON for crash recovery.
    """

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.completed: List[dict] = []
        self.failed: List[dict] = []
        self.skipped: List[dict] = []
        self.batch_start_time: Optional[float] = None
        self._load()

    def _load(self):
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file) as f:
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
        try:
            data = {
                "version": "1.0",
                "last_updated": time.time(),
                "batch_start_time": self.batch_start_time,
                "completed": self.completed,
                "failed": self.failed,
                "skipped": self.skipped,
            }
            tmp = self.state_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def add_completed(self, degradation, label, model_name, duration):
        self.completed.append({
            "run_key": make_run_key(degradation, label),
            "degradation": degradation,
            "label": label,
            "model_name": model_name,
            "duration": round(duration, 1),
            "timestamp": time.time(),
        })
        self.save()

    def add_failed(self, degradation, label, model_name, error, return_code):
        self.failed.append({
            "run_key": make_run_key(degradation, label),
            "degradation": degradation,
            "label": label,
            "model_name": model_name,
            "error": error,
            "return_code": return_code,
            "timestamp": time.time(),
        })
        self.save()

    def add_skipped(self, degradation, label, model_name):
        self.skipped.append({
            "run_key": make_run_key(degradation, label),
            "degradation": degradation,
            "label": label,
            "model_name": model_name,
        })

    def get_completed_keys(self) -> set:
        return {r["run_key"] for r in self.completed}

    def get_skipped_keys(self) -> set:
        return {r["run_key"] for r in self.skipped}


# =============================================================================
# Completion check
# =============================================================================

def is_run_completed(output_dir: Path, degradation: str,
                     model: str, args: dict) -> bool:
    """
    A run is considered complete when both its summary .txt and metrics .csv
    exist in <output_dir>/<degradation>/metrics/.
    """
    metrics_dir = output_dir / degradation / "metrics"
    summary = metrics_dir / get_summary_filename(model, args)
    csv     = metrics_dir / get_csv_filename(model, args)
    return summary.exists() and csv.exists()


# =============================================================================
# Single run execution
# =============================================================================

def execute_run(
    project_root: Path,
    output_dir: Path,
    input_dir: Path,
    degradation: str,
    run_config: dict,
    gpu: int,
    max_images: int,
    test_mode: bool = False,
) -> tuple:
    """
    Execute a single (degradation, run_config) pair.

    Returns (success, duration, error_msg, return_code).
    """
    model     = run_config["model"]
    args      = run_config["args"]
    script_path = str(project_root / SCRIPT_MAP[model])

    # Absolute paths for input and reference
    degraded_dir = str(input_dir / "degraded" / degradation)
    clean_dir    = str(input_dir / "clean")

    # The shell scripts build their output as:
    #   output/<subfolder>/<model_subdir>/restored/
    # We pass a subfolder that resolves to <output_dir>/<degradation> by
    # going through output/../<relative_path>.
    # Since project_root/output is always the anchor, we compute:
    #   output/../<relative path from project_root to output_dir>/<degradation>
    output_rel = output_dir.relative_to(project_root)
    subfolder = f"../{output_rel}/{degradation}"

    cmd = CMD_BUILDERS[model](
        script_path=script_path,
        subfolder=subfolder,
        input_dir=degraded_dir,
        clean_dir=clean_dir,
        args=args,
        gpu=gpu,
        max_images=max_images,
    )

    logger.info(f"  Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
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

        output_lines = []
        for line in iter(process.stdout.readline, ""):
            line_stripped = line.rstrip()
            if line_stripped:
                logger.info(f"    | {line_stripped}")
            output_lines.append(line)
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
# Formatting helpers
# =============================================================================

def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run all baseline models on every degradation type in a crops directory."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: 2 images per run, separate state file",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index (default: 0)",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Path to input directory containing clean/ and degraded/ "
             "(default: <project_root>/comparison_crops/input)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to output directory "
             "(default: <project_root>/comparison_crops/output)",
    )
    args = parser.parse_args()

    # Project root: this script lives at script/restoration/eval/
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Resolve input / output directories
    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else (project_root / "comparison_crops" / "input").resolve()
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (project_root / "comparison_crops" / "output").resolve()
    )

    # Validate clean directory
    clean_dir = input_dir / "clean"
    if not clean_dir.is_dir():
        logger.error(f"Clean directory not found: {clean_dir}")
        sys.exit(1)

    # Validate degraded root
    degraded_root = input_dir / "degraded"
    if not degraded_root.is_dir():
        logger.error(f"Degraded directory not found: {degraded_root}")
        sys.exit(1)

    if args.test:
        state_file  = output_dir / "crops_comparison_test_state.json"
        max_images  = 2
        mode_label  = "TEST MODE"
    else:
        state_file  = output_dir / "crops_comparison_state.json"
        max_images  = 0   # 0 = all images
        mode_label  = "FULL MODE"

    # Discover degradation types
    degradations = discover_degradations(degraded_root)
    if not degradations:
        logger.error("No valid degradation subdirectories found. Exiting.")
        sys.exit(1)

    # Build full run queue
    marigold_configs = build_marigold_run_configs()
    all_configs      = RUN_CONFIGS + marigold_configs

    run_queue = []
    for degradation in degradations:
        for run_config in all_configs:
            run_queue.append((degradation, run_config))

    total_runs = len(run_queue)

    # Load state and identify already-completed runs
    output_dir.mkdir(parents=True, exist_ok=True)
    state = ComparisonState(state_file)

    pending_runs = []
    for degradation, run_config in run_queue:
        model     = run_config["model"]
        run_args  = run_config["args"]
        label     = run_config["label"]
        model_name = get_model_name(model, run_args)

        if is_run_completed(output_dir, degradation, model, run_args):
            run_key = make_run_key(degradation, label)
            if (run_key not in state.get_completed_keys()
                    and run_key not in state.get_skipped_keys()):
                state.add_skipped(degradation, label, model_name)
        else:
            pending_runs.append((degradation, run_config))

    skipped_count = total_runs - len(pending_runs)

    # Print summary header
    logger.info("=" * 70)
    logger.info(f"CROPS COMPARISON — {mode_label}")
    logger.info("=" * 70)
    logger.info(f"Input dir:      {input_dir}")
    logger.info(f"Output dir:     {output_dir}")
    logger.info(f"Clean dir:      {clean_dir}")
    logger.info(f"Degradations:   {', '.join(degradations)}")
    logger.info(
        f"Run configs:    {len(all_configs)} "
        f"({len(RUN_CONFIGS)} baseline + {len(marigold_configs)} marigold/hybrid)"
    )
    logger.info(f"Total runs:     {total_runs}")
    logger.info(f"Already done:   {skipped_count}")
    logger.info(f"Pending:        {len(pending_runs)}")
    logger.info(f"GPU:            {args.gpu}")
    if args.test:
        logger.info(f"Max images:     {max_images} (test mode)")
    logger.info(f"State file:     {state_file}")
    logger.info("=" * 70)

    if not pending_runs:
        logger.info("All runs already completed. Nothing to do.")
        state.save()
        return

    logger.info("Pending runs:")
    for i, (degradation, run_config) in enumerate(pending_runs, 1):
        logger.info(f"  {i:3d}. [{degradation}] {run_config['label']}")
    logger.info("")

    # Graceful SIGINT handler
    interrupted = False

    def sigint_handler(signum, frame):
        nonlocal interrupted
        if interrupted:
            logger.warning("Force exit requested.")
            sys.exit(1)
        interrupted = True
        logger.warning(
            "Interrupt received. Will stop after current run completes. "
            "Press Ctrl+C again to force exit."
        )

    signal.signal(signal.SIGINT, sigint_handler)

    if state.batch_start_time is None:
        state.batch_start_time = time.time()

    completed_count = 0
    failed_count    = 0

    for i, (degradation, run_config) in enumerate(pending_runs, 1):
        if interrupted:
            logger.warning("Stopping due to interrupt.")
            break

        model      = run_config["model"]
        run_args   = run_config["args"]
        label      = run_config["label"]
        model_name = get_model_name(model, run_args)

        # Re-check in case it completed in a previous iteration
        if is_run_completed(output_dir, degradation, model, run_args):
            logger.info(
                f"[{i}/{len(pending_runs)}] [{degradation}] {label} — "
                f"already completed, skipping"
            )
            state.add_skipped(degradation, label, model_name)
            continue

        eta_str = ""
        if completed_count > 0:
            elapsed     = time.time() - state.batch_start_time
            avg_per_run = elapsed / completed_count
            remaining   = len(pending_runs) - i
            eta_str     = f" | ETA: {format_duration(avg_per_run * remaining)}"

        logger.info("")
        logger.info("-" * 70)
        logger.info(
            f"[{i}/{len(pending_runs)}] [{degradation}] {label}{eta_str}"
        )
        logger.info(f"  Model name: {model_name}")
        logger.info("-" * 70)

        success, duration, error_msg, return_code = execute_run(
            project_root=project_root,
            output_dir=output_dir,
            input_dir=input_dir,
            degradation=degradation,
            run_config=run_config,
            gpu=args.gpu,
            max_images=max_images,
            test_mode=args.test,
        )

        if success:
            completed_count += 1
            state.add_completed(degradation, label, model_name, duration)
            logger.info(f"  COMPLETED in {format_duration(duration)}")
        else:
            failed_count += 1
            state.add_failed(degradation, label, model_name, error_msg, return_code)
            logger.error(f"  FAILED ({error_msg}) after {format_duration(duration)}")

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("CROPS COMPARISON COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Completed:  {completed_count}")
    logger.info(f"Failed:     {failed_count}")
    logger.info(f"Skipped:    {skipped_count}")
    if interrupted:
        remaining = max(0, len(pending_runs) - completed_count - failed_count)
        logger.info(f"Remaining:  {remaining} (interrupted)")
    total_elapsed = time.time() - state.batch_start_time
    logger.info(f"Total time: {format_duration(total_elapsed)}")
    logger.info(f"State saved to: {state_file}")
    logger.info("=" * 70)

    state.save()

    # -------------------------------------------------------------------------
    # Generate baseline metrics (clean vs degraded) for each degradation type.
    # 05_calculate_baselines.py expects --results_dir pointing to the folder
    # that contains the metrics/ subdirectory, i.e. output_dir/<degradation>.
    # -------------------------------------------------------------------------
    if not interrupted:
        logger.info("")
        logger.info("=" * 70)
        logger.info("GENERATING BASELINE METRICS")
        logger.info("=" * 70)

        baseline_script = str(
            project_root / "script" / "restoration" / "eval" / "05_calculate_baselines.py"
        )

        baseline_args = []
        for degradation in degradations:
            deg_results = output_dir / degradation
            if deg_results.is_dir():
                baseline_args.extend(["--results_dir", str(deg_results)])

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
                    text=True,
                    timeout=14400,
                )
                if result.returncode == 0:
                    logger.info("  Baseline metrics generated successfully")
                else:
                    logger.warning(
                        f"  Baseline metrics failed (exit code {result.returncode})"
                    )
            except subprocess.TimeoutExpired:
                logger.warning("  Baseline metrics timed out")
            except Exception as e:
                logger.warning(f"  Baseline metrics error: {e}")
        else:
            logger.info("  No output directories found, skipping baselines")

        logger.info("=" * 70)

    if failed_count > 0:
        sys.exit(1)
    if interrupted:
        sys.exit(2)


if __name__ == "__main__":
    main()
