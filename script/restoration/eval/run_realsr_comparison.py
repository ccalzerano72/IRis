#!/usr/bin/env python3
"""
RealSR Comparison Script — Hybrid + baselines on RealSR datasets.

Trimmed version of run_academic_comparison.py that uses comparison_realsr/
as base directory and runs:
  - Restormer Real_Denoising (degraded_1x)
  - Real-ESRGAN x4plus scale=1 (degraded_1x)
  - DiffBIR steps=25 strength=1.0 (degraded_1x)
  - HYPIR SD2 upscale=1 (degraded_1x)
  - DFPIR noise15/noise25/noise50/blur/general (degraded_1x)
  - StableSR s20 w0.0 (degraded_4x, skipped if not available)
  - Hybrid checkpoints with various steps/ensemble configs (degraded_1x)

Usage:
    # Full run
    python script/restoration/eval/run_realsr_comparison.py

    # Test mode (2 images per run)
    python script/restoration/eval/run_realsr_comparison.py --test

    # Specify GPU
    python script/restoration/eval/run_realsr_comparison.py --gpu 1

Output structure:
    comparison_realsr/results/<DATASET>/<model_subdir>/restored/
    comparison_realsr/results/<DATASET>/metrics/summary_<model_name>.txt
    comparison_realsr/results/<DATASET>/metrics/metrics_<model_name>.csv

State file:
    comparison_realsr/comparison_state.json  (or test_comparison_state.json)
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
# BASE DIRECTORY — uses comparison_realsr/ instead of comparison/
# =============================================================================
COMPARISON_DIR_NAME = "comparison_realsr"


# =============================================================================
# RUN CONFIGURATIONS — Only Restormer Real_Denoising
# =============================================================================

RUN_CONFIGS = [
    {
        "model": "restormer",
        "label": "Restormer Real_Denoising",
        "input_subdir": "degraded_1x",
        "args": {"task": "Real_Denoising", "tile": 256},
    },
    {
        "model": "realesrgan",
        "label": "Real-ESRGAN x4plus scale=1",
        "input_subdir": "degraded_1x",
        "args": {"outscale": 1, "tile": 256},
    },
    {
        "model": "diffbir",
        "label": "DiffBIR steps=25 strength=1.0",
        "input_subdir": "degraded_1x",
        "args": {"steps": 25, "strength": 1.0},
    },
    {
        "model": "hypir",
        "label": "HYPIR SD2 upscale=1",
        "input_subdir": "degraded_1x",
        "args": {"upscale": 1},
    },
    # --- DFPIR: all degradation variants ---
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
    # --- StableSR: s20 w0.0, upscale=4 (requires degraded_4x) ---
    {
        "model": "stablesr",
        "label": "StableSR s20 w0.0",
        "input_subdir": "degraded_4x",
        "args": {"ddim_steps": 20, "dec_w": 0.0, "upscale": 4, "colorfix": "wavelet"},
    },
]


# =============================================================================
# Marigold/Hybrid checkpoint configurations — same as original
# =============================================================================

MARIGOLD_CHECKPOINTS = [
    {
        "checkpoint": "checkpoints/002_re_015000/latest",
        "steps": [5, 10, 25],
        "ensembles": [1],
    },
]


# =============================================================================
# Shell script paths (relative to project root)
# =============================================================================

SCRIPT_MAP = {
    "restormer":  "script/restoration/eval/06_infer_restormer.sh",
    "realesrgan": "script/restoration/eval/07_infer_realesrgan.sh",
    "diffbir":    "script/restoration/eval/03_infer_diffbir.sh",
    "hypir":      "script/restoration/eval/09_infer_hypir.sh",
    "dfpir":      "script/restoration/eval/05_infer_dfpir.sh",
    "stablesr":   "script/restoration/eval/08_infer_stablesr.sh",
    "marigold":   "script/restoration/eval/02_infer_unified.sh",
}

MODEL_NAME_BUILDERS = {
    "restormer": lambda args: f"restormer_{args['task'].lower()}",
    "realesrgan": lambda args: f"realesrgan_x4plus_s{args['outscale']}",
    # 03_infer_diffbir.sh: MODEL_NAME="diffbir_steps${steps}_strength${strength}_up${upscale}_nocapt"
    "diffbir": lambda args: f"diffbir_steps{args['steps']}_strength{args['strength']}_up1_nocapt",
    # 09_infer_hypir.sh: MODEL_NAME="hypir_sd2_up${upscale}"
    "hypir": lambda args: f"hypir_sd2_up{args['upscale']}",
    # 05_infer_dfpir.sh: MODEL_NAME="dfpir_${degradation}"
    "dfpir": lambda args: f"dfpir_{args['degradation']}",
    # 08_infer_stablesr.sh: MODEL_NAME="stablesr_s${ddim_steps}_w${dec_w}"
    "stablesr": lambda args: f"stablesr_s{args['ddim_steps']}_w{args['dec_w']}",
    "marigold": lambda args: (
        f"{args['_ckpt_type']}_{args['_ckpt_parent']}"
        f"_prediction_{args.get('resolution', 0)}"
        f"_s{args['steps']:02d}_e{args['ensemble']:02d}"
        f"_{args.get('scheduler', 'ddim')}"
    ),
}


# =============================================================================
# Checkpoint detection — copied from run_academic_comparison.py
# =============================================================================

def detect_checkpoint_type(ckpt_path: str) -> str:
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
    abs_path = os.path.abspath(ckpt_path)
    return os.path.basename(os.path.dirname(abs_path))


def get_model_name(model: str, args: dict) -> str:
    return MODEL_NAME_BUILDERS[model](args)


def get_summary_filename(model: str, args: dict) -> str:
    return f"summary_{get_model_name(model, args)}.txt"


def get_csv_filename(model: str, args: dict) -> str:
    return f"metrics_{get_model_name(model, args)}.csv"


def make_run_key(dataset: str, label: str) -> str:
    return f"{dataset}|{label}"


# =============================================================================
# Build Marigold/Hybrid run configs
# =============================================================================

def build_marigold_run_configs() -> List[dict]:
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
# Command builders
# =============================================================================

def build_restormer_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
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


def build_diffbir_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 03_infer_diffbir.sh.

    Verified from run_academic_comparison.py positional args:
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

    Verified from run_academic_comparison.py positional args:
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


def build_hypir_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 09_infer_hypir.sh.

    Verified from run_academic_comparison.py positional args:
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


def build_stablesr_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
    """
    Build command for 08_infer_stablesr.sh.

    Verified from run_academic_comparison.py positional args:
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


def build_marigold_cmd(
    script_path: str, subfolder: str, input_dir: str, clean_dir: str,
    args: dict, gpu: int, max_images: int,
) -> List[str]:
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
        "1.0",
        "",
        str(max_images) if max_images > 0 else "",
    ]


CMD_BUILDERS = {
    "restormer":  build_restormer_cmd,
    "realesrgan": build_realesrgan_cmd,
    "diffbir":    build_diffbir_cmd,
    "dfpir":      build_dfpir_cmd,
    "hypir":      build_hypir_cmd,
    "stablesr":   build_stablesr_cmd,
    "marigold":   build_marigold_cmd,
}


# =============================================================================
# Dataset discovery
# =============================================================================

def discover_datasets(datasets_dir: Path) -> List[str]:
    datasets = []
    if not datasets_dir.is_dir():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return datasets

    for entry in sorted(datasets_dir.iterdir()):
        if not entry.is_dir():
            continue
        clean = entry / "clean"
        deg_1x = entry / "degraded_1x"
        if clean.is_dir() and deg_1x.is_dir():
            datasets.append(entry.name)
        else:
            logger.warning(f"Skipping {entry.name}: missing clean/ or degraded_1x/")

    return datasets


# =============================================================================
# State management
# =============================================================================

class ComparisonState:
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
            "dataset": dataset, "label": label,
            "model_name": model_name,
            "duration": round(duration, 1),
            "timestamp": time.time(),
        })
        self.save()

    def add_failed(self, dataset: str, label: str, model_name: str,
                   error: str, return_code: Optional[int]):
        self.failed.append({
            "run_key": make_run_key(dataset, label),
            "dataset": dataset, "label": label,
            "model_name": model_name,
            "error": error, "return_code": return_code,
            "timestamp": time.time(),
        })
        self.save()

    def add_skipped(self, dataset: str, label: str, model_name: str):
        self.skipped.append({
            "run_key": make_run_key(dataset, label),
            "dataset": dataset, "label": label,
            "model_name": model_name,
        })

    def get_completed_keys(self) -> set:
        return {r["run_key"] for r in self.completed}

    def get_skipped_keys(self) -> set:
        return {r["run_key"] for r in self.skipped}


# =============================================================================
# Run completion check
# =============================================================================

def is_run_completed(results_base: Path, dataset: str, model: str,
                     args: dict) -> bool:
    metrics_dir = results_base / dataset / "metrics"
    summary = metrics_dir / get_summary_filename(model, args)
    csv = metrics_dir / get_csv_filename(model, args)
    return summary.exists() and csv.exists()


# =============================================================================
# Single run execution
# =============================================================================

def execute_run(
    project_root: Path, results_base: Path, datasets_dir: Path,
    dataset: str, run_config: dict, gpu: int, max_images: int,
    test_mode: bool = False,
) -> tuple:
    model = run_config["model"]
    args = run_config["args"]
    input_subdir = run_config["input_subdir"]

    script_path = str(project_root / SCRIPT_MAP[model])
    input_dir = str(datasets_dir / dataset / input_subdir)
    clean_dir = str(datasets_dir / dataset / "clean")

    results_subdir = "test_results" if test_mode else "results"
    subfolder = f"../{COMPARISON_DIR_NAME}/{results_subdir}/{dataset}"

    cmd = CMD_BUILDERS[model](
        script_path=script_path, subfolder=subfolder,
        input_dir=input_dir, clean_dir=clean_dir,
        args=args, gpu=gpu, max_images=max_images,
    )

    logger.info(f"  Command: {' '.join(cmd)}")
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(project_root), preexec_fn=os.setsid,
            text=True, bufsize=1,
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
    oom_patterns = ["CUDA out of memory", "OutOfMemoryError", "torch.cuda.OutOfMemoryError"]
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
    return f"Process exited with code {return_code}"


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
        description="RealSR comparison: Hybrid + Restormer Real_Denoising."
    )
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 2 images per run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--datasets-dir", type=str, default=None,
                        help="Override datasets directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    comparison_base = project_root / COMPARISON_DIR_NAME

    if args.datasets_dir:
        datasets_dir = Path(args.datasets_dir).resolve()
    else:
        datasets_dir = (comparison_base / "datasets").resolve()

    if args.test:
        results_base = (comparison_base / "test_results").resolve()
        state_file = comparison_base / "test_comparison_state.json"
        max_images = 2
        mode_label = "TEST MODE"
    else:
        results_base = (comparison_base / "results").resolve()
        state_file = comparison_base / "comparison_state.json"
        max_images = 0
        mode_label = "FULL MODE"

    # Discover datasets
    datasets = discover_datasets(datasets_dir)
    if not datasets:
        logger.error("No valid datasets found. Exiting.")
        sys.exit(1)

    # Build run queue
    marigold_configs = build_marigold_run_configs()
    all_configs = RUN_CONFIGS + marigold_configs

    run_queue = []
    for dataset in datasets:
        for run_config in all_configs:
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

    # Determine pending runs
    pending_runs = []
    for dataset, run_config in run_queue:
        model = run_config["model"]
        run_args = run_config["args"]
        label = run_config["label"]
        model_name = get_model_name(model, run_args)

        if is_run_completed(results_base, dataset, model, run_args):
            run_key = make_run_key(dataset, label)
            if run_key not in state.get_completed_keys() and \
               run_key not in state.get_skipped_keys():
                state.add_skipped(dataset, label, model_name)
        else:
            pending_runs.append((dataset, run_config))

    skipped_count = total_runs - len(pending_runs)

    # Print summary
    logger.info("=" * 70)
    logger.info(f"REALSR COMPARISON — {mode_label}")
    logger.info("=" * 70)
    logger.info(f"Datasets:       {', '.join(datasets)}")
    logger.info(f"Run configs:    {len(all_configs)} ({len(RUN_CONFIGS)} baseline + {len(marigold_configs)} hybrid)")
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

    # SIGINT handler
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

    # Execute
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

        if is_run_completed(results_base, dataset, model, run_args):
            logger.info(f"[{i}/{len(pending_runs)}] [{dataset}] {label} — already completed, skipping")
            state.add_skipped(dataset, label, model_name)
            continue

        eta_str = ""
        if completed_count > 0:
            elapsed = time.time() - state.batch_start_time
            avg_per_run = elapsed / completed_count
            remaining = len(pending_runs) - i
            eta_str = f" | ETA: {format_duration(avg_per_run * remaining)}"

        logger.info("")
        logger.info("-" * 70)
        logger.info(f"[{i}/{len(pending_runs)}] [{dataset}] {label}{eta_str}")
        logger.info(f"  Model name: {model_name}")
        logger.info("-" * 70)

        success, duration, error_msg, return_code = execute_run(
            project_root=project_root, results_base=results_base,
            datasets_dir=datasets_dir, dataset=dataset,
            run_config=run_config, gpu=args.gpu,
            max_images=max_images, test_mode=args.test,
        )

        if success:
            completed_count += 1
            state.add_completed(dataset, label, model_name, duration)
            logger.info(f"  COMPLETED in {format_duration(duration)}")
        else:
            failed_count += 1
            state.add_failed(dataset, label, model_name, error_msg, return_code)
            logger.error(f"  FAILED ({error_msg}) after {format_duration(duration)}")

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("REALSR COMPARISON COMPLETE")
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

    # Generate baseline metrics
    if not interrupted:
        logger.info("")
        logger.info("=" * 70)
        logger.info("GENERATING BASELINE METRICS")
        logger.info("=" * 70)

        baseline_script = str(
            project_root / "script" / "restoration" / "eval" / "05_calculate_baselines.py"
        )

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
                    baseline_cmd, cwd=str(project_root),
                    capture_output=False, text=True, timeout=3600,
                )
                if result.returncode == 0:
                    logger.info("  Baseline metrics generated successfully")
                else:
                    logger.warning(f"  Baseline metrics generation failed (exit code {result.returncode})")
            except subprocess.TimeoutExpired:
                logger.warning("  Baseline metrics generation timed out")
            except Exception as e:
                logger.warning(f"  Baseline metrics generation error: {e}")
        else:
            logger.info("  No dataset result directories found, skipping baselines")

        logger.info("=" * 70)

    if failed_count > 0:
        sys.exit(1)
    if interrupted:
        sys.exit(2)


if __name__ == "__main__":
    main()
