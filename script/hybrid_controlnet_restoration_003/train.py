# --------------------------------------------------------------------------
# Thesis Implementation: Hybrid-003 Joint UNet + ControlNet + ARNIQA
# Blind Image Restoration
#
# Training script for Hybrid-003 approach.
# Based on script/hybrid_controlnet_restoration/train.py with key differences:
# - Uses MarigoldHybridControlNetArniqa003Pipeline (8ch UNet + ControlNet + ARNIQA)
# - Uses MarigoldHybridControlNetArniqa003Trainer
# - No base_checkpoint_path — UNet starts from SD2 original weights
# - UNet loaded in float32 (trainable, not frozen like hybrid-002)
# --------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Workaround for transformers>=4.52 / huggingface_hub>=1.0 bug:
# list_repo_templates() raises RepositoryNotFoundError when a repo doesn't have
# additional_chat_templates/ (e.g., SD2). Monkey-patch to return empty list on error.
import transformers.tokenization_utils_base as _tub
_orig_list_repo_templates = _tub.list_repo_templates
def _safe_list_repo_templates(*args, **kwargs):
    try:
        return _orig_list_repo_templates(*args, **kwargs)
    except Exception:
        return []
_tub.list_repo_templates = _safe_list_repo_templates
import argparse
import logging
import shutil
import torch
from datetime import datetime, timedelta
from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, UNet2DConditionModel
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List

from marigold import MarigoldHybridControlNetArniqa003Pipeline
from src.dataset import DatasetMode
from src.dataset.restoration_dataset_factory import RestorationDatasetFactory
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer.marigold_hybrid_controlnet_arniqa_003_trainer import (
    MarigoldHybridControlNetArniqa003Trainer,
)
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.slurm_util import get_local_scratch_dir, is_on_slurm


if "__main__" == __name__:
    t_start = datetime.now()
    logging.info(f"Started at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Hybrid-003 Joint UNet+ControlNet+ARNIQA Image Restoration : Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold_hybrid_controlnet_restoration_003.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save checkpoints."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Run without Weights and Biases logging.",
    )
    parser.add_argument(
        "--do_not_copy_data",
        action="store_true",
        help="On Slurm cluster, do not copy data to the local scratch.",
    )
    parser.add_argument(
        "--base_data_dir", type=str, default=None, help="Base path to the datasets."
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="Base path to the pretrained checkpoints.",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name.",
    )

    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ.get("BASE_DATA_DIR", "./data")
    )
    base_data_dir = os.path.abspath(base_data_dir)
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ.get("BASE_CKPT_DIR", "./checkpoints")
    )

    # -------------------- Initialization --------------------
    if resume_run is not None:
        logging.info(f"Resuming run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        cfg = recursive_load_config(args.config)
        pure_job_name = os.path.basename(args.config).split(".")[0]
        if args.add_datetime_prefix:
            job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
        else:
            job_name = pure_job_name

        if output_dir is not None:
            out_dir_run = os.path.join(output_dir, job_name)
        else:
            out_dir_run = os.path.join("./output", job_name)
        os.makedirs(out_dir_run, exist_ok=False)

    cfg_data = cfg.dataset

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    if not os.path.exists(out_dir_ckpt):
        os.makedirs(out_dir_ckpt)
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    if not os.path.exists(out_dir_tb):
        os.makedirs(out_dir_tb)
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    if not os.path.exists(out_dir_eval):
        os.makedirs(out_dir_eval)
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    if not os.path.exists(out_dir_vis):
        os.makedirs(out_dir_vis)

    # -------------------- Logging settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")

    # Initialize wandb
    if not args.no_wandb:
        if resume_run is not None:
            wandb_id = load_wandb_job_id(out_dir_run)
            wandb_cfg_dict = {
                "id": wandb_id,
                "resume": "must",
                **cfg.wandb,
            }
        else:
            wandb_cfg_dict = {
                "config": dict(cfg),
                "name": job_name,
                "mode": "online",
                **cfg.wandb,
            }
        wandb_cfg_dict.update({"dir": out_dir_run})
        wandb_run = init_wandb(enable=True, **wandb_cfg_dict)
        save_wandb_job_id(wandb_run, out_dir_run)
    else:
        init_wandb(enable=False)

    # Tensorboard
    tb_logger.set_dir(out_dir_tb)
    log_slurm_job_id(step=0)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None:
        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"Config saved to {_output_path}")
        _temp_code_dir = os.path.join(out_dir_run, "code_tar")
        _code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
        os.system(
            f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'"
        )
        os.system(f"tar -cf {_code_snapshot_path} {_temp_code_dir}")
        os.system(f"rm -rf {_temp_code_dir}")
        logging.info(f"Code snapshot saved to: {_code_snapshot_path}")

    # -------------------- Copy data to local scratch (Slurm) --------------------
    if is_on_slurm() and (not args.do_not_copy_data):
        original_data_dir = base_data_dir
        base_data_dir = os.path.join(get_local_scratch_dir(), "Marigold_data")
        required_data_list = find_value_in_omegaconf("dir", cfg_data)
        required_data_list = list(set(required_data_list))
        logging.info(f"Required_data_list: {required_data_list}")
        for d in tqdm(required_data_list, desc="Copy data to local scratch"):
            ori_dir = os.path.join(original_data_dir, d)
            dst_dir = os.path.join(base_data_dir, d)
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            if os.path.isfile(ori_dir):
                shutil.copyfile(ori_dir, dst_dir)
            elif os.path.isdir(ori_dir):
                shutil.copytree(ori_dir, dst_dir)
        logging.info(f"Data copied to: {base_data_dir}")

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)
    logging.info(
        f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}"
    )

    # -------------------- Data --------------------
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    train_dataset = RestorationDatasetFactory.create_dataset(
        cfg_data.train,
        mode=DatasetMode.TRAIN,
        base_data_dir=base_data_dir,
    )

    logging.debug("Augmentation: ", cfg.augmentation)
    if "mixed" == cfg_data.train.name:
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=True,
            generator=loader_generator,
        )

    # Validation dataset
    val_batch_size = cfg.validation.get('batch_size', 1)
    val_num_workers = cfg.validation.get('num_workers', cfg.dataloader.num_workers)
    val_pin_memory = cfg.validation.get('pin_memory', True)
    val_persistent_workers = cfg.validation.get('persistent_workers', True) and val_num_workers > 0

    val_loaders: List[DataLoader] = []
    for _val_dict in cfg_data.val:
        _val_dataset = RestorationDatasetFactory.create_dataset(
            _val_dict,
            mode=DatasetMode.EVAL,
            base_data_dir=base_data_dir,
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=val_pin_memory,
            persistent_workers=val_persistent_workers,
        )
        val_loaders.append(_val_loader)

    # Visualization dataset
    vis_loaders: List[DataLoader] = []
    for _vis_dict in cfg_data.vis:
        _vis_dataset = RestorationDatasetFactory.create_dataset(
            _vis_dict,
            mode=DatasetMode.EVAL,
            base_data_dir=base_data_dir,
        )
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=val_pin_memory,
            persistent_workers=val_persistent_workers,
        )
        vis_loaders.append(_vis_loader)

    # -------------------- Model --------------------
    # KEY DIFFERENCE from hybrid-002: UNet loaded in float32 (trainable)
    # ControlNet also float32 (trainable). VAE and text_encoder in float16 (frozen).
    pretrained_path = cfg.model.pretrained_path
    logging.info(f"Loading SD2 base model from: {pretrained_path}")

    # UNet in float32 — TRAINABLE (unlike hybrid-002 which used float16 for frozen UNet)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_path, subfolder="unet", torch_dtype=torch.float32
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_path, subfolder="vae", torch_dtype=torch.float16
    )
    scheduler = DDIMScheduler.from_pretrained(
        pretrained_path, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_path, subfolder="text_encoder", torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_path, subfolder="tokenizer"
    )

    # Create ControlNet from UNet (float32 — trainable)
    controlnet = ControlNetModel.from_unet(unet)
    logging.info(
        f"ControlNet created from UNet: "
        f"{sum(p.numel() for p in controlnet.parameters()) / 1e6:.1f}M params (float32)"
    )

    # Pipeline kwargs from config
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}

    model = MarigoldHybridControlNetArniqa003Pipeline(
        unet=unet,
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        **_pipeline_kwargs,
    )
    logging.info("MarigoldHybridControlNetArniqa003Pipeline instantiated")

    # -------------------- Trainer --------------------
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    # Hybrid-003: NO base_checkpoint_path — UNet starts from SD2 original weights.
    # The trainer handles 4→8ch conv_in expansion internally.
    trainer = MarigoldHybridControlNetArniqa003Trainer(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        device=device,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=val_loaders,
        vis_dataloaders=vis_loaders,
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=True
        )

    # -------------------- Training --------------------
    try:
        trainer.train(t_end=t_end)
    except Exception as e:
        logging.exception(e)
