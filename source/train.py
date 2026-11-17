#!/usr/bin/env python3
"""训练入口：读配置 → 建模型 → 写实验元信息 → 交给 lucidrains Trainer（参考模板）。"""
import argparse, json, random, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(base_path, exp_path):
    base = yaml.safe_load(Path(base_path).read_text()) or {}
    exp = yaml.safe_load(Path(exp_path).read_text()) or {}
    return deep_merge(base, exp)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return bool(out)
    except Exception:
        return None


def write_run_meta(run_dir, cfg):
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=False)  # 新实验禁止覆盖已存在目录

    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    meta = {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "entrypoint": "scripts/train.py",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    env_dir = run_dir / "environment"
    env_dir.mkdir()
    with open(env_dir / "pip-freeze.txt", "w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)
    with open(env_dir / "torch.txt", "w") as f:
        f.write(
            f"torch {torch.__version__}\n"
            f"cuda build {torch.version.cuda}\n"
            f"cuda available {torch.cuda.is_available()}\n"
        )
    try:
        with open(env_dir / "nvidia-smi.txt", "w") as f:
            subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--base-config", default=str(ROOT / "configs" / "base.yaml"))
    p.add_argument("--resume", type=int, default=None, help="从 model-{N}.pt 续训（N 是 milestone）")
    p.add_argument("--max-steps", type=int, default=None, help="覆盖 train_num_steps（记录进 config.resolved.yaml）")
    args = p.parse_args()

    cfg = load_config(args.base_config, args.config)
    if args.max_steps is not None:
        cfg["training"]["train_num_steps"] = args.max_steps  # 显式覆盖，会被写进 config.resolved.yaml
    set_seed(cfg.get("seed", 42))
    run_dir = Path(args.run_dir).expanduser()

    if args.resume is None:
        write_run_meta(run_dir, cfg)  # 先占坑：新实验拒绝覆盖已存在目录

    model = Unet(dim=cfg["model"]["dim"], dim_mults=tuple(cfg["model"]["dim_mults"]))
    diffusion = GaussianDiffusion(
        model,
        image_size=cfg["data"]["image_size"],
        timesteps=cfg["diffusion"]["timesteps"],
        sampling_timesteps=cfg["diffusion"].get("sampling_timesteps"),
        objective=cfg["diffusion"]["objective"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    )

    t = cfg["training"]
    trainer = Trainer(
        diffusion,
        str(Path(cfg["data"]["image_folder"]).expanduser()),
        train_batch_size=t["train_batch_size"],
        gradient_accumulate_every=t["gradient_accumulate_every"],
        train_lr=t["train_lr"],
        train_num_steps=t["train_num_steps"],
        ema_decay=t["ema_decay"],
        ema_update_every=t["ema_update_every"],
        save_and_sample_every=t["save_and_sample_every"],
        num_samples=t["num_samples"],
        results_folder=str(run_dir),
        amp=t.get("amp", False),
        calculate_fid=t.get("calculate_fid", False),
        num_fid_samples=t.get("num_fid_samples", 50000),
        tensorboard_dir=str(run_dir / "tensorboard"),
        metrics_jsonl=str(run_dir / "metrics.jsonl"),
        log_file=str(run_dir / "train.log"),
    )

    if args.resume is not None:
        trainer.load(args.resume)
        print(f"resumed from milestone {args.resume}, step={trainer.step}")

    trainer.train()


if __name__ == "__main__":
    main()