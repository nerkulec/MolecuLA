#!/usr/bin/env python
"""Run one W&B sweep trial for autoregressive lignin VAE training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

from train_lignin_vae import train


def parse_bool(value: str) -> bool:
    normalized = str(value).lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], required=True)
    parser.add_argument("--dataset-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-wall-clock-hours", type=float, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--log-checkpoint", type=parse_bool, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument("--attn-heads", type=int, required=True)
    parser.add_argument("--num-slots", type=int, required=True)
    parser.add_argument("--encoder-layers", type=int, required=True)
    parser.add_argument("--decoder-layers", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--max-beta", type=float, required=True)
    parser.add_argument(
        "--beta-warmup-epochs",
        "--beta-cycle-epochs",
        dest="beta_warmup_epochs",
        type=int,
        default=2,
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def flatten_record(record: dict) -> dict:
    payload = {
        "epoch": record["epoch"],
        "beta": record["beta"],
        "epoch/elapsed_seconds": record["elapsed_seconds"],
        "val/selection_loss": record["val_selection_loss"],
        "val/selection_beta": record["selection_beta"],
    }
    for split in ("train", "val"):
        for name, value in record[split].items():
            payload[f"{split}/{name}"] = value
    return payload


def main() -> int:
    sweep_args = parse_args()
    shard_dirs = sorted(sweep_args.shard_root.resolve().glob("shard_*"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories under {sweep_args.shard_root}")

    wandb_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(sweep_args).items()
    }
    with wandb.init(
        project=sweep_args.wandb_project,
        mode=sweep_args.wandb_mode,
        config=wandb_config,
    ) as run:
        output_dir = sweep_args.output_root.resolve() / run.id
        training_args = argparse.Namespace(
            model="autoregressive",
            tokenizer=sweep_args.tokenizer,
            shards=shard_dirs,
            output_dir=output_dir,
            epochs=sweep_args.epochs,
            batch_size=sweep_args.batch_size,
            num_workers=sweep_args.num_workers,
            precision=sweep_args.precision,
            dataset_device=sweep_args.dataset_device,
            max_train_samples=sweep_args.max_train_samples,
            max_val_samples=sweep_args.max_val_samples,
            max_wall_clock_hours=sweep_args.max_wall_clock_hours,
            save_every=sweep_args.save_every,
            log_every_batches=sweep_args.log_every_batches,
            hidden_size=sweep_args.hidden_size,
            latent_size=sweep_args.latent_size,
            attn_heads=sweep_args.attn_heads,
            num_slots=sweep_args.num_slots,
            encoder_layers=sweep_args.encoder_layers,
            decoder_layers=sweep_args.decoder_layers,
            learning_rate=sweep_args.learning_rate,
            weight_decay=sweep_args.weight_decay,
            max_beta=sweep_args.max_beta,
            beta_warmup_epochs=sweep_args.beta_warmup_epochs,
            lr_warmup_steps=sweep_args.lr_warmup_steps,
            grad_clip=sweep_args.grad_clip,
            seed=sweep_args.seed,
        )

        best = {"loss": float("inf"), "record": None}

        def log_epoch(record: dict) -> None:
            run.log(flatten_record(record))
            if record["val_selection_loss"] < best["loss"]:
                best["loss"] = record["val_selection_loss"]
                best["record"] = record

        train(training_args, epoch_callback=log_epoch, batch_callback=run.log)
        run.summary["val/selection_loss"] = best["loss"]
        run.summary["best/val_selection_loss"] = best["loss"]
        if best["record"] is not None:
            run.summary["best/epoch"] = best["record"]["epoch"]
            run.summary["best/val_token_accuracy"] = best["record"]["val"][
                "teacher_forced_token_accuracy"
            ]
            run.summary["best/val_reconstruction_loss"] = best["record"]["val"][
                "reconstruction_loss"
            ]
        if sweep_args.log_checkpoint:
            artifact = wandb.Artifact(
                name=f"autoregressive-{run.id}",
                type="model",
                metadata=json.loads((output_dir / "run_config.json").read_text()),
            )
            artifact.add_file(output_dir / "best.pt")
            artifact.add_file(output_dir / "run_config.json")
            run.log_artifact(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
