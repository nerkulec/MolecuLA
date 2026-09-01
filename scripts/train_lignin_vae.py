#!/usr/bin/env python
"""Train the autoregressive VAE on explicit packed lignin shards."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lignin_training_data import (
    BucketBatchSampler,
    GpuBatchLoader,
    GpuPackedLigninDataset,
    PackedLigninDataset,
    pad_collate,
)
from models.autoregressive_vae import VaeTransformer, vae_loss


MODEL_CONFIG = {
    "hidden_size": 256,
    "latent_size": 512,
    "attn_heads": 8,
    "num_slots": 8,
    "encoder_layers": 3,
    "decoder_layers": 2,
    "padding_invariant_encoder": True,
}
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
MAX_BETA = 0.03
SELECTION_BETA = 0.03
BETA_WARMUP_EPOCHS = 2
LR_WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the autoregressive MolecuLA VAE on packed lignin shards."
    )
    parser.add_argument("--model", choices=["autoregressive"], required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], required=True)
    parser.add_argument("--dataset-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-wall-clock-hours", type=float)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=MODEL_CONFIG["hidden_size"])
    parser.add_argument("--latent-size", type=int, default=MODEL_CONFIG["latent_size"])
    parser.add_argument("--attn-heads", type=int, default=MODEL_CONFIG["attn_heads"])
    parser.add_argument("--num-slots", type=int, default=MODEL_CONFIG["num_slots"])
    parser.add_argument("--encoder-layers", type=int, default=MODEL_CONFIG["encoder_layers"])
    parser.add_argument("--decoder-layers", type=int, default=MODEL_CONFIG["decoder_layers"])
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-beta", type=float, default=MAX_BETA)
    parser.add_argument(
        "--beta-warmup-epochs",
        "--beta-cycle-epochs",
        dest="beta_warmup_epochs",
        type=int,
        default=BETA_WARMUP_EPOCHS,
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=LR_WARMUP_STEPS)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.tokenizer = args.tokenizer.resolve()
    args.shards = sorted(path.resolve() for path in args.shards)
    args.output_dir = args.output_dir.resolve()
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    incomplete = [path for path in args.shards if not (path / "manifest.json").is_file()]
    if incomplete:
        raise FileNotFoundError(f"Missing shard manifests: {incomplete[:5]}")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.log_every_batches < 1:
        raise ValueError("--log-every-batches must be at least 1")
    if args.max_wall_clock_hours is not None and args.max_wall_clock_hours <= 0:
        raise ValueError("--max-wall-clock-hours must be positive")
    if args.hidden_size % args.attn_heads:
        raise ValueError("--hidden-size must be divisible by --attn-heads")
    if args.hidden_size % args.num_slots:
        raise ValueError("--hidden-size must be divisible by --num-slots")
    if args.latent_size % args.num_slots:
        raise ValueError("--latent-size must be divisible by --num-slots")
    for name in ("hidden_size", "latent_size", "attn_heads", "num_slots", "encoder_layers", "decoder_layers"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.max_beta < 0:
        raise ValueError("Learning rate must be positive; weight decay and beta cannot be negative")
    if args.beta_warmup_epochs < 1 or args.grad_clip <= 0:
        raise ValueError("--beta-warmup-epochs and --grad-clip must be positive")
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps cannot be negative")
    return args


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def objective(model: VaeTransformer, x: torch.Tensor, beta: float):
    logits, mu, logvar = model(x, mode="train")
    targets = x[:, 1:]
    loss, reconstruction, kl = vae_loss(logits, targets, mu, logvar, beta=beta)
    predicted = logits.argmax(-1)
    mask = targets.ne(0)
    token_correct = (predicted.eq(targets) & mask).sum()
    exact = exact_match_categories(predicted, targets).sum()
    return loss, reconstruction, kl, token_correct, mask.sum(), exact, mu, logvar


def exact_match_categories(
    predicted: torch.Tensor, targets: torch.Tensor, pad_id: int = 0
) -> torch.Tensor:
    """Return one teacher-forced full-sequence correctness flag per row."""
    if predicted.shape != targets.shape:
        raise ValueError(
            f"Predictions and targets must have the same shape, got "
            f"{tuple(predicted.shape)} and {tuple(targets.shape)}"
        )
    return (predicted.eq(targets) | targets.eq(pad_id)).all(dim=1)


def run_epoch(
    args: argparse.Namespace,
    model: VaeTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    beta: float,
    train: bool,
    description: str,
    epoch: int,
    batch_callback=None,
    beta_schedule=None,
    global_step_start: int = 0,
) -> np.ndarray:
    model.train(train)
    sums = np.zeros(7, dtype=np.float64)
    stage = "train" if train else "val"
    stage_started = time.time()
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)
    for batch_number, (x, _) in enumerate(progress, start=1):
        batch_beta = (
            beta_schedule(global_step_start + batch_number - 1)
            if train and beta_schedule is not None
            else beta
        )
        # The resident corpus is compact int32; embedding and cross-entropy use
        # int64 only for the current padded batch.
        x = x.to(device=device, dtype=torch.long, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), amp_context(device, args.precision):
            (
                loss,
                reconstruction,
                kl,
                token_correct,
                token_total,
                exact,
                mu,
                logvar,
            ) = objective(model, x, batch_beta)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss before backward: "
                f"reconstruction={float(reconstruction.detach())}, "
                f"kl={float(kl.detach())}, mu_abs_max={float(mu.detach().abs().max())}, "
                f"logvar_min={float(logvar.detach().min())}, "
                f"logvar_max={float(logvar.detach().max())}"
            )
        grad_norm = None
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        count = len(x)
        sums += [
            float(loss.detach()) * count,
            float(reconstruction.detach()) * count,
            float(kl.detach()) * count,
            int(token_correct),
            int(token_total),
            int(exact),
            count,
        ]
        if batch_number == 1 or batch_number % 50 == 0:
            progress.set_postfix(
                loss=f"{sums[0] / max(1, sums[6]):.4f}",
                token_acc=f"{sums[3] / max(1, sums[4]):.3f}",
                beta=f"{batch_beta:.5f}",
            )
        should_log = (
            batch_callback is not None
            and (
                batch_number == 1
                or batch_number % args.log_every_batches == 0
                or batch_number == len(loader)
            )
        )
        if should_log:
            elapsed = max(time.time() - stage_started, 1e-9)
            payload = {
                "trainer/epoch": epoch,
                "trainer/stage": stage,
                f"{stage}/batch_index": batch_number,
                f"{stage}/batch_total": len(loader),
                f"{stage}/batch_rows": count,
                f"{stage}/batch_sequence_width": x.shape[1],
                f"{stage}/batch_loss": float(loss.detach()),
                f"{stage}/batch_reconstruction_loss": float(reconstruction.detach()),
                f"{stage}/batch_kl_loss": float(kl.detach()),
                f"{stage}/batch_token_accuracy": int(token_correct) / max(1, int(token_total)),
                f"{stage}/running_loss": sums[0] / max(1, sums[6]),
                f"{stage}/running_reconstruction_loss": sums[1] / max(1, sums[6]),
                f"{stage}/running_kl_loss": sums[2] / max(1, sums[6]),
                f"{stage}/running_token_accuracy": sums[3] / max(1, sums[4]),
                f"{stage}/processed_rows": int(sums[6]),
                f"{stage}/rows_per_second": sums[6] / elapsed,
                "optimization/beta": batch_beta,
                "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                "latent/mu_mean": float(mu.detach().float().mean()),
                "latent/mu_std": float(mu.detach().float().std()),
                "latent/mu_abs_max": float(mu.detach().float().abs().max()),
                "latent/logvar_mean": float(logvar.detach().float().mean()),
                "latent/logvar_min": float(logvar.detach().float().min()),
                "latent/logvar_max": float(logvar.detach().float().max()),
            }
            if grad_norm is not None:
                payload["optimization/gradient_norm"] = float(grad_norm.detach())
            if device.type == "cuda":
                payload.update(
                    {
                        "cuda/memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                        "cuda/memory_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                        "cuda/max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                )
            batch_callback(payload)
    return sums


def metrics(sums: np.ndarray) -> dict:
    rows = max(1, sums[6])
    return {
        "loss": sums[0] / rows,
        "reconstruction_loss": sums[1] / rows,
        "kl_loss": sums[2] / rows,
        "teacher_forced_token_accuracy": sums[3] / max(1, sums[4]),
        "exact_accuracy": sums[5] / rows,
        "rows": int(sums[6]),
    }


def train(args: argparse.Namespace, epoch_callback=None, batch_callback=None) -> int:
    args = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset_device == "cuda" and device.type != "cuda":
        raise RuntimeError("--dataset-device cuda requires a visible CUDA GPU")
    print(f"Training autoregressive model on {device}; output: {args.output_dir}", flush=True)

    tokenizer_bytes = args.tokenizer.read_bytes()
    tokenizer_hash = hashlib.sha256(tokenizer_bytes).hexdigest()
    tokenizer = json.loads(tokenizer_bytes)
    if tokenizer["vocab"][:4] != ["<PAD>", "<SOS>", "<EOS>", "MASK"]:
        raise ValueError("Expected unified PAD/SOS/EOS/MASK indices 0/1/2/3")

    if args.dataset_device == "cuda":
        packed_data = GpuPackedLigninDataset(args.shards, device)
        train_data = packed_data.split("train", args.max_train_samples)
        val_data = packed_data.split("val", args.max_val_samples)
    else:
        train_data = PackedLigninDataset(args.shards, "train", args.max_train_samples)
        val_data = PackedLigninDataset(args.shards, "val", args.max_val_samples)
    if train_data.tokenizer_sha256 != tokenizer_hash:
        raise ValueError("Tokenizer does not match encoded shards")

    train_sampler = BucketBatchSampler(
        train_data.lengths, args.batch_size, True, seed=args.seed
    )
    val_sampler = BucketBatchSampler(
        val_data.lengths, args.batch_size, False, seed=args.seed
    )
    if args.dataset_device == "cuda":
        train_loader = GpuBatchLoader(train_data, train_sampler)
        val_loader = GpuBatchLoader(val_data, val_sampler)
        if args.num_workers:
            print(
                "Dataset is GPU-resident; --num-workers is ignored because batches are "
                "assembled directly on the GPU.",
                flush=True,
            )
    else:
        loader_options = {
            "num_workers": args.num_workers,
            "collate_fn": pad_collate,
            "pin_memory": device.type == "cuda",
        }
        if args.num_workers:
            loader_options["persistent_workers"] = True
        train_loader = DataLoader(train_data, batch_sampler=train_sampler, **loader_options)
        val_loader = DataLoader(val_data, batch_sampler=val_sampler, **loader_options)

    model_config = {
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "attn_heads": args.attn_heads,
        "num_slots": args.num_slots,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "padding_invariant_encoder": MODEL_CONFIG["padding_invariant_encoder"],
        "vocab_size": tokenizer["vocab_size"],
        "max_len": tokenizer["max_sequence_length"],
    }
    model = VaeTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.05,
            end_factor=1.0,
            total_iters=args.lr_warmup_steps,
        )
        if args.lr_warmup_steps > 0
        else None
    )
    scaler_enabled = device.type == "cuda" and args.precision == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "model_config": model_config,
        "tokenizer_sha256": tokenizer_hash,
        "train_rows": len(train_data),
        "val_rows": len(val_data),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_beta": args.max_beta,
        "beta_warmup_epochs": args.beta_warmup_epochs,
        "lr_warmup_steps": args.lr_warmup_steps,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
    }
    serializable_config = json.loads(json.dumps(config, default=str))
    (args.output_dir / "run_config.json").write_text(
        json.dumps(serializable_config, indent=2, sort_keys=True) + "\n"
    )

    best_selection_loss = math.inf
    training_started = time.time()
    global_train_step = 0
    beta_warmup_steps = max(1, args.beta_warmup_epochs * len(train_loader))

    def beta_for_step(step: int) -> float:
        return args.max_beta * min(1.0, (step + 1) / beta_warmup_steps)

    deadline = (
        training_started + args.max_wall_clock_hours * 3600
        if args.max_wall_clock_hours is not None
        else None
    )
    for epoch in range(args.epochs):
        started = time.time()
        train_sampler.set_epoch(epoch)
        # Increase beta smoothly on every optimizer update, reaching max_beta
        # after the configured number of complete passes through the train split.
        beta = beta_for_step(global_train_step)
        train_sums = run_epoch(
            args,
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            beta,
            True,
            f"epoch {epoch + 1}/{args.epochs} train",
            epoch,
            batch_callback,
            beta_for_step,
            global_train_step,
        )
        global_train_step += len(train_loader)
        beta = beta_for_step(global_train_step - 1)
        val_sums = run_epoch(
            args,
            model,
            val_loader,
            optimizer,
            None,
            scaler,
            device,
            beta,
            False,
            f"epoch {epoch + 1}/{args.epochs} val",
            epoch,
            batch_callback,
        )
        train_metrics = metrics(train_sums)
        val_metrics = metrics(val_sums)
        record = {
            "epoch": epoch,
            "beta": beta,
            "train": train_metrics,
            "val": val_metrics,
            "val_selection_loss": (
                val_metrics["reconstruction_loss"] + SELECTION_BETA * val_metrics["kl_loss"]
            ),
            "selection_beta": SELECTION_BETA,
            "elapsed_seconds": time.time() - started,
        }
        is_best = record["val_selection_loss"] < best_selection_loss
        best_selection_loss = min(best_selection_loss, record["val_selection_loss"])
        print(json.dumps(record), flush=True)
        if epoch_callback is not None:
            epoch_callback(record)
        with (args.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {
            "format_version": 1,
            "model_name": "autoregressive",
            "model_config": model_config,
            "tokenizer_sha256": tokenizer_hash,
            "epoch": epoch,
            "best_val_selection_loss": best_selection_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "record": record,
        }
        torch.save(state, args.output_dir / "last.pt")
        if is_best:
            torch.save(state, args.output_dir / "best.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(state, args.output_dir / f"epoch_{epoch:03d}.pt")
        if deadline is not None and epoch + 1 < args.epochs:
            remaining = deadline - time.time()
            # Stop before an additional epoch is likely to overrun the budget.
            if remaining < record["elapsed_seconds"] * 1.1:
                print(
                    json.dumps(
                        {
                            "stopped": "wall_clock_budget",
                            "completed_epochs": epoch + 1,
                            "elapsed_seconds": time.time() - training_started,
                            "budget_hours": args.max_wall_clock_hours,
                        }
                    ),
                    flush=True,
                )
                break
    return 0


def main() -> int:
    return train(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
