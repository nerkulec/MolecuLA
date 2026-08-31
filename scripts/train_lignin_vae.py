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

from lignin_training_data import BucketBatchSampler, PackedLigninDataset, pad_collate
from models.autoregressive_vae import VaeTransformer, vae_loss


MODEL_CONFIG = {
    "hidden_size": 256,
    "latent_size": 512,
    "attn_heads": 8,
    "num_slots": 8,
    "encoder_layers": 3,
    "decoder_layers": 2,
}
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
MAX_BETA = 0.03
SELECTION_BETA = 0.03
BETA_CYCLE_EPOCHS = 10
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
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--greedy-val-samples", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=MODEL_CONFIG["hidden_size"])
    parser.add_argument("--latent-size", type=int, default=MODEL_CONFIG["latent_size"])
    parser.add_argument("--attn-heads", type=int, default=MODEL_CONFIG["attn_heads"])
    parser.add_argument("--num-slots", type=int, default=MODEL_CONFIG["num_slots"])
    parser.add_argument("--encoder-layers", type=int, default=MODEL_CONFIG["encoder_layers"])
    parser.add_argument("--decoder-layers", type=int, default=MODEL_CONFIG["decoder_layers"])
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-beta", type=float, default=MAX_BETA)
    parser.add_argument("--beta-cycle-epochs", type=int, default=BETA_CYCLE_EPOCHS)
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
    if args.greedy_val_samples < 0:
        raise ValueError("--greedy-val-samples cannot be negative")
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
    if args.beta_cycle_epochs < 1 or args.grad_clip <= 0:
        raise ValueError("--beta-cycle-epochs and --grad-clip must be positive")
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
    exact = (predicted.eq(targets) | ~mask).all(1).sum()
    return loss, reconstruction, kl, token_correct, mask.sum(), exact


def run_epoch(
    args: argparse.Namespace,
    model: VaeTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    beta: float,
    train: bool,
) -> np.ndarray:
    model.train(train)
    sums = np.zeros(7, dtype=np.float64)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), amp_context(device, args.precision):
            loss, reconstruction, kl, token_correct, token_total, exact = objective(
                model, x, beta
            )
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
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
    return sums


@torch.inference_mode()
def greedy_exact(
    model: VaeTransformer, loader: DataLoader, device: torch.device, limit: int
) -> np.ndarray:
    if limit == 0:
        return np.zeros(2, dtype=np.int64)
    model.eval()
    correct = total = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu, _ = model.encode(x)
        generated = model.decode(mu, max_len=model.max_len)
        for target, prediction in zip(x, generated):
            target = target[target.ne(0)]
            eos = torch.nonzero(prediction.eq(2), as_tuple=False)
            if len(eos):
                prediction = prediction[: int(eos[0]) + 1]
            correct += int(torch.equal(target, prediction))
            total += 1
            if total >= limit:
                return np.array([correct, total])
    return np.array([correct, total])


def metrics(sums: np.ndarray) -> dict:
    rows = max(1, sums[6])
    return {
        "loss": sums[0] / rows,
        "reconstruction_loss": sums[1] / rows,
        "kl_loss": sums[2] / rows,
        "teacher_forced_token_accuracy": sums[3] / max(1, sums[4]),
        "teacher_forced_exact_accuracy": sums[5] / rows,
        "rows": int(sums[6]),
    }


def train(args: argparse.Namespace, epoch_callback=None) -> int:
    args = validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training autoregressive model on {device}; output: {args.output_dir}", flush=True)

    tokenizer_bytes = args.tokenizer.read_bytes()
    tokenizer_hash = hashlib.sha256(tokenizer_bytes).hexdigest()
    tokenizer = json.loads(tokenizer_bytes)
    if tokenizer["vocab"][:4] != ["<PAD>", "<SOS>", "<EOS>", "MASK"]:
        raise ValueError("Expected unified PAD/SOS/EOS/MASK indices 0/1/2/3")

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
        "vocab_size": tokenizer["vocab_size"],
        "max_len": tokenizer["max_sequence_length"],
    }
    model = VaeTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
        "beta_cycle_epochs": args.beta_cycle_epochs,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
    }
    serializable_config = json.loads(json.dumps(config, default=str))
    (args.output_dir / "run_config.json").write_text(
        json.dumps(serializable_config, indent=2, sort_keys=True) + "\n"
    )

    best_selection_loss = math.inf
    for epoch in range(args.epochs):
        started = time.time()
        train_sampler.set_epoch(epoch)
        if args.beta_cycle_epochs == 1:
            beta = args.max_beta
        else:
            beta = args.max_beta * (
                (epoch % args.beta_cycle_epochs) / (args.beta_cycle_epochs - 1)
            )
        train_sums = run_epoch(
            args, model, train_loader, optimizer, scaler, device, beta, True
        )
        val_sums = run_epoch(
            args, model, val_loader, optimizer, scaler, device, beta, False
        )
        greedy = greedy_exact(model, val_loader, device, args.greedy_val_samples)
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
            "val_greedy_exact_accuracy": greedy[0] / max(1, greedy[1]),
            "val_greedy_rows": int(greedy[1]),
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
            "record": record,
        }
        torch.save(state, args.output_dir / "last.pt")
        if is_best:
            torch.save(state, args.output_dir / "best.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(state, args.output_dir / f"epoch_{epoch:03d}.pt")
    return 0


def main() -> int:
    return train(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
