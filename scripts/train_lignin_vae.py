#!/usr/bin/env python
"""Train any MolecuLA VAE on unified packed lignin SELFIES shards."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from lignin_training_data import BucketBatchSampler, PackedLigninDataset, pad_collate


MODEL_CONFIGS = {
    "simple_attention": dict(hidden_size=256, latent_size=512, attn_heads=8, num_slots=8, layers=1),
    "linear_attention": dict(hidden_size=256, latent_size=1024, attn_heads=8, num_slots=8, layers=1),
    "autoregressive": dict(hidden_size=256, latent_size=512, attn_heads=8, num_slots=8, encoder_layers=3, decoder_layers=2),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--shards", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256, help="Per-process batch size")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-beta", type=float, default=0.03)
    p.add_argument("--beta-cycle-epochs", type=int, default=10)
    p.add_argument("--length-loss-weight", type=float)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path)
    p.add_argument("--greedy-val-samples", type=int, default=64)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-val-samples", type=int)
    p.add_argument("--save-every", type=int, default=1)
    return p.parse_args()


def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world, device


def build_model(name, vocab_size, max_len):
    config = MODEL_CONFIGS[name]
    if name == "simple_attention":
        from models.simple_attention_vae import VaeTransformer
    elif name == "linear_attention":
        from models.linear_attention_vae import VaeTransformer
    else:
        from models.autoregressive_vae import VaeTransformer
    return VaeTransformer(vocab_size=vocab_size, max_len=max_len, **config)


def amp_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def reduce(values, device, world):
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tensor)
    return tensor.cpu().numpy()


def objective(name, model, x, beta, length_weight):
    if name == "autoregressive":
        from models.autoregressive_vae import vae_loss
        logits, mu, logvar = model(x, mode="train")
        targets = x[:, 1:]
        loss, rec, kl = vae_loss(logits, targets, mu, logvar, beta=beta)
        length_loss = loss.new_zeros(())
    else:
        module = __import__(f"models.{name}_vae", fromlist=["vae_loss"])
        logits, mu, logvar, pred_len = model(x, mode="train")
        targets = x
        loss, rec, kl, length_loss = module.vae_loss(
            logits, targets, mu, logvar, pred_len, beta=beta, alpha=length_weight
        )
    predicted = logits.argmax(-1)
    mask = targets.ne(0)
    token_correct = (predicted.eq(targets) & mask).sum()
    exact = (predicted.eq(targets) | ~mask).all(1).sum()
    return loss, rec, kl, length_loss, token_correct, mask.sum(), exact


def run_epoch(args, model, loader, optimizer, scaler, device, beta, train):
    model.train(train)
    sums = np.zeros(8, dtype=np.float64)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), amp_context(device, args.precision):
            loss, rec, kl, length_loss, token_correct, token_total, exact = objective(
                args.model, model, x, beta, args.length_loss_weight
            )
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        count = len(x)
        sums += [float(loss.detach()) * count, float(rec.detach()) * count,
                 float(kl.detach()) * count, float(length_loss.detach()) * count,
                 int(token_correct), int(token_total), int(exact), count]
    return sums


@torch.inference_mode()
def greedy_exact(name, model, loader, device, limit):
    if limit <= 0:
        return np.zeros(2)
    base = model.module if isinstance(model, DDP) else model
    base.eval()
    correct = total = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu, _ = base.encode(x)
        if name == "autoregressive":
            generated = base.decode(mu, max_len=base.max_len)
            lengths = torch.full((len(x),), generated.size(1), device=device)
        else:
            logits, pred_len = base.decode(mu, mode="eval")
            generated = logits.argmax(-1)
            lengths = pred_len.round().long().clamp(min=1, max=base.max_len)
        for target, prediction, length in zip(x, generated, lengths):
            target = target[target.ne(0)]
            prediction = prediction[: int(length)]
            eos = torch.nonzero(prediction.eq(2), as_tuple=False)
            if len(eos):
                prediction = prediction[: int(eos[0]) + 1]
            correct += int(torch.equal(target, prediction))
            total += 1
            if total >= limit:
                return np.array([correct, total])
    return np.array([correct, total])


def metric_dict(sums):
    rows = max(1, sums[7])
    return {"loss": sums[0] / rows, "reconstruction_loss": sums[1] / rows,
            "kl_loss": sums[2] / rows, "length_loss": sums[3] / rows,
            "teacher_forced_token_accuracy": sums[4] / max(1, sums[5]),
            "teacher_forced_exact_accuracy": sums[6] / rows, "rows": int(sums[7])}


def main():
    args = parse_args()
    rank, local_rank, world, device = setup_distributed()
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    tokenizer_bytes = args.tokenizer.read_bytes()
    tokenizer_hash = hashlib.sha256(tokenizer_bytes).hexdigest()
    tokenizer = json.loads(tokenizer_bytes)
    if tokenizer["vocab"][:4] != ["<PAD>", "<SOS>", "<EOS>", "MASK"]:
        raise ValueError("Expected unified PAD/SOS/EOS/MASK indices 0/1/2/3")
    train_data = PackedLigninDataset(args.shards, "train", args.max_train_samples)
    val_data = PackedLigninDataset(args.shards, "val", args.max_val_samples)
    if train_data.tokenizer_sha256 != tokenizer_hash:
        raise ValueError("Tokenizer does not match encoded shards")
    train_sampler = BucketBatchSampler(train_data.lengths, args.batch_size, True, args.seed, rank, world)
    val_sampler = BucketBatchSampler(val_data.lengths, args.batch_size, False, args.seed, rank, world)
    loader_args = dict(num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=device.type == "cuda")
    if args.num_workers: loader_args["persistent_workers"] = True
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_data, batch_sampler=val_sampler, **loader_args)
    model_config = {**MODEL_CONFIGS[args.model], "vocab_size": tokenizer["vocab_size"],
                    "max_len": tokenizer["max_sequence_length"]}
    model = build_model(args.model, tokenizer["vocab_size"], tokenizer["max_sequence_length"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler_enabled = device.type == "cuda" and args.precision == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.1 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    start_epoch, best_val = 0, math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint["model_name"] != args.model or checkpoint["model_config"] != model_config:
            raise ValueError("Resume checkpoint architecture mismatch")
        if checkpoint["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError("Resume checkpoint tokenizer mismatch")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_val = checkpoint["epoch"] + 1, checkpoint.get("best_val_loss", best_val)
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=args.model == "autoregressive")
    if args.length_loss_weight is None:
        # Raw squared token errors can be O(max_len^2) and otherwise overwhelm
        # the roughly O(1) token cross-entropy. Keep the old relative weights,
        # but make their scale independent of this dataset's sequence length.
        base_weight = 0.1 if args.model == "linear_attention" else 1.0
        args.length_loss_weight = base_weight / tokenizer["max_sequence_length"] ** 2
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = vars(args).copy()
        config.update(model_config=model_config, tokenizer_sha256=tokenizer_hash, world_size=world,
                      train_rows=len(train_data), val_rows=len(val_data))
        config = json.loads(json.dumps(config, default=str))
        (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for epoch in range(start_epoch, args.epochs):
        started = time.time(); train_sampler.set_epoch(epoch)
        beta = args.max_beta if args.beta_cycle_epochs <= 0 else args.max_beta * ((epoch % args.beta_cycle_epochs) / max(1, args.beta_cycle_epochs - 1))
        train_sums = reduce(run_epoch(args, model, train_loader, optimizer, scaler, device, beta, True), device, world)
        val_sums = reduce(run_epoch(args, model, val_loader, optimizer, scaler, device, beta, False), device, world)
        greedy = reduce(greedy_exact(args.model, model, val_loader, device, math.ceil(args.greedy_val_samples / world)), device, world)
        record = {"epoch": epoch, "beta": beta, "train": metric_dict(train_sums), "val": metric_dict(val_sums),
                  "val_greedy_exact_accuracy": greedy[0] / max(1, greedy[1]), "val_greedy_rows": int(greedy[1]),
                  "elapsed_seconds": time.time() - started}
        is_best = record["val"]["loss"] < best_val; best_val = min(best_val, record["val"]["loss"])
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (args.output_dir / "metrics.jsonl").open("a") as handle: handle.write(json.dumps(record) + "\n")
            base = model.module if isinstance(model, DDP) else model
            state = {"format_version": 1, "model_name": args.model, "model_config": model_config,
                     "tokenizer_sha256": tokenizer_hash, "epoch": epoch, "best_val_loss": best_val,
                     "model": base.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "record": record}
            torch.save(state, args.output_dir / "last.pt")
            if is_best: torch.save(state, args.output_dir / "best.pt")
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(state, args.output_dir / f"epoch_{epoch:03d}.pt")
        if world > 1: dist.barrier()
    if world > 1: dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
