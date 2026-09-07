#!/usr/bin/env python
"""Export checkpoint encoder means in SQLite row order with resumable writes."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from lignin_training_data import GpuPackedLigninDataset, PackedLigninDataset, pad_collate
from models.autoregressive_vae import VaeTransformer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), required=True)
    parser.add_argument(
        "--padding-mode",
        choices=("checkpoint-max", "batch-max"),
        default="checkpoint-max",
        help=(
            "checkpoint-max gives legacy checkpoints a deterministic latent independent "
            "of batch composition; batch-max is allowed only for padding-invariant models"
        ),
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def database_layout(path: Path) -> tuple[int, int, int]:
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        minimum, maximum, count = connection.execute(
            "select min(rowid), max(rowid), count(*) from functionalized_lignins"
        ).fetchone()
    return int(minimum), int(maximum), int(count)


def validate_and_write_rowids(
    shard_dirs: list[Path], database_rows: int, output: Path
) -> tuple[list[dict], set[str]]:
    rowids_output = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.int64, shape=(database_rows,)
    )
    shard_layout = []
    tokenizer_hashes = set()
    cursor = 0
    for directory in shard_dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        tokenizer_hashes.add(manifest["tokenizer_sha256"])
        rowids = np.load(directory / "rowids.npy", mmap_mode="r", allow_pickle=False)
        rows = len(rowids)
        expected = np.arange(cursor + 1, cursor + rows + 1, dtype=np.int64)
        if not np.array_equal(rowids, expected):
            mismatch = int(np.flatnonzero(np.asarray(rowids) != expected)[0])
            raise ValueError(
                f"{directory} is not in dense database order at local row {mismatch}: "
                f"found {int(rowids[mismatch])}, expected {int(expected[mismatch])}"
            )
        if int(manifest["rows"]) != rows:
            raise ValueError(f"Manifest row count does not match {directory / 'rowids.npy'}")
        rowids_output[cursor : cursor + rows] = rowids
        shard_layout.append(
            {
                "name": directory.name,
                "path": str(directory),
                "output_start": cursor,
                "output_stop": cursor + rows,
                "rowid_start": int(rowids[0]),
                "rowid_stop": int(rowids[-1]),
                "rows": rows,
            }
        )
        cursor += rows
    rowids_output.flush()
    del rowids_output
    if cursor != database_rows:
        raise ValueError(
            f"Encoded shards contain {cursor:,} rows but the database contains "
            f"{database_rows:,}"
        )
    return shard_layout, tokenizer_hashes


def padded_batch(x: torch.Tensor, width: int | None) -> torch.Tensor:
    if width is None:
        return x
    if x.shape[1] > width:
        raise ValueError(f"Encoded sequence width {x.shape[1]} exceeds padding width {width}")
    return F.pad(x, (0, width - x.shape[1]))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.database = args.database.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.shards = sorted(path.resolve() for path in args.shards)
    args.output_dir = args.output_dir.resolve()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.database.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Database and checkpoint must both exist")
    if not args.shards or any(not (path / "manifest.json").is_file() for path in args.shards):
        raise FileNotFoundError("Every --shards entry must be an encoded shard directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires a visible CUDA GPU")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("CPU export supports only --precision fp32")

    database_min, database_max, database_rows = database_layout(args.database)
    if (database_min, database_max) != (1, database_rows):
        raise ValueError(
            "The exporter requires dense database rowids 1..N; found "
            f"{database_min}..{database_max} for {database_rows} rows"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rowids_path = args.output_dir / "rowids.npy"
    shard_layout, tokenizer_hashes = validate_and_write_rowids(
        args.shards, database_rows, rowids_path
    )
    if len(tokenizer_hashes) != 1:
        raise ValueError(f"Encoded shards use different tokenizers: {tokenizer_hashes}")

    checkpoint_sha256 = sha256_file(args.checkpoint)
    database_sha256 = sha256_file(args.database)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    # Missing means legacy behavior. New checkpoints persist an explicit True.
    model_config.setdefault("padding_invariant_encoder", False)
    checkpoint_tokenizer = checkpoint.get("tokenizer_sha256")
    shard_tokenizer = next(iter(tokenizer_hashes))
    if checkpoint_tokenizer != shard_tokenizer:
        raise ValueError(
            "Checkpoint tokenizer hash does not match encoded shards: "
            f"{checkpoint_tokenizer} != {shard_tokenizer}"
        )

    device = torch.device(args.device)
    model = VaeTransformer(**model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    latent_dim = int(model_config["latent_size"])
    if args.padding_mode == "batch-max" and not model.padding_invariant_encoder:
        raise ValueError(
            "Legacy checkpoints require --padding-mode checkpoint-max because their "
            "latents depend on right-padding width"
        )
    padding_width = (
        int(model_config["max_len"]) if args.padding_mode == "checkpoint-max" else None
    )

    identity = {
        "format_version": 1,
        "database": str(args.database),
        "database_sha256": database_sha256,
        "database_rows": database_rows,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_record": checkpoint.get("record"),
        "tokenizer_sha256": shard_tokenizer,
        "latent_dim": latent_dim,
        "dtype": "float32",
        "shape": [database_rows, latent_dim],
        "padding_mode": args.padding_mode,
        "padding_width": padding_width,
        "padding_invariant_encoder": bool(model.padding_invariant_encoder),
        "shards": shard_layout,
    }
    final_path = args.output_dir / "latents.npy"
    manifest_path = args.output_dir / "manifest.json"
    state_path = args.output_dir / "export_state.json"
    partial_path = args.output_dir / "latents.partial.npy"
    if final_path.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"{final_path} exists without {manifest_path}")
        existing = json.loads(manifest_path.read_text())
        for key in (
            "database_sha256",
            "checkpoint_sha256",
            "database_rows",
            "latent_dim",
            "padding_mode",
        ):
            if existing.get(key) != identity[key]:
                raise ValueError(f"Existing export has incompatible {key}")
        print(f"Latent export already complete: {final_path}")
        return 0

    if state_path.exists():
        state = json.loads(state_path.read_text())
        for key in (
            "database_sha256",
            "checkpoint_sha256",
            "database_rows",
            "latent_dim",
            "padding_mode",
        ):
            if state.get(key) != identity[key]:
                raise ValueError(f"Partial export has incompatible {key}")
        completed = set(state.get("completed_shards", []))
        if not partial_path.is_file():
            raise FileNotFoundError(f"State exists but partial matrix is missing: {partial_path}")
        latents = np.lib.format.open_memmap(partial_path, mode="r+")
        if latents.shape != (database_rows, latent_dim) or latents.dtype != np.float32:
            raise ValueError("Partial latent matrix has the wrong shape or dtype")
    else:
        completed = set()
        latents = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(database_rows, latent_dim),
        )
        state = {**identity, "completed_shards": [], "complete": False}
        atomic_json_dump(state, state_path)

    if device.type == "cuda":
        packed = GpuPackedLigninDataset(args.shards, device)
        corpus = packed.all()
        cpu_corpus = None
    else:
        packed = None
        corpus = None
        cpu_corpus = PackedLigninDataset(args.shards, split=None)

    started = time.time()
    for shard in shard_layout:
        if shard["name"] in completed:
            continue
        start, stop = shard["output_start"], shard["output_stop"]
        progress = tqdm(
            range(start, stop, args.batch_size),
            desc=f"encode {shard['name']}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_start in progress:
            batch_stop = min(batch_start + args.batch_size, stop)
            if corpus is not None:
                x, _ = corpus.fetch(list(range(batch_start, batch_stop)))
                x = x.to(dtype=torch.long)
            else:
                items = [cpu_corpus[index] for index in range(batch_start, batch_stop)]
                x, _ = pad_collate(items)
                x = x.to(device=device, non_blocking=True)
            x = padded_batch(x, padding_width)
            with torch.inference_mode(), amp_context(device, args.precision):
                mu, _ = model.encode(x)
            mu_numpy = mu.detach().float().cpu().numpy()
            if not np.isfinite(mu_numpy).all():
                raise FloatingPointError(
                    f"Non-finite latent values in rows {batch_start + 1}..{batch_stop}"
                )
            latents[batch_start:batch_stop] = mu_numpy
        latents.flush()
        completed.add(shard["name"])
        state = {
            **identity,
            "completed_shards": [
                item["name"] for item in shard_layout if item["name"] in completed
            ],
            "complete": False,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json_dump(state, state_path)

    latents.flush()
    del latents
    os.replace(partial_path, final_path)
    manifest = {
        **identity,
        "complete": True,
        "elapsed_seconds": time.time() - started,
        "latents": str(final_path),
        "rowids": str(rowids_path),
        "latent_bytes": database_rows * latent_dim * np.dtype(np.float32).itemsize,
    }
    atomic_json_dump(manifest, manifest_path)
    atomic_json_dump({**manifest, "completed_shards": [x["name"] for x in shard_layout]}, state_path)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
