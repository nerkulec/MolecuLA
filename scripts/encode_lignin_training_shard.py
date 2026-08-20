#!/usr/bin/env python
"""Encode a preprocessed row shard into packed, unified SELFIES token arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-by", choices=["rowid", "scaffold"], default="rowid")
    p.add_argument("--split-seed", type=int, default=42)
    return p.parse_args()


def splitmix64(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=False)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def main() -> int:
    args = parse_args()
    tokenizer_bytes = args.tokenizer.read_bytes()
    tokenizer = json.loads(tokenizer_bytes)
    vocab = tokenizer["vocab"]
    if vocab[:4] != ["<PAD>", "<SOS>", "<EOS>", "MASK"]:
        raise ValueError("Tokenizer does not use unified special-token indexing")
    tok2id = {token: i for i, token in enumerate(vocab)}
    columns = ["rowid", "preprocess_status", "selfies_final", "scaffold_hash"]
    frame = pd.read_csv(args.rows, compression="infer", usecols=columns, keep_default_na=False)
    frame = frame.loc[frame.preprocess_status.eq("ok")].reset_index(drop=True)

    sequences: list[np.ndarray] = []
    offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    for i, value in enumerate(frame.selfies_final):
        try:
            ids = [1, *(tok2id[token] for token in sf.split_selfies(value)), 2]
        except KeyError as exc:
            raise ValueError(f"OOV token in rowid {frame.rowid.iloc[i]}: {exc}") from exc
        sequence = np.asarray(ids, dtype=np.uint16)
        sequences.append(sequence)
        offsets[i + 1] = offsets[i] + len(sequence)
    tokens = np.concatenate(sequences) if sequences else np.empty(0, dtype=np.uint16)
    lengths = np.diff(offsets)
    length_dtype = np.uint16 if tokenizer["max_sequence_length"] < 65536 else np.uint32
    lengths = lengths.astype(length_dtype)
    rowids = frame.rowid.to_numpy(dtype=np.int64)
    if args.split_by == "scaffold":
        groups = frame.scaffold_hash.astype(str).map(int).to_numpy(dtype=np.uint64)
    else:
        groups = rowids.astype(np.uint64)
    hashed = splitmix64(groups ^ np.uint64(args.split_seed)) % np.uint64(100)
    # 80/10/10 train/validation/test.  Values are stable across shard layouts.
    splits = np.where(hashed < 80, 0, np.where(hashed < 90, 1, 2)).astype(np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "tokens.npy", tokens, allow_pickle=False)
    np.save(args.output_dir / "offsets.npy", offsets, allow_pickle=False)
    np.save(args.output_dir / "lengths.npy", lengths, allow_pickle=False)
    np.save(args.output_dir / "rowids.npy", rowids, allow_pickle=False)
    np.save(args.output_dir / "groups.npy", groups, allow_pickle=False)
    np.save(args.output_dir / "splits.npy", splits, allow_pickle=False)
    manifest = {
        "format_version": 1,
        "source_rows": str(args.rows),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "split_by": args.split_by,
        "split_seed": args.split_seed,
        "rows": len(frame),
        "tokens": len(tokens),
        "max_length": int(lengths.max()) if len(lengths) else 0,
        "split_counts": {name: int((splits == i).sum()) for i, name in enumerate(("train", "val", "test"))},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
