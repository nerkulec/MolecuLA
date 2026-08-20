#!/usr/bin/env python
"""Build one SELFIES vocabulary shared by all three VAE architectures."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import selfies as sf


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "MASK"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, nargs="+", required=True, help="Preprocessed rows.csv[.gz] shards")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunksize", type=int, default=100_000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    counts: Counter[str] = Counter()
    rows_seen = rows_ok = max_chemical_len = 0
    for path in sorted(args.rows):
        for frame in pd.read_csv(
            path, compression="infer", usecols=["preprocess_status", "selfies_final"],
            keep_default_na=False, chunksize=args.chunksize,
        ):
            rows_seen += len(frame)
            for value in frame.loc[frame.preprocess_status.eq("ok"), "selfies_final"]:
                tokens = list(sf.split_selfies(value))
                counts.update(tokens)
                rows_ok += 1
                max_chemical_len = max(max_chemical_len, len(tokens))
    vocab = SPECIAL_TOKENS + sorted(counts)
    if len(vocab) >= 65536:
        raise ValueError("Vocabulary cannot be represented as uint16")
    canonical = json.dumps(vocab, ensure_ascii=False, separators=(",", ":"))
    payload = {
        "format_version": 1,
        "indexing": {"pad": 0, "sos": 1, "eos": 2, "mask": 3},
        "special_tokens": SPECIAL_TOKENS,
        "vocab": vocab,
        "token_to_id": {token: i for i, token in enumerate(vocab)},
        "vocab_size": len(vocab),
        "max_chemical_tokens": max_chemical_len,
        "max_sequence_length": max_chemical_len + 2,
        "rows_seen": rows_seen,
        "rows_ok": rows_ok,
        "chemical_token_counts": dict(sorted(counts.items())),
        "vocab_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {rows_ok:,} molecules, {len(vocab)} tokens, max length {max_chemical_len + 2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
