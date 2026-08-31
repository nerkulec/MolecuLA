#!/usr/bin/env python
"""Display dataset-wide progress for a shared lignin preparation queue."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preprocess", "encode"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--total-rows", type=int, required=True)
    parser.add_argument("--rows-per-shard", type=int, required=True)
    parser.add_argument("--tokenizer-sha")
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def shard_name(index: int) -> str:
    return f"shard_{index:03d}"


def marker_is_valid(args: argparse.Namespace, directory: Path) -> bool:
    if args.stage == "preprocess":
        return (directory / ".complete").is_file()
    marker = directory / ".tokenizer_sha256"
    if not marker.is_file() or args.tokenizer_sha is None:
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == args.tokenizer_sha
    except OSError:
        return False


def inspect(args: argparse.Namespace) -> tuple[int, int, list[Path]]:
    completed_rows = 0
    active = 0
    failures: list[Path] = []
    for index in range(args.shards):
        directory = args.root / shard_name(index)
        if marker_is_valid(args, directory):
            completed_rows += min(
                args.rows_per_shard,
                args.total_rows - index * args.rows_per_shard,
            )
        elif (directory / ".failed").is_file():
            failures.append(directory / ".failed")
        elif (directory / ".claim").is_dir():
            active += 1
    return completed_rows, active, failures


def main() -> int:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    description = "canonicalize (global)" if args.stage == "preprocess" else "encode (global)"
    bar = tqdm(total=args.total_rows, desc=description, unit="mol", dynamic_ncols=True, mininterval=args.interval)
    try:
        while True:
            completed_rows, active, failures = inspect(args)
            bar.update(completed_rows - int(bar.n))
            bar.set_postfix(active=active, failed=len(failures), refresh=True)
            if failures:
                bar.close()
                print(f"{args.stage} failed for {len(failures)} shard(s):", flush=True)
                for path in failures[:20]:
                    print(f"  {path}", flush=True)
                if len(failures) > 20:
                    print(f"  ... and {len(failures) - 20} more", flush=True)
                return 1
            if completed_rows >= args.total_rows:
                return 0
            time.sleep(args.interval)
    finally:
        if not bar.disable:
            bar.close()


if __name__ == "__main__":
    raise SystemExit(main())
