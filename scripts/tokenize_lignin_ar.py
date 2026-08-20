#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from lignin_pipeline.tokenize import tokenize_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize canonical lignin SELFIES for the AR checkpoint.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, default=Path("data/selfies_tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tokenize_file("ar", args.rows, args.base_tokenizer, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
