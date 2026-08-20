#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {"linear_attention": 512, "simple_attention": 256, "autoregressive": 256}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate row alignment and numerical integrity of a lignin run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    failures: list[str] = []
    rows = pd.read_csv(root / "preprocess" / "rows.csv.gz", compression="gzip")
    common = np.load(root / "common_eligible.npy", allow_pickle=False).astype(bool)
    if len(rows) != len(common):
        failures.append("preprocessing rows and common mask differ in length")
    expected_rowids = rows.loc[common, "rowid"].to_numpy(dtype=np.int64)
    targets = np.load(root / "probe_targets.npz", allow_pickle=False)
    if not np.array_equal(expected_rowids, targets["rowids"]):
        failures.append("probe target rowids are not aligned with the common mask")
    for encoding in ("nonar", "ar"):
        tokens = np.load(root / "tokenization" / encoding / "tokens.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(root / "tokenization" / encoding / "eligible_mask.npy", allow_pickle=False)
        if tokens.shape != (len(rows), 77):
            failures.append(f"{encoding} token shape is {tokens.shape}, expected {(len(rows), 77)}")
        if mask.shape != (len(rows),):
            failures.append(f"{encoding} mask shape is {mask.shape}")
    for model, dimension in MODELS.items():
        directory = root / "models" / model
        rowids = np.load(directory / "rowids.npy", allow_pickle=False)
        latents = np.load(directory / "latents.npy", mmap_mode="r", allow_pickle=False)
        reconstruction = pd.read_csv(directory / "reconstruction_rows.csv.gz", compression="gzip")
        if not np.array_equal(rowids, expected_rowids):
            failures.append(f"{model} rowids are misaligned")
        if latents.shape != (len(expected_rowids), dimension):
            failures.append(f"{model} latent shape is {latents.shape}")
        if not np.isfinite(latents).all():
            failures.append(f"{model} latents contain non-finite values")
        if len(reconstruction) != len(expected_rowids):
            failures.append(f"{model} reconstruction row count differs")
    probe_report = json.loads((root / "probes" / "probe_report.json").read_text(encoding="utf-8"))
    if probe_report["rows"] != len(expected_rowids):
        failures.append("probe report row count differs")
    if failures:
        print("Lignin run verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Lignin run verification passed: {len(rows)} sampled, {len(expected_rowids)} common eligible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
