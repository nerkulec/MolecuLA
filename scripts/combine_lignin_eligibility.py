#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lignin_pipeline.common import json_dump, percentage, read_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Intersect non-AR and AR eligibility masks.")
    parser.add_argument("--nonar-mask", type=Path, required=True)
    parser.add_argument("--ar-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=Path, help="Aligned preprocessing rows; saves numeric probe_targets.npz")
    args = parser.parse_args()
    nonar = np.load(args.nonar_mask, allow_pickle=False)
    ar = np.load(args.ar_mask, allow_pickle=False)
    if nonar.shape != ar.shape:
        raise ValueError(f"Mask shape mismatch: {nonar.shape} != {ar.shape}")
    common = nonar.astype(bool) & ar.astype(bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, common, allow_pickle=False)
    if args.rows:
        rows = read_rows(args.rows)
        if len(rows) != len(common):
            raise ValueError("Rows and eligibility masks are not aligned")
        selected = rows.iloc[np.flatnonzero(common)]
        np.savez(
            args.output.parent / "probe_targets.npz",
            rowids=selected["rowid"].to_numpy(dtype=np.int64),
            y=selected["predicted_log_solubility"].to_numpy(dtype=np.float64),
            confounds=selected[
                ["selfies_len_tokens", "branch_token_count", "ring_token_count", "token_entropy"]
            ].to_numpy(dtype=np.float64),
            groups=selected["scaffold_hash"].to_numpy(dtype=np.uint64) if "scaffold_hash" in selected else np.arange(len(selected), dtype=np.uint64),
        )
    report = {
        "rows": len(common),
        "nonar_eligible": int(nonar.sum()),
        "ar_eligible": int(ar.sum()),
        "common_eligible": int(common.sum()),
        "common_eligible_percent": percentage(int(common.sum()), len(common)),
        "common_failure_percent": percentage(int((~common).sum()), len(common)),
    }
    json_dump(report, args.output.with_suffix(".json"))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
