#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], expected: Path, resume: bool) -> None:
    if resume and expected.exists():
        print(f"[resume] keeping {expected}")
        return
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the seeded local lignin evaluation end to end.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lignin_solubility/local_10k"))
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nonar-batch-size", type=int, default=256)
    parser.add_argument("--ar-batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    root = args.output_dir
    rows = root / "preprocess" / "rows.csv.gz"
    nonar = root / "tokenization" / "nonar"
    ar = root / "tokenization" / "ar"
    common = root / "common_eligible.npy"

    run(
        [py, "scripts/preprocess_lignin_solubility.py", "--sample-size", str(args.sample_size), "--seed", str(args.seed), "--output-dir", str(root / "preprocess")],
        rows,
        args.resume,
    )
    run([py, "scripts/tokenize_lignin_nonar.py", "--rows", str(rows), "--output-dir", str(nonar)], nonar / "tokens.npy", args.resume)
    run([py, "scripts/tokenize_lignin_ar.py", "--rows", str(rows), "--output-dir", str(ar)], ar / "tokens.npy", args.resume)
    run(
        [py, "scripts/combine_lignin_eligibility.py", "--nonar-mask", str(nonar / "eligible_mask.npy"), "--ar-mask", str(ar / "eligible_mask.npy"), "--rows", str(rows), "--output", str(common)],
        root / "probe_targets.npz",
        args.resume,
    )
    for model in ("linear_attention", "simple_attention", "autoregressive"):
        encoding = ar if model == "autoregressive" else nonar
        batch = args.ar_batch_size if model == "autoregressive" else args.nonar_batch_size
        output = root / "models" / model
        run(
            [py, "scripts/encode_decode_lignin.py", "--model", model, "--rows", str(rows), "--tokens", str(encoding / "tokens.npy"), "--eligible-mask", str(common), "--tokenizer", str(encoding / "tokenizer.json"), "--output-dir", str(output), "--device", args.device, "--batch-size", str(batch)],
            output / "reconstruction_report.json",
            args.resume,
        )
    probe_command = [py, "scripts/fit_lignin_probes.py", "--targets", str(root / "probe_targets.npz")]
    for model in ("linear_attention", "simple_attention", "autoregressive"):
        probe_command.extend(["--model-shard", f"{model}={root / 'models' / model}"])
    probe_command.extend(["--output-dir", str(root / "probes"), "--seed", str(args.seed)])
    run(probe_command, root / "probes" / "probe_report.json", args.resume)
    run([py, "scripts/report_lignin_run.py", "--run-dir", str(root)], root / "summary.json", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
