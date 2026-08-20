#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lignin_pipeline.common import json_dump


MODELS = ("linear_attention", "simple_attention", "autoregressive")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate lignin pipeline reports into JSON and Markdown.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    preprocess = load(root / "preprocess" / "preprocess_report.json")
    nonar = load(root / "tokenization" / "nonar" / "tokenization_report.json")
    ar = load(root / "tokenization" / "ar" / "tokenization_report.json")
    common = load(root / "common_eligible.json")
    reconstruction = {
        model: load(root / "models" / model / "reconstruction_report.json") for model in MODELS
    }
    probes = load(root / "probes" / "probe_report.json")
    summary = {
        "sample": {
            "rows": preprocess["requested_rows"],
            "seed": preprocess["seed"],
        },
        "preprocessing": preprocess,
        "tokenization": {"nonar": nonar, "ar": ar, "common": common},
        "reconstruction": reconstruction,
        "probes": probes,
    }
    json_dump(summary, root / "summary.json")

    lines = [
        "# Lignin solubility local evaluation",
        "",
        f"Random sample: {preprocess['requested_rows']:,} rows; seed {preprocess['seed']}.",
        "",
        "## Preprocessing and eligibility failures",
        "",
        "| Stage | Failure | Count | Percent of sampled rows |",
        "|---|---|---:|---:|",
    ]
    for status, values in preprocess["status"].items():
        if status != "ok":
            lines.append(f"| Canonicalization | {status} | {values['count']} | {values['percent']:.4f}% |")
    preprocess_ok = preprocess["status"].get("ok", {}).get("count", 0)
    preprocess_fail = preprocess["requested_rows"] - preprocess_ok
    lines.append(f"| Canonicalization | any failure | {preprocess_fail} | {100 * preprocess_fail / preprocess['requested_rows']:.4f}% |")
    lines.extend(
        [
            f"| Tokenization (non-AR) | any OOV | {nonar['oov_failure_count']} | {nonar['oov_failure_percent']:.4f}% |",
            f"| Tokenization (non-AR) | too long (>77 incl. SOS/EOS) | {nonar['length_failure_count']} | {nonar['length_failure_percent']:.4f}% |",
            f"| Tokenization (non-AR) | OOV and too long | {nonar['oov_and_length_failure_count']} | {nonar['oov_and_length_failure_percent']:.4f}% |",
            f"| Common model cohort | any eligibility failure | {preprocess['requested_rows'] - common['common_eligible']} | {common['common_failure_percent']:.4f}% |",
            "",
            "Eligibility flags overlap; the common-cohort row is the union failure rate.",
            "",
            "## Reconstruction failures on the common eligible cohort",
            "",
            "| Model | N | Strict sequence failure | Length failure | EOS failure | SELFIES failure | SMILES failure | Canonical mismatch |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        row = reconstruction[model]
        lines.append(
            f"| {model} | {row['rows']} | {row['strict_reconstruction_failure_percent']:.4f}% | "
            f"{row['length_failure_percent']:.4f}% | {row['eos_failure_percent']:.4f}% | "
            f"{row['selfies_decode_failure_percent']:.4f}% | {row['smiles_validation_failure_percent']:.4f}% | "
            f"{row['canonical_mismatch_percent']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Ridge probe results",
            "",
            f"Confound-only test R²: {probes['confound_probe']['r2_test']:.6f}.",
            "",
            "| Model | Raw test R² | Residual test R² | Combined test R² |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in probes["models"]:
        lines.append(
            f"| {row['model']} | {row['raw_r2_test']:.6f} | {row['residual_r2_test']:.6f} | {row['combined_r2_test']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Probe split sizes: {probes['split_sizes']}. Only common-eligible molecules enter these fits.",
            "",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
