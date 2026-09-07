#!/usr/bin/env python
"""Validate and combine solubility analyses for compact latent models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = {
    64: "634itik5",
    128: "3bv5lnom",
    256: "17nxcyq1",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-64", type=Path, required=True)
    parser.add_argument("--analysis-128", type=Path, required=True)
    parser.add_argument("--analysis-256", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def frame_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analyses = {
        64: args.analysis_64.resolve(),
        128: args.analysis_128.resolve(),
        256: args.analysis_256.resolve(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    database_hash = None
    solubility_frames = []
    probe_frames = []
    pca_frames = []
    model_rows = []
    for latent_size, directory in analyses.items():
        report_path = directory / "analysis_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing completed analysis: {report_path}")
        report = json.loads(report_path.read_text())
        if int(report["latent_dim"]) != latent_size:
            raise ValueError(
                f"{directory} has latent dimension {report['latent_dim']}, "
                f"expected {latent_size}"
            )
        if database_hash is None:
            database_hash = report["database_sha256"]
        elif report["database_sha256"] != database_hash:
            raise ValueError("The analyses were produced from different databases")

        manifest = report["latent_manifest"]
        checkpoint = Path(manifest["checkpoint"])
        if MODELS[latent_size] not in str(checkpoint):
            raise ValueError(
                f"Unexpected checkpoint for latent {latent_size}: {checkpoint}"
            )
        record = manifest.get("checkpoint_record") or {}
        validation = record.get("val") or {}
        metadata = {
            "latent_size": latent_size,
            "run_id": MODELS[latent_size],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "checkpoint_epoch": manifest["checkpoint_epoch"],
        }
        model_rows.append(
            {
                **metadata,
                "val_reconstruction_loss": validation.get("reconstruction_loss"),
                "val_kl_loss": validation.get("kl_loss"),
                "val_token_accuracy": validation.get("teacher_forced_token_accuracy"),
                "val_exact_accuracy": validation.get("exact_accuracy"),
                "val_selection_loss": record.get("val_selection_loss"),
            }
        )
        for filename, destination in (
            ("solubility_r2_summary.csv", solubility_frames),
            ("probe_metrics.csv", probe_frames),
            ("pca_explained_variance.csv", pca_frames),
        ):
            frame = pd.read_csv(directory / filename)
            frame.insert(0, "run_id", MODELS[latent_size])
            frame.insert(0, "latent_size", latent_size)
            destination.append(frame)

    models = pd.DataFrame(model_rows).sort_values("latent_size")
    solubility = pd.concat(solubility_frames, ignore_index=True)
    probes = pd.concat(probe_frames, ignore_index=True)
    pca = pd.concat(pca_frames, ignore_index=True)
    full_test = solubility[
        (solubility["representation"] == "full")
        & (solubility["evaluation_split"] == "test")
    ].sort_values(["split_mode", "latent_size"])

    models.to_csv(output_dir / "model_checkpoints.csv", index=False)
    solubility.to_csv(output_dir / "solubility_r2_comparison.csv", index=False)
    probes.to_csv(output_dir / "probe_metrics_comparison.csv", index=False)
    pca.to_csv(output_dir / "pca_variance_comparison.csv", index=False)
    full_test.to_csv(output_dir / "full_latent_test_r2.csv", index=False)

    winners = {}
    for split_mode, rows in full_test.groupby("split_mode"):
        winners[split_mode] = {
            "best_raw_r2": frame_records(rows.nlargest(1, "raw_r2"))[0],
            "best_combined_r2": frame_records(rows.nlargest(1, "combined_r2"))[0],
        }
    comparison = {
        "format_version": 1,
        "database_sha256": database_hash,
        "models": frame_records(models),
        "full_latent_test_results": frame_records(full_test),
        "winners": winners,
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
