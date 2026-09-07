"""Tests for combining compact-latent solubility results."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_compact_lignin_solubility import MODELS, main as compare_main


class CompareCompactLigninSolubilityTest(unittest.TestCase):
    def test_combines_and_selects_full_latent_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analyses = {}
            for latent_size, run_id in MODELS.items():
                directory = root / f"latent_{latent_size}_{run_id}"
                directory.mkdir()
                analyses[latent_size] = directory
                checkpoint_record = {
                    "epoch": latent_size,
                    "val": {
                        "reconstruction_loss": 1.0 / latent_size,
                        "kl_loss": 0.5,
                        "teacher_forced_token_accuracy": 0.99,
                        "exact_accuracy": 0.9,
                    },
                    "val_selection_loss": 0.02,
                }
                report = {
                    "latent_dim": latent_size,
                    "database_sha256": "same-database",
                    "latent_manifest": {
                        "checkpoint": f"/runs/{run_id}/best.pt",
                        "checkpoint_sha256": f"hash-{latent_size}",
                        "checkpoint_epoch": latent_size,
                        "checkpoint_record": checkpoint_record,
                    },
                }
                (directory / "analysis_report.json").write_text(json.dumps(report))
                pd.DataFrame(
                    [
                        {
                            "split_mode": mode,
                            "representation": representation,
                            "evaluation_split": "test",
                            "raw_r2": latent_size / 1000 + (representation == "full") * 0.1,
                            "residual_r2": 0.2,
                            "combined_r2": latent_size / 1000 + 0.2,
                            "confound_only_r2": 0.1,
                            "raw_alpha": 1.0,
                            "residual_alpha": 1.0,
                            "confound_alpha": 1.0,
                        }
                        for mode in ("rowid", "scaffold")
                        for representation in ("full", "pca_8")
                    ]
                ).to_csv(directory / "solubility_r2_summary.csv", index=False)
                pd.DataFrame(
                    [
                        {
                            "split_mode": "rowid",
                            "representation": "full",
                            "target": "log_solubility",
                            "r2_test": latent_size / 1000,
                        }
                    ]
                ).to_csv(directory / "probe_metrics.csv", index=False)
                pd.DataFrame(
                    [
                        {
                            "split_mode": "rowid",
                            "component": 1,
                            "explained_variance_ratio": 0.5,
                        }
                    ]
                ).to_csv(directory / "pca_explained_variance.csv", index=False)

            output = root / "comparison"
            result = compare_main(
                [
                    "--analysis-64",
                    str(analyses[64]),
                    "--analysis-128",
                    str(analyses[128]),
                    "--analysis-256",
                    str(analyses[256]),
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            models = pd.read_csv(output / "model_checkpoints.csv")
            self.assertEqual(models.latent_size.tolist(), [64, 128, 256])
            full = pd.read_csv(output / "full_latent_test_r2.csv")
            self.assertEqual(len(full), 6)
            report = json.loads((output / "comparison_report.json").read_text())
            self.assertEqual(report["winners"]["rowid"]["best_raw_r2"]["latent_size"], 256)
            self.assertEqual(
                report["winners"]["scaffold"]["best_combined_r2"]["latent_size"],
                256,
            )


if __name__ == "__main__":
    unittest.main()
