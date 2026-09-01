"""End-to-end test for streaming solubility probes and PCA artifacts."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_lignin_solubility_latents import main as analyze_main
from encode_lignin_training_shard import splitmix64


class AnalyzeLigninSolubilityLatentsTest(unittest.TestCase):
    def test_analysis_outputs_aligned_probe_and_pca_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, dimensions = 240, 8
            rng = np.random.default_rng(7)
            latents = rng.normal(size=(rows, dimensions)).astype(np.float32)
            lengths = rng.integers(8, 80, size=rows).astype(np.float64)
            branches = rng.integers(0, 8, size=rows).astype(np.float64)
            rings = rng.integers(0, 5, size=rows).astype(np.float64)
            entropy = rng.normal(2.0, 0.2, size=rows)
            confounds = np.column_stack([lengths, branches, rings, entropy])
            y = 1.8 * latents[:, 0] - 0.7 * latents[:, 1] + 0.03 * lengths
            y += rng.normal(scale=0.03, size=rows)

            database = root / "lignin.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "create table functionalized_lignins ("
                    "smiles text primary key, predicted_log_solubility real not null)"
                )
                connection.executemany(
                    "insert into functionalized_lignins values (?, ?)",
                    [(f"C{index}", float(value)) for index, value in enumerate(y, 1)],
                )
            database_hash = hashlib.sha256(database.read_bytes()).hexdigest()

            latent_dir = root / "latents"
            latent_dir.mkdir()
            np.save(latent_dir / "latents.npy", latents, allow_pickle=False)
            np.save(
                latent_dir / "rowids.npy",
                np.arange(1, rows + 1, dtype=np.int64),
                allow_pickle=False,
            )
            (latent_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "complete": True,
                        "database_sha256": database_hash,
                        "shape": [rows, dimensions],
                    }
                )
            )

            panel_paths = []
            shard_paths = []
            rowids = np.arange(1, rows + 1, dtype=np.int64)
            groups = (rowids // 3).astype(np.uint64)
            splits = (
                splitmix64(rowids.astype(np.uint64) ^ np.uint64(42))
                % np.uint64(100)
            )
            splits = np.where(splits < 80, 0, np.where(splits < 90, 1, 2)).astype(
                np.uint8
            )
            for shard_index, (start, stop) in enumerate(((0, 120), (120, rows))):
                panel_dir = root / "preprocessed" / f"shard_{shard_index:03d}"
                panel_dir.mkdir(parents=True)
                panel_path = panel_dir / "rows.csv.gz"
                pd.DataFrame(
                    {
                        "rowid": rowids[start:stop],
                        "preprocess_status": "ok",
                        "predicted_log_solubility": y[start:stop],
                        "selfies_len_tokens": lengths[start:stop],
                        "branch_token_count": branches[start:stop],
                        "ring_token_count": rings[start:stop],
                        "token_entropy": entropy[start:stop],
                        "scaffold_hash": groups[start:stop],
                    }
                ).to_csv(panel_path, index=False, compression="gzip")
                panel_paths.append(panel_path)

                encoded = root / "encoded" / f"shard_{shard_index:03d}"
                encoded.mkdir(parents=True)
                np.save(encoded / "rowids.npy", rowids[start:stop], allow_pickle=False)
                np.save(encoded / "splits.npy", splits[start:stop], allow_pickle=False)
                (encoded / "manifest.json").write_text(
                    json.dumps({"split_by": "rowid", "split_seed": 42})
                )
                shard_paths.append(encoded)

            output = root / "analysis"
            arguments = [
                "--database", str(database),
                "--latents-dir", str(latent_dir),
                "--preprocessed-rows", *(str(path) for path in panel_paths),
                "--encoded-shards", *(str(path) for path in shard_paths),
                "--output-dir", str(output),
                "--split-modes", "rowid", "scaffold",
                "--pca-components", "2", "4", "6",
                "--batch-size", "47",
                "--device", "cpu",
                "--seed", "42",
                "--prediction-sample-size", "40",
            ]
            self.assertEqual(analyze_main(arguments), 0)

            metrics = pd.read_csv(output / "probe_metrics.csv")
            self.assertEqual(set(metrics.split_mode), {"rowid", "scaffold"})
            self.assertEqual(set(metrics.representation), {"full", "pca_2", "pca_4", "pca_6"})
            self.assertEqual(
                set(metrics.target),
                {
                    "log_solubility",
                    "residual_log_solubility",
                    "selfies_len_tokens",
                    "branch_token_count",
                    "ring_token_count",
                    "token_entropy",
                },
            )
            raw_full = metrics[
                (metrics.split_mode == "rowid")
                & (metrics.representation == "full")
                & (metrics.target == "log_solubility")
            ].iloc[0]
            self.assertGreater(raw_full.r2_test, 0.8)
            train, test = splits == 0, splits == 2
            x_mean, x_std = latents[train].mean(0), latents[train].std(0)
            y_mean, y_std = y[train].mean(), y[train].std()
            reference = Ridge(alpha=float(raw_full.alpha), fit_intercept=False).fit(
                (latents[train] - x_mean) / x_std,
                (y[train] - y_mean) / y_std,
            )
            reference_r2 = r2_score(
                (y[test] - y_mean) / y_std,
                reference.predict((latents[test] - x_mean) / x_std),
            )
            self.assertAlmostEqual(raw_full.r2_test, reference_r2, places=5)
            residual = metrics[metrics.target == "residual_log_solubility"]
            self.assertTrue(residual[["combined_r2_train", "combined_r2_val", "combined_r2_test"]].notna().all().all())
            summary = pd.read_csv(output / "solubility_r2_summary.csv")
            self.assertEqual(len(summary), 2 * 4 * 3)
            self.assertTrue(
                summary[
                    ["raw_r2", "residual_r2", "combined_r2", "confound_only_r2"]
                ].notna().all().all()
            )

            report = json.loads((output / "analysis_report.json").read_text())
            self.assertEqual(report["rows"], rows)
            self.assertEqual(report["pca_components"], [2, 4, 6])
            self.assertTrue((output / "pca_rowid.npz").is_file())
            self.assertTrue((output / "prediction_sample_scaffold.csv.gz").is_file())
            self.assertEqual(
                np.load(output / "pca_sample_rowid.npz")["standardized_scores"].shape,
                (40, 6),
            )


if __name__ == "__main__":
    unittest.main()
