"""Integration test for row-ordered, resumable lignin latent export."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_lignin_latents import main as export_main
from models.autoregressive_vae import VaeTransformer


class ExportLigninLatentsTest(unittest.TestCase):
    def test_export_is_in_dense_database_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "lignin.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "create table functionalized_lignins ("
                    "smiles text primary key, predicted_log_solubility real not null, "
                    "aleatoric_std real not null, epistemic_std real not null, "
                    "total_std real not null)"
                )
                connection.executemany(
                    "insert into functionalized_lignins values (?, ?, ?, ?, ?)",
                    [(f"C{i}", float(i), 0.1, 0.2, 0.3) for i in range(1, 6)],
                )

            sequences = [
                np.asarray(row, dtype=np.uint16)
                for row in ([1, 4, 2], [1, 5, 4, 2], [1, 6, 2], [1, 4, 6, 5, 2], [1, 5, 2])
            ]
            shard = root / "shard_000"
            shard.mkdir()
            offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(row) for row in sequences])
            np.save(shard / "tokens.npy", np.concatenate(sequences), allow_pickle=False)
            np.save(shard / "offsets.npy", offsets, allow_pickle=False)
            np.save(
                shard / "lengths.npy",
                np.asarray([len(row) for row in sequences], dtype=np.uint16),
                allow_pickle=False,
            )
            np.save(shard / "rowids.npy", np.arange(1, 6, dtype=np.int64), allow_pickle=False)
            np.save(shard / "groups.npy", np.arange(1, 6, dtype=np.uint64), allow_pickle=False)
            np.save(shard / "splits.npy", np.zeros(5, dtype=np.uint8), allow_pickle=False)
            tokenizer_hash = hashlib.sha256(b"test-tokenizer").hexdigest()
            (shard / "manifest.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "rows": 5,
                        "tokens": int(offsets[-1]),
                        "tokenizer_sha256": tokenizer_hash,
                    }
                )
            )

            torch.manual_seed(11)
            config = {
                "vocab_size": 7,
                "hidden_size": 16,
                "latent_size": 16,
                "max_len": 8,
                "attn_heads": 4,
                "num_slots": 4,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "padding_invariant_encoder": True,
            }
            model = VaeTransformer(**config).eval()
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "model_config": config,
                    "model": model.state_dict(),
                    "tokenizer_sha256": tokenizer_hash,
                    "epoch": 3,
                    "record": {
                        "epoch": 3,
                        "val": {
                            "reconstruction_loss": 0.012,
                            "exact_accuracy": 0.75,
                        },
                    },
                },
                checkpoint,
            )
            output = root / "latents"
            result = export_main(
                [
                    "--database",
                    str(database),
                    "--checkpoint",
                    str(checkpoint),
                    "--shards",
                    str(shard),
                    "--output-dir",
                    str(output),
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                    "--precision",
                    "fp32",
                    "--padding-mode",
                    "checkpoint-max",
                ]
            )
            self.assertEqual(result, 0)
            rowids = np.load(output / "rowids.npy", allow_pickle=False)
            latents = np.load(output / "latents.npy", allow_pickle=False)
            self.assertTrue(np.array_equal(rowids, np.arange(1, 6)))
            self.assertEqual(latents.shape, (5, 16))
            self.assertTrue(np.isfinite(latents).all())

            x = torch.zeros((5, config["max_len"]), dtype=torch.long)
            for index, sequence in enumerate(sequences):
                x[index, : len(sequence)] = torch.as_tensor(sequence.astype(np.int64))
            with torch.inference_mode():
                expected, _ = model.encode(x)
            np.testing.assert_allclose(latents, expected.numpy(), rtol=1e-6, atol=1e-6)

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["shape"], [5, 16])
            self.assertEqual(manifest["padding_width"], 8)
            self.assertTrue(manifest["complete"])
            self.assertEqual(
                manifest["checkpoint_record"]["val"]["reconstruction_loss"],
                0.012,
            )


if __name__ == "__main__":
    unittest.main()
