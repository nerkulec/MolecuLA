#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import platform
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from lignin_pipeline.common import confounds_from_selfies, json_dump, status_summary, write_rows


COLUMNS = [
    "rowid",
    "raw_smiles",
    "predicted_log_solubility",
    "aleatoric_std",
    "epistemic_std",
    "total_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize lignin-solubility SMILES through the paper conversion path.")
    parser.add_argument("--db", type=Path, default=Path("data/lignin_solubility.db"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-rowid", type=int)
    parser.add_argument("--end-rowid", type=int)
    parser.add_argument("--fetch-size", type=int, default=10_000)
    parser.add_argument("--no-progress", action="store_true", help="Disable the per-process tqdm progress bar.")
    return parser.parse_args()


def fetch_rows(connection: sqlite3.Connection, args: argparse.Namespace) -> list[tuple]:
    base = "select rowid,smiles,predicted_log_solubility,aleatoric_std,epistemic_std,total_std from functionalized_lignins"
    if args.sample_size is not None:
        if args.start_rowid is not None or args.end_rowid is not None:
            raise ValueError("Use either --sample-size or a rowid range, not both")
        lo, hi, count = connection.execute(
            "select min(rowid),max(rowid),count(*) from functionalized_lignins"
        ).fetchone()
        if count != hi - lo + 1:
            raise ValueError("Random rowid sampling requires dense rowids")
        if args.sample_size > count:
            raise ValueError("Sample size exceeds table size")
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(np.arange(lo, hi + 1), size=args.sample_size, replace=False))
        rows: list[tuple] = []
        for start in range(0, len(selected), 900):
            block = selected[start : start + 900].tolist()
            placeholders = ",".join("?" for _ in block)
            rows.extend(connection.execute(f"{base} where rowid in ({placeholders})", block).fetchall())
        rows.sort(key=lambda row: row[0])
        return rows
    clauses: list[str] = []
    params: list[int] = []
    if args.start_rowid is not None:
        clauses.append("rowid >= ?")
        params.append(args.start_rowid)
    if args.end_rowid is not None:
        clauses.append("rowid <= ?")
        params.append(args.end_rowid)
    query = base + ((" where " + " and ".join(clauses)) if clauses else "") + " order by rowid"
    return connection.execute(query, params).fetchall()


def process_row(row: tuple) -> dict:
    rowid, raw_smiles, prediction, aleatoric, epistemic, total = row
    result = {
        "rowid": int(rowid),
        "raw_smiles": raw_smiles,
        "rdkit_smiles": "",
        "murcko_scaffold": "",
        "scaffold_hash": 0,
        "selfies_first": "",
        "roundtrip_smiles": "",
        "selfies_final": "",
        "predicted_log_solubility": float(prediction),
        "aleatoric_std": float(aleatoric),
        "epistemic_std": float(epistemic),
        "total_std": float(total),
        "selfies_len_tokens": np.nan,
        "branch_token_count": np.nan,
        "ring_token_count": np.nan,
        "token_entropy": np.nan,
        "preprocess_status": "ok",
        "failure_detail": "",
    }
    try:
        mol = Chem.MolFromSmiles(str(raw_smiles), sanitize=True)
        if mol is None:
            raise ValueError("RDKit returned None")
        result["rdkit_smiles"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        result["murcko_scaffold"] = scaffold or result["rdkit_smiles"]
        result["scaffold_hash"] = int.from_bytes(
            hashlib.blake2b(result["murcko_scaffold"].encode("utf-8"), digest_size=8).digest(), "little"
        )
    except Exception as exc:
        result["preprocess_status"] = "rdkit_smiles_failed"
        result["failure_detail"] = str(exc)
        return result
    try:
        result["selfies_first"] = sf.encoder(result["rdkit_smiles"], strict=True)
    except Exception as exc:
        result["preprocess_status"] = "first_selfies_encode_failed"
        result["failure_detail"] = str(exc)
        return result
    try:
        result["roundtrip_smiles"] = sf.decoder(result["selfies_first"])
    except Exception as exc:
        result["preprocess_status"] = "selfies_decode_failed"
        result["failure_detail"] = str(exc)
        return result
    try:
        result["selfies_final"] = sf.encoder(result["roundtrip_smiles"], strict=True)
        length, branches, rings, entropy = confounds_from_selfies(result["selfies_final"])
        result.update(
            selfies_len_tokens=length,
            branch_token_count=branches,
            ring_token_count=rings,
            token_entropy=entropy,
        )
    except Exception as exc:
        result["preprocess_status"] = "second_selfies_encode_failed"
        result["failure_detail"] = str(exc)
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    started = time.time()
    uri = f"file:{args.db.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = fetch_rows(connection, args)
    processed = [
        process_row(row)
        for row in tqdm(rows, desc="canonicalize", unit="mol", disable=args.no_progress)
    ]
    frame = pd.DataFrame(processed)
    rows_path = args.output_dir / "rows.csv.gz"
    write_rows(frame, rows_path)
    report = {
        "stage": "preprocess",
        "database": str(args.db),
        "requested_rows": len(rows),
        "seed": args.seed if args.sample_size is not None else None,
        "sample_size": args.sample_size,
        "start_rowid": args.start_rowid,
        "end_rowid": args.end_rowid,
        "status": status_summary(frame["preprocess_status"]),
        "elapsed_seconds": time.time() - started,
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "selfies": sf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "output": str(rows_path),
    }
    json_dump(report, args.output_dir / "preprocess_report.json")
    print(pd.Series(frame["preprocess_status"]).value_counts(dropna=False).to_string())
    print(f"Wrote {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
