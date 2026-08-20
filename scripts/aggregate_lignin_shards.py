#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lignin_pipeline.common import json_dump, percentage


MODELS = ("linear_attention", "simple_attention", "autoregressive")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate preprocessing/reconstruction failure rates over cluster shards.")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shards = sorted(path for path in args.shard_root.glob("shard_*") if path.is_dir())
    if not shards:
        raise ValueError("No shard directories found")
    sampled = 0
    preprocess_counts: Counter[str] = Counter()
    token_totals = Counter()
    common_eligible = 0
    reconstruction = {model: Counter() for model in MODELS}
    for shard in shards:
        pre = load(shard / "preprocess" / "preprocess_report.json")
        sampled += pre["requested_rows"]
        preprocess_counts.update({key: value["count"] for key, value in pre["status"].items()})
        tok = load(shard / "tokenization" / "nonar" / "tokenization_report.json")
        token_totals.update(
            rows=tok["rows"], eligible=tok["eligible"], oov=tok["oov_failure_count"],
            too_long=tok["length_failure_count"], both=tok["oov_and_length_failure_count"]
        )
        common_eligible += load(shard / "common_eligible.json")["common_eligible"]
        for model in MODELS:
            rec = load(shard / "models" / model / "reconstruction_report.json")
            reconstruction[model].update(
                rows=rec["rows"], token_correct=rec["token_correct"], token_total=rec["token_total"],
                strict=rec["strict_reconstruction_failure_count"], length=rec["length_failure_count"],
                eos=rec["eos_failure_count"], selfies=rec["selfies_decode_failure_count"],
                smiles=rec["smiles_validation_failure_count"], canonical=rec["canonical_mismatch_count"]
            )
    payload = {
        "shards": len(shards),
        "sampled_rows": sampled,
        "preprocessing_status": {
            key: {"count": count, "percent": percentage(count, sampled)} for key, count in sorted(preprocess_counts.items())
        },
        "tokenization": {
            "eligible": int(token_totals["eligible"]),
            "eligible_percent": percentage(token_totals["eligible"], sampled),
            "oov_failure_percent": percentage(token_totals["oov"], sampled),
            "length_failure_percent": percentage(token_totals["too_long"], sampled),
            "oov_and_length_failure_percent": percentage(token_totals["both"], sampled),
            "common_eligible": common_eligible,
            "common_eligible_percent": percentage(common_eligible, sampled),
        },
        "reconstruction": {},
    }
    for model, counts in reconstruction.items():
        n = counts["rows"]
        payload["reconstruction"][model] = {
            "rows": n,
            "token_accuracy_percent": percentage(counts["token_correct"], counts["token_total"]),
            "strict_reconstruction_failure_percent": percentage(counts["strict"], n),
            "length_failure_percent": percentage(counts["length"], n),
            "eos_failure_percent": percentage(counts["eos"], n),
            "selfies_decode_failure_percent": percentage(counts["selfies"], n),
            "smiles_validation_failure_percent": percentage(counts["smiles"], n),
            "canonical_mismatch_percent": percentage(counts["canonical"], n),
        }
    if args.probe_report:
        payload["probes"] = load(args.probe_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(payload, args.output_dir / "full_summary.json")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
