from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf

from .common import MAX_LEN, json_dump, load_tokenizer, percentage, read_rows, write_rows


def tokenize_file(
    kind: str,
    rows_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
) -> dict:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(rows_path)
    tokenizer = load_tokenizer(kind, tokenizer_path)
    tok2id = tokenizer["tok2id"]
    known_tokens = set(tok2id)
    tokens = np.zeros((len(rows), MAX_LEN), dtype=np.uint8)
    eligible = np.zeros(len(rows), dtype=bool)
    statuses: list[str] = []
    total_lengths = np.full(len(rows), -1, dtype=np.int16)
    has_oov = np.zeros(len(rows), dtype=bool)
    too_long = np.zeros(len(rows), dtype=bool)
    oov_text = [""] * len(rows)
    oov_counter: Counter[str] = Counter()

    for index, row in rows.iterrows():
        if row["preprocess_status"] != "ok":
            statuses.append("preprocess_failed")
            continue
        sequence = list(sf.split_selfies(str(row["selfies_final"])))
        total_lengths[index] = len(sequence) + 2
        missing = sorted(set(sequence) - known_tokens)
        has_oov[index] = bool(missing)
        too_long[index] = len(sequence) + 2 > MAX_LEN
        oov_text[index] = " ".join(missing)
        if missing:
            statuses.append("out_of_vocabulary")
            oov_counter.update(missing)
            continue
        if len(sequence) + 2 > MAX_LEN:
            statuses.append("too_long")
            continue
        encoded = [tok2id["<SOS>"], *(tok2id[token] for token in sequence), tok2id["<EOS>"]]
        tokens[index, : len(encoded)] = encoded
        eligible[index] = True
        statuses.append("ok")

    token_rows = pd.DataFrame(
        {
            "rowid": rows["rowid"].astype(np.int64),
            "tokenization_status": statuses,
            "total_tokens": total_lengths,
            "has_oov": has_oov.astype(np.int8),
            "too_long": too_long.astype(np.int8),
            "oov_tokens": oov_text,
        }
    )
    np.save(output_dir / "tokens.npy", tokens, allow_pickle=False)
    np.save(output_dir / "eligible_mask.npy", eligible, allow_pickle=False)
    write_rows(token_rows, output_dir / "tokenization_rows.csv.gz")
    json_dump(tokenizer, output_dir / "tokenizer.json")
    counts = Counter(statuses)
    report = {
        "stage": "tokenize",
        "kind": kind,
        "rows": len(rows),
        "eligible": int(eligible.sum()),
        "eligible_percent": percentage(int(eligible.sum()), len(rows)),
        "failure_percent": percentage(int((~eligible).sum()), len(rows)),
        "oov_failure_count": int(has_oov.sum()),
        "oov_failure_percent": percentage(int(has_oov.sum()), len(rows)),
        "length_failure_count": int(too_long.sum()),
        "length_failure_percent": percentage(int(too_long.sum()), len(rows)),
        "oov_and_length_failure_count": int((has_oov & too_long).sum()),
        "oov_and_length_failure_percent": percentage(int((has_oov & too_long).sum()), len(rows)),
        "status": {
            key: {"count": value, "percent": percentage(value, len(rows))}
            for key, value in sorted(counts.items())
        },
        "oov_tokens": dict(oov_counter.most_common()),
        "vocab_size": len(tokenizer["vocab"]),
        "max_len": MAX_LEN,
        "elapsed_seconds": time.time() - started,
    }
    json_dump(report, output_dir / "tokenization_report.json")
    print(json.dumps(report, indent=2))
    return report
