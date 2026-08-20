from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import selfies as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_LEN = 77
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


def json_dump(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def percentage(count: int, denominator: int) -> float:
    return 100.0 * count / denominator if denominator else float("nan")


def status_summary(values: Iterable[str]) -> dict[str, dict[str, float | int]]:
    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    return {
        key: {"count": count, "percent": percentage(count, total)}
        for key, count in sorted(counts.items())
    }


def shannon_entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = np.asarray(list(Counter(tokens).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def confounds_from_selfies(selfies_value: str) -> tuple[int, int, int, float]:
    tokens = list(sf.split_selfies(selfies_value))
    return (
        len(tokens),
        sum("Branch" in token for token in tokens),
        sum("Ring" in token for token in tokens),
        shannon_entropy(tokens),
    )


def load_tokenizer(kind: str, tokenizer_path: str | Path) -> dict:
    payload = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    nonar_vocab = list(payload["vocab"])
    if nonar_vocab[:3] != ["<PAD>", "<SOS>", "<EOS>"]:
        raise ValueError("Unexpected non-autoregressive special-token ordering")
    if kind == "nonar":
        vocab = nonar_vocab
    elif kind == "ar":
        # The AR training notebook inserted MASK before the same sorted chemical tokens.
        vocab = ["<PAD>", "<SOS>", "<EOS>", "MASK", *nonar_vocab[3:]]
    else:
        raise ValueError(f"Unknown tokenizer kind: {kind}")
    expected = 110 if kind == "nonar" else 111
    if len(vocab) != expected or len(vocab) != len(set(vocab)):
        raise ValueError(f"Unexpected {kind} vocabulary size/duplicates: {len(vocab)}")
    return {
        "kind": kind,
        "vocab": vocab,
        "tok2id": {token: index for index, token in enumerate(vocab)},
        "id2tok": {str(index): token for index, token in enumerate(vocab)},
        "max_len": MAX_LEN,
        "vocab_size": len(vocab),
    }


def read_rows(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", keep_default_na=False)


def write_rows(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def ids_until_pad(row: np.ndarray) -> np.ndarray:
    pad = np.flatnonzero(row == PAD_ID)
    stop = int(pad[0]) if len(pad) else len(row)
    return row[:stop]


def ids_to_selfies(ids: Iterable[int], id2tok: dict[str, str]) -> tuple[str | None, str | None]:
    tokens: list[str] = []
    for value in ids:
        token = id2tok.get(str(int(value)))
        if token is None:
            return None, "unknown_token_id"
        if token == "MASK":
            return None, "mask_token_emitted"
        if token in {"<PAD>", "<SOS>", "<EOS>"}:
            continue
        tokens.append(token)
    return "".join(tokens), None


def finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None
