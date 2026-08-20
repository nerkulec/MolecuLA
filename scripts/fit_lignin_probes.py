#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

from lignin_pipeline.common import json_dump, write_rows


ALPHAS = np.logspace(-3, 3, 13)
MODEL_NAMES = ("linear_attention", "simple_attention", "autoregressive")
CONFOUND_NAMES = ("selfies_len_tokens", "branch_token_count", "ring_token_count", "token_entropy")


@dataclass
class LatentShard:
    rowids: np.ndarray
    latents: np.ndarray


def parse_assignment(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected MODEL=OUTPUT_DIR") from exc
    if name not in MODEL_NAMES:
        raise argparse.ArgumentTypeError(f"Unknown model {name}")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit raw and confound-residualized streaming Ridge probes.")
    parser.add_argument("--targets", type=Path, nargs="+", help="Ordered probe_targets.npz shards")
    parser.add_argument(
        "--model-shard",
        action="append",
        type=parse_assignment,
        help="Repeat MODEL=OUTPUT_DIR in the same shard order as --targets",
    )
    parser.add_argument("--shard-root", type=Path, help="Discover shard_*/probe_targets.npz and model outputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=("random", "scaffold"), default="random")
    return parser.parse_args()


def make_split(n: int, seed: int, mode: str, groups: np.ndarray | None = None) -> np.ndarray:
    positions = np.arange(n)
    if mode == "random":
        train, temp = train_test_split(positions, test_size=0.2, random_state=seed, shuffle=True)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed, shuffle=True)
    else:
        if groups is None or len(np.unique(groups)) < 3:
            raise ValueError("Scaffold splitting requires at least three groups")
        first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_rel, temp_rel = next(first.split(positions, groups=groups))
        train = positions[train_rel]
        temp = positions[temp_rel]
        second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_rel, test_rel = next(second.split(temp, groups=groups[temp]))
        val = temp[val_rel]
        test = temp[test_rel]
    split = np.full(n, -1, dtype=np.int8)
    split[train] = 0
    split[val] = 1
    split[test] = 2
    if np.any(split < 0):
        raise RuntimeError("Incomplete split assignment")
    return split


def select_ridge(X: np.ndarray, y: np.ndarray, split: np.ndarray) -> tuple[Ridge, float, dict]:
    train = split == 0
    val = split == 1
    test = split == 2
    x_mean = X[train].mean(axis=0)
    x_std = X[train].std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean = float(y[train].mean())
    y_std = float(y[train].std())
    if y_std < 1e-12:
        raise ValueError("Target has effectively zero training variance")
    Xs = (X - x_mean) / x_std
    ys = (y - y_mean) / y_std
    candidates = []
    for alpha in ALPHAS:
        model = Ridge(alpha=float(alpha), fit_intercept=False)
        model.fit(Xs[train], ys[train])
        candidates.append((r2_score(ys[val], model.predict(Xs[val])), float(alpha), model))
    val_r2, alpha, model = max(candidates, key=lambda item: item[0])
    scores = {
        "alpha": alpha,
        "r2_train": float(r2_score(ys[train], model.predict(Xs[train]))),
        "r2_val": float(val_r2),
        "r2_test": float(r2_score(ys[test], model.predict(Xs[test]))),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return model, alpha, scores


def latent_stats(shards: list[LatentShard], split_parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_x = None
    sum_x2 = None
    for shard, shard_split in zip(shards, split_parts, strict=True):
        mask = shard_split == 0
        X = np.asarray(shard.latents[mask], dtype=np.float64)
        if sum_x is None:
            sum_x = np.zeros(X.shape[1], dtype=np.float64)
            sum_x2 = np.zeros(X.shape[1], dtype=np.float64)
        sum_x += X.sum(axis=0)
        sum_x2 += np.square(X).sum(axis=0)
        total += len(X)
    mean = sum_x / total
    variance = np.maximum(sum_x2 / total - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-12] = 1.0
    return mean, std


def fit_streaming_probe(
    shards: list[LatentShard],
    target_parts: list[np.ndarray],
    split_parts: list[np.ndarray],
) -> tuple[dict, dict]:
    x_mean, x_std = latent_stats(shards, split_parts)
    train_y = np.concatenate([y[s == 0] for y, s in zip(target_parts, split_parts, strict=True)])
    y_mean = float(train_y.mean())
    y_std = float(train_y.std())
    if y_std < 1e-12:
        raise ValueError("Target has effectively zero training variance")
    dim = len(x_mean)
    xtx = np.zeros((dim, dim), dtype=np.float64)
    xty = np.zeros(dim, dtype=np.float64)
    for shard, y, split in zip(shards, target_parts, split_parts, strict=True):
        mask = split == 0
        Xs = (np.asarray(shard.latents[mask], dtype=np.float64) - x_mean) / x_std
        ys = (y[mask] - y_mean) / y_std
        xtx += Xs.T @ Xs
        xty += Xs.T @ ys
    coefficients = {float(alpha): np.linalg.solve(xtx + float(alpha) * np.eye(dim), xty) for alpha in ALPHAS}

    def score(coef: np.ndarray, split_id: int) -> float:
        truth: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        for shard, y, split in zip(shards, target_parts, split_parts, strict=True):
            mask = split == split_id
            Xs = (np.asarray(shard.latents[mask], dtype=np.float64) - x_mean) / x_std
            truth.append((y[mask] - y_mean) / y_std)
            predicted.append(Xs @ coef)
        return float(r2_score(np.concatenate(truth), np.concatenate(predicted)))

    validation = [(score(coef, 1), alpha, coef) for alpha, coef in coefficients.items()]
    val_r2, selected_alpha, selected_coef = max(validation, key=lambda item: item[0])
    metrics = {
        "alpha": selected_alpha,
        "r2_train": score(selected_coef, 0),
        "r2_val": val_r2,
        "r2_test": score(selected_coef, 2),
    }
    parameters = {
        "coef": selected_coef,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": np.asarray(y_mean),
        "y_std": np.asarray(y_std),
    }
    return metrics, parameters


def predict_streaming(shards: list[LatentShard], parameters: dict) -> np.ndarray:
    predictions = []
    for shard in shards:
        Xs = (np.asarray(shard.latents, dtype=np.float64) - parameters["x_mean"]) / parameters["x_std"]
        standardized = Xs @ parameters["coef"]
        predictions.append(float(parameters["y_mean"]) + float(parameters["y_std"]) * standardized)
    return np.concatenate(predictions)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = list(args.model_shard or [])
    if args.shard_root:
        if args.targets or assignments:
            raise ValueError("Use --shard-root or explicit --targets/--model-shard, not both")
        shard_dirs = sorted(path for path in args.shard_root.glob("shard_*") if path.is_dir())
        if not shard_dirs:
            raise ValueError(f"No shard_* directories under {args.shard_root}")
        args.targets = [path / "probe_targets.npz" for path in shard_dirs]
        assignments = [(model, path / "models" / model) for model in MODEL_NAMES for path in shard_dirs]
    if not args.targets or not assignments:
        raise ValueError("Provide --shard-root or explicit --targets and --model-shard values")
    target_payloads = [np.load(path, allow_pickle=False) for path in args.targets]
    rowid_parts = [payload["rowids"].astype(np.int64) for payload in target_payloads]
    y_parts = [payload["y"].astype(np.float64) for payload in target_payloads]
    confound_parts = [payload["confounds"].astype(np.float64) for payload in target_payloads]
    group_parts = [
        payload["groups"].astype(np.uint64) if "groups" in payload.files else payload["rowids"].astype(np.uint64)
        for payload in target_payloads
    ]
    rowids = np.concatenate(rowid_parts)
    y = np.concatenate(y_parts)
    confounds = np.concatenate(confound_parts)
    groups = np.concatenate(group_parts)
    split = make_split(len(rowids), args.seed, args.split_mode, groups)
    offsets = np.cumsum([0, *[len(part) for part in rowid_parts]])
    split_parts = [split[offsets[i] : offsets[i + 1]] for i in range(len(rowid_parts))]

    confound_model, _, confound_scores = select_ridge(confounds, y, split)
    Cs = (confounds - confound_scores["x_mean"]) / confound_scores["x_std"]
    y_hat = confound_scores["y_mean"] + confound_scores["y_std"] * confound_model.predict(Cs)
    residual = y - y_hat
    residual_parts = [residual[offsets[i] : offsets[i + 1]] for i in range(len(rowid_parts))]

    by_model: dict[str, list[Path]] = {name: [] for name in MODEL_NAMES}
    for name, path in assignments:
        by_model[name].append(path)
    if any(len(by_model[name]) != len(args.targets) for name in MODEL_NAMES):
        raise ValueError("Provide one ordered --model-shard per model for every target shard")

    metric_rows: list[dict] = []
    for model_name in MODEL_NAMES:
        shards: list[LatentShard] = []
        for expected_rowids, directory in zip(rowid_parts, by_model[model_name], strict=True):
            actual_rowids = np.load(directory / "rowids.npy", allow_pickle=False)
            if not np.array_equal(expected_rowids, actual_rowids):
                raise ValueError(f"Row alignment mismatch for {model_name}: {directory}")
            shards.append(LatentShard(actual_rowids, np.load(directory / "latents.npy", mmap_mode="r", allow_pickle=False)))
        raw_metrics, raw_parameters = fit_streaming_probe(shards, y_parts, split_parts)
        residual_metrics, residual_parameters = fit_streaming_probe(shards, residual_parts, split_parts)
        combined_prediction = y_hat + predict_streaming(shards, residual_parameters)
        combined_r2_test = float(r2_score(y[split == 2], combined_prediction[split == 2]))
        np.savez(args.output_dir / f"{model_name}_raw_probe.npz", **raw_parameters)
        np.savez(args.output_dir / f"{model_name}_residual_probe.npz", **residual_parameters)
        metric_rows.append(
            {
                "model": model_name,
                "n": len(rowids),
                "raw_alpha": raw_metrics["alpha"],
                "raw_r2_train": raw_metrics["r2_train"],
                "raw_r2_val": raw_metrics["r2_val"],
                "raw_r2_test": raw_metrics["r2_test"],
                "residual_alpha": residual_metrics["alpha"],
                "residual_r2_train": residual_metrics["r2_train"],
                "residual_r2_val": residual_metrics["r2_val"],
                "residual_r2_test": residual_metrics["r2_test"],
                "combined_r2_test": combined_r2_test,
                "delta_test": residual_metrics["r2_test"] - raw_metrics["r2_test"],
            }
        )

    metrics = pd.DataFrame(metric_rows)
    write_rows(metrics, args.output_dir / "probe_metrics.csv")
    split_labels = np.asarray(["train", "val", "test"])[split]
    write_rows(
        pd.DataFrame(
            {
                "rowid": rowids,
                "split": split_labels,
                "predicted_log_solubility": y,
                "confound_prediction": y_hat,
                "residual_log_solubility": residual,
            }
        ),
        args.output_dir / "probe_rows.csv.gz",
    )
    serializable_confound_scores = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in confound_scores.items()
    }
    report = {
        "rows": len(rowids),
        "seed": args.seed,
        "split_mode": args.split_mode,
        "scaffold_group_count": int(len(np.unique(groups))),
        "split_sizes": {
            "train": int((split == 0).sum()),
            "val": int((split == 1).sum()),
            "test": int((split == 2).sum()),
        },
        "confounds": list(CONFOUND_NAMES),
        "confound_probe": serializable_confound_scores,
        "models": metric_rows,
    }
    json_dump(report, args.output_dir / "probe_report.json")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
