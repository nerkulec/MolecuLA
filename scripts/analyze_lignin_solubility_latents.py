#!/usr/bin/env python
"""Run streaming raw, residualized, confound, and PCA Ridge analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from encode_lignin_training_shard import splitmix64


ALPHAS = np.logspace(-3, 3, 13)
SPLIT_NAMES = ("train", "val", "test")
CONFOUND_NAMES = (
    "selfies_len_tokens",
    "branch_token_count",
    "ring_token_count",
    "token_entropy",
)


@dataclass
class RawMoments:
    counts: np.ndarray
    sum_x: np.ndarray
    gram_x: np.ndarray
    sum_t: np.ndarray
    sum_t2: np.ndarray
    cross_xt: np.ndarray
    target_names: tuple[str, ...]


@dataclass
class FeatureMoments:
    name: str
    center: np.ndarray
    scale: np.ndarray
    projection: np.ndarray | None
    counts: np.ndarray
    sum_z: np.ndarray
    gram_z: np.ndarray
    sum_t: np.ndarray
    sum_t2: np.ndarray
    cross_zt: np.ndarray
    target_names: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-rows", type=Path, nargs="+", required=True)
    parser.add_argument("--encoded-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-modes", choices=("rowid", "scaffold"), nargs="+", required=True)
    parser.add_argument("--pca-components", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prediction-sample-size", type=int, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_panel(paths: list[Path], expected_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.empty(expected_rows, dtype=np.float64)
    confounds = np.empty((expected_rows, len(CONFOUND_NAMES)), dtype=np.float64)
    groups = np.empty(expected_rows, dtype=np.uint64)
    cursor = 0
    columns = [
        "rowid",
        "preprocess_status",
        "predicted_log_solubility",
        *CONFOUND_NAMES,
        "scaffold_hash",
    ]
    for path in tqdm(sorted(paths), desc="load preprocessing panel", unit="shard"):
        frame = pd.read_csv(path, compression="infer", usecols=columns, keep_default_na=False)
        if not frame["preprocess_status"].eq("ok").all():
            failures = int((~frame["preprocess_status"].eq("ok")).sum())
            raise ValueError(f"{path} contains {failures} preprocessing failures")
        stop = cursor + len(frame)
        if stop > expected_rows:
            raise ValueError("Preprocessing rows exceed latent row count")
        expected = np.arange(cursor + 1, stop + 1, dtype=np.int64)
        if not np.array_equal(frame["rowid"].to_numpy(dtype=np.int64), expected):
            raise ValueError(
                f"Preprocessing rows are not aligned to SQLite rowid order: {path}"
            )
        y[cursor:stop] = frame["predicted_log_solubility"].to_numpy(dtype=np.float64)
        confounds[cursor:stop] = frame[list(CONFOUND_NAMES)].to_numpy(dtype=np.float64)
        groups[cursor:stop] = frame["scaffold_hash"].to_numpy(dtype=np.uint64)
        cursor = stop
    if cursor != expected_rows:
        raise ValueError(f"Preprocessing shards contain {cursor} rows; expected {expected_rows}")
    if not np.isfinite(y).all() or not np.isfinite(confounds).all():
        raise ValueError("Target or confound arrays contain non-finite values")
    return y, confounds, groups


def load_rowid_split(paths: list[Path], expected_rows: int, expected_seed: int) -> np.ndarray:
    split_parts = []
    cursor = 0
    for path in sorted(paths):
        shard_manifest = json.loads((path / "manifest.json").read_text())
        if shard_manifest.get("split_by") != "rowid":
            raise ValueError(f"Encoded shard does not contain a rowid split: {path}")
        if shard_manifest.get("split_seed") != expected_seed:
            raise ValueError(
                f"Encoded split seed mismatch in {path}: "
                f"{shard_manifest.get('split_seed')} != {expected_seed}"
            )
        rowids = np.load(path / "rowids.npy", mmap_mode="r", allow_pickle=False)
        expected = np.arange(cursor + 1, cursor + len(rowids) + 1, dtype=np.int64)
        if not np.array_equal(rowids, expected):
            raise ValueError(f"Encoded row alignment mismatch: {path}")
        split_parts.append(np.asarray(np.load(path / "splits.npy", mmap_mode="r"), dtype=np.int8))
        cursor += len(rowids)
    if cursor != expected_rows:
        raise ValueError(f"Encoded shards contain {cursor} rows; expected {expected_rows}")
    return np.concatenate(split_parts)


def scaffold_split(groups: np.ndarray, seed: int) -> np.ndarray:
    hashed = splitmix64(groups.astype(np.uint64) ^ np.uint64(seed)) % np.uint64(100)
    return np.where(hashed < 80, 0, np.where(hashed < 90, 1, 2)).astype(np.int8)


def r2_from_sufficient_statistics(
    coefficient: np.ndarray,
    gram: np.ndarray,
    cross: np.ndarray,
    sum_y: float,
    sum_y2: float,
    n: int,
) -> tuple[float, float]:
    sse = float(sum_y2 - 2.0 * coefficient @ cross + coefficient @ gram @ coefficient)
    sse = max(sse, 0.0)
    tss = float(sum_y2 - sum_y * sum_y / n)
    if tss <= 0.0:
        raise ValueError("Cannot compute R2 for a target with zero variance")
    return 1.0 - sse / tss, sse


def dense_ridge(
    X: np.ndarray, y: np.ndarray, split: np.ndarray
) -> tuple[dict, dict, np.ndarray]:
    train = split == 0
    x_mean = X[train].mean(axis=0)
    x_std = X[train].std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean = float(y[train].mean())
    y_std = float(y[train].std())
    if y_std < 1e-12:
        raise ValueError("Target has zero training variance")
    split_stats = []
    for split_id in range(3):
        mask = split == split_id
        xs = (X[mask] - x_mean) / x_std
        ys = (y[mask] - y_mean) / y_std
        split_stats.append(
            {
                "n": int(mask.sum()),
                "gram": xs.T @ xs,
                "cross": xs.T @ ys,
                "sum_y": float(ys.sum()),
                "sum_y2": float(ys @ ys),
            }
        )
    train_stats = split_stats[0]
    eigenvalues, eigenvectors = np.linalg.eigh(train_stats["gram"])
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_cross = eigenvectors.T @ train_stats["cross"]
    candidates = []
    for alpha in ALPHAS:
        coefficient = eigenvectors @ (projected_cross / (eigenvalues + alpha))
        val_r2, _ = r2_from_sufficient_statistics(coefficient, **split_stats[1])
        candidates.append((val_r2, float(alpha), coefficient))
    _, alpha, coefficient = max(candidates, key=lambda item: item[0])
    scores = {}
    for split_id, split_name in enumerate(SPLIT_NAMES):
        r2, sse = r2_from_sufficient_statistics(coefficient, **split_stats[split_id])
        scores[f"r2_{split_name}"] = r2
        scores[f"sse_{split_name}"] = sse * y_std * y_std
    prediction = y_mean + y_std * (((X - x_mean) / x_std) @ coefficient)
    parameters = {
        "coefficient_standardized": coefficient,
        "coefficient_original": y_std * coefficient / x_std,
        "intercept_original": np.asarray(
            y_mean - x_mean @ (y_std * coefficient / x_std)
        ),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": np.asarray(y_mean),
        "y_std": np.asarray(y_std),
        "alpha": np.asarray(alpha),
    }
    return {"alpha": alpha, **scores}, parameters, prediction


def initialize_gpu_moments(
    modes: list[str], latent_dim: int, target_count: int, device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        mode: {
            "counts": torch.zeros(3, dtype=torch.int64, device=device),
            "sum_x": torch.zeros((3, latent_dim), dtype=torch.float64, device=device),
            "gram_x": torch.zeros((3, latent_dim, latent_dim), dtype=torch.float64, device=device),
            "sum_t": torch.zeros((3, target_count), dtype=torch.float64, device=device),
            "sum_t2": torch.zeros((3, target_count), dtype=torch.float64, device=device),
            "cross_xt": torch.zeros(
                (3, latent_dim, target_count), dtype=torch.float64, device=device
            ),
        }
        for mode in modes
    }


def accumulate_latent_moments(
    latents: np.ndarray,
    targets: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    target_names: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> dict[str, RawMoments]:
    latent_dim = latents.shape[1]
    modes = list(splits)
    if device.type == "cuda":
        accumulators = initialize_gpu_moments(modes, latent_dim, len(target_names), device)
        # PCA and Ridge depend on accurate covariance matrices. Do not trade FP32
        # mantissa precision for TensorFloat-32 throughput on Ampere/Hopper GPUs.
        torch.set_float32_matmul_precision("highest")
    else:
        accumulators = {
            mode: {
                "counts": np.zeros(3, dtype=np.int64),
                "sum_x": np.zeros((3, latent_dim), dtype=np.float64),
                "gram_x": np.zeros((3, latent_dim, latent_dim), dtype=np.float64),
                "sum_t": np.zeros((3, len(target_names)), dtype=np.float64),
                "sum_t2": np.zeros((3, len(target_names)), dtype=np.float64),
                "cross_xt": np.zeros((3, latent_dim, len(target_names)), dtype=np.float64),
            }
            for mode in modes
        }
    batches = math.ceil(len(latents) / batch_size)
    for start in tqdm(
        range(0, len(latents), batch_size),
        total=batches,
        desc="latent sufficient statistics",
        unit="batch",
        dynamic_ncols=True,
    ):
        stop = min(start + batch_size, len(latents))
        x_np = np.asarray(latents[start:stop], dtype=np.float32)
        if not np.isfinite(x_np).all():
            raise ValueError(f"Non-finite latents in rows {start + 1}..{stop}")
        if device.type == "cuda":
            x = torch.as_tensor(x_np, device=device)
            for mode in modes:
                t = torch.as_tensor(targets[mode][start:stop], dtype=torch.float32, device=device)
                split = torch.as_tensor(splits[mode][start:stop], device=device)
                acc = accumulators[mode]
                for split_id in range(3):
                    mask = split == split_id
                    if not bool(mask.any()):
                        continue
                    xs = x[mask]
                    ts = t[mask]
                    acc["counts"][split_id] += len(xs)
                    acc["sum_x"][split_id] += xs.sum(0, dtype=torch.float64)
                    acc["gram_x"][split_id] += (xs.T @ xs).double()
                    acc["sum_t"][split_id] += ts.sum(0, dtype=torch.float64)
                    acc["sum_t2"][split_id] += (ts * ts).sum(0, dtype=torch.float64)
                    acc["cross_xt"][split_id] += (xs.T @ ts).double()
        else:
            x = x_np.astype(np.float64)
            for mode in modes:
                t = targets[mode][start:stop].astype(np.float64, copy=False)
                split = splits[mode][start:stop]
                acc = accumulators[mode]
                for split_id in range(3):
                    mask = split == split_id
                    xs, ts = x[mask], t[mask]
                    if not len(xs):
                        continue
                    acc["counts"][split_id] += len(xs)
                    acc["sum_x"][split_id] += xs.sum(axis=0)
                    acc["gram_x"][split_id] += xs.T @ xs
                    acc["sum_t"][split_id] += ts.sum(axis=0)
                    acc["sum_t2"][split_id] += np.square(ts).sum(axis=0)
                    acc["cross_xt"][split_id] += xs.T @ ts
    result = {}
    for mode, acc in accumulators.items():
        converted = {
            key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
            for key, value in acc.items()
        }
        result[mode] = RawMoments(target_names=target_names, **converted)
    return result


def centered_gram(raw: RawMoments, split_id: int, center: np.ndarray) -> np.ndarray:
    n = int(raw.counts[split_id])
    sum_x = raw.sum_x[split_id]
    return (
        raw.gram_x[split_id]
        - np.outer(center, sum_x)
        - np.outer(sum_x, center)
        + n * np.outer(center, center)
    )


def feature_moments(
    raw: RawMoments,
    name: str,
    center: np.ndarray,
    scale: np.ndarray,
    projection: np.ndarray | None,
) -> FeatureMoments:
    dimensions = len(scale)
    sum_z = np.zeros((3, dimensions), dtype=np.float64)
    gram_z = np.zeros((3, dimensions, dimensions), dtype=np.float64)
    cross_zt = np.zeros((3, dimensions, len(raw.target_names)), dtype=np.float64)
    for split_id in range(3):
        n = int(raw.counts[split_id])
        centered_sum = raw.sum_x[split_id] - n * center
        centered_cross = raw.cross_xt[split_id] - np.outer(center, raw.sum_t[split_id])
        gram = centered_gram(raw, split_id, center)
        if projection is not None:
            centered_sum = projection.T @ centered_sum
            centered_cross = projection.T @ centered_cross
            gram = projection.T @ gram @ projection
        sum_z[split_id] = centered_sum / scale
        cross_zt[split_id] = centered_cross / scale[:, None]
        gram_z[split_id] = gram / np.outer(scale, scale)
    return FeatureMoments(
        name=name,
        center=center,
        scale=scale,
        projection=projection,
        counts=raw.counts,
        sum_z=sum_z,
        gram_z=gram_z,
        sum_t=raw.sum_t,
        sum_t2=raw.sum_t2,
        cross_zt=cross_zt,
        target_names=raw.target_names,
    )


def feature_eigendecomposition(feature: FeatureMoments) -> tuple[np.ndarray, np.ndarray]:
    gram = (feature.gram_z[0] + feature.gram_z[0].T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    return np.maximum(eigenvalues, 0.0), eigenvectors


def fit_feature_target(
    feature: FeatureMoments,
    target_name: str,
    eigendecomposition: tuple[np.ndarray, np.ndarray],
) -> tuple[dict, dict]:
    target_index = feature.target_names.index(target_name)
    train_n = int(feature.counts[0])
    y_mean = float(feature.sum_t[0, target_index] / train_n)
    y_variance = (
        feature.sum_t2[0, target_index] / train_n - y_mean * y_mean
    )
    y_std = math.sqrt(max(float(y_variance), 0.0))
    if y_std < 1e-12:
        raise ValueError(f"{target_name} has zero training variance")
    standardized = []
    for split_id in range(3):
        n = int(feature.counts[split_id])
        sum_y_raw = float(feature.sum_t[split_id, target_index])
        sum_y = (sum_y_raw - n * y_mean) / y_std
        sum_y2 = (
            feature.sum_t2[split_id, target_index]
            - 2 * y_mean * sum_y_raw
            + n * y_mean * y_mean
        ) / (y_std * y_std)
        cross = (
            feature.cross_zt[split_id, :, target_index]
            - y_mean * feature.sum_z[split_id]
        ) / y_std
        standardized.append(
            {
                "n": n,
                "gram": feature.gram_z[split_id],
                "cross": cross,
                "sum_y": sum_y,
                "sum_y2": float(sum_y2),
            }
        )
    eigenvalues, eigenvectors = eigendecomposition
    projected_cross = eigenvectors.T @ standardized[0]["cross"]
    candidates = []
    for alpha in ALPHAS:
        coefficient = eigenvectors @ (projected_cross / (eigenvalues + alpha))
        val_r2, _ = r2_from_sufficient_statistics(coefficient, **standardized[1])
        candidates.append((val_r2, float(alpha), coefficient))
    _, alpha, coefficient = max(candidates, key=lambda item: item[0])
    metrics = {"alpha": alpha}
    sse_original = {}
    for split_id, split_name in enumerate(SPLIT_NAMES):
        r2, sse = r2_from_sufficient_statistics(coefficient, **standardized[split_id])
        metrics[f"r2_{split_name}"] = r2
        sse_original[split_name] = sse * y_std * y_std
    if feature.projection is None:
        latent_direction = y_std * coefficient / feature.scale
    else:
        latent_direction = y_std * (
            feature.projection @ (coefficient / feature.scale)
        )
    parameters = {
        "coefficient_standardized": coefficient,
        "latent_direction": latent_direction,
        "latent_center": feature.center,
        "feature_scale": feature.scale,
        "projection": (
            feature.projection if feature.projection is not None else np.empty((0, 0))
        ),
        "y_mean": np.asarray(y_mean),
        "y_std": np.asarray(y_std),
        "alpha": np.asarray(alpha),
        "sse_train": np.asarray(sse_original["train"]),
        "sse_val": np.asarray(sse_original["val"]),
        "sse_test": np.asarray(sse_original["test"]),
    }
    return metrics, parameters


def predict_feature(x: np.ndarray, feature: FeatureMoments, parameters: dict) -> np.ndarray:
    centered = x.astype(np.float64) - feature.center
    if feature.projection is not None:
        centered = centered @ feature.projection
    standardized = centered / feature.scale
    return float(parameters["y_mean"]) + float(parameters["y_std"]) * (
        standardized @ parameters["coefficient_standardized"]
    )


def target_tss(y: np.ndarray, split: np.ndarray, split_id: int) -> float:
    values = y[split == split_id]
    return float(np.square(values - values.mean()).sum())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    args.database = args.database.resolve()
    args.latents_dir = args.latents_dir.resolve()
    args.preprocessed_rows = sorted(path.resolve() for path in args.preprocessed_rows)
    args.encoded_shards = sorted(path.resolve() for path in args.encoded_shards)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size < 1 or args.prediction_sample_size < 1:
        raise ValueError("Batch and prediction sample sizes must be positive")
    pca_sizes = sorted(set(args.pca_components))
    if not pca_sizes or pca_sizes[0] < 1:
        raise ValueError("PCA component counts must be positive")
    if len(set(args.split_modes)) != len(args.split_modes):
        raise ValueError("Duplicate --split-modes values")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires a visible GPU")

    manifest = json.loads((args.latents_dir / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Latent export manifest is not complete")
    database_hash = sha256_file(args.database)
    if manifest.get("database_sha256") != database_hash:
        raise ValueError("Latent export was produced from a different database")
    latents = np.load(args.latents_dir / "latents.npy", mmap_mode="r", allow_pickle=False)
    rowids = np.load(args.latents_dir / "rowids.npy", mmap_mode="r", allow_pickle=False)
    rows, latent_dim = latents.shape
    if manifest.get("shape") != [rows, latent_dim]:
        raise ValueError("Latent matrix shape does not match manifest")
    if not np.array_equal(rowids, np.arange(1, rows + 1, dtype=np.int64)):
        raise ValueError("Latent rowids are not dense database order")
    if pca_sizes[-1] > latent_dim:
        raise ValueError("Requested PCA dimension exceeds latent dimension")
    if args.prediction_sample_size > rows:
        raise ValueError("Prediction sample exceeds dataset size")

    y, confounds, groups = load_panel(args.preprocessed_rows, rows)
    rowid_split = load_rowid_split(args.encoded_shards, rows, args.seed)
    splits = {}
    if "rowid" in args.split_modes:
        splits["rowid"] = rowid_split
    if "scaffold" in args.split_modes:
        splits["scaffold"] = scaffold_split(groups, args.seed)
    for mode, split in splits.items():
        if set(np.unique(split)) != {0, 1, 2}:
            raise ValueError(f"{mode} split does not contain train, val, and test")
        np.save(args.output_dir / f"split_{mode}.npy", split, allow_pickle=False)

    confound_results = {}
    residuals = {}
    confound_predictions = {}
    for mode, split in splits.items():
        metrics, parameters, prediction = dense_ridge(confounds, y, split)
        residual = y - prediction
        confound_results[mode] = {"metrics": metrics, "parameters": parameters}
        residuals[mode] = residual
        confound_predictions[mode] = prediction
        np.savez(args.output_dir / f"confound_probe_{mode}.npz", **parameters)

    target_names = ("log_solubility", "residual_log_solubility", *CONFOUND_NAMES)
    targets = {
        mode: np.column_stack([y, residuals[mode], confounds])
        for mode in splits
    }
    raw_moments = accumulate_latent_moments(
        latents, targets, splits, target_names, args.batch_size, device
    )

    metric_rows = []
    pca_rows = []
    feature_spaces: dict[str, dict[str, FeatureMoments]] = {}
    fitted_probes: dict[str, dict[str, dict[str, dict]]] = {}
    for mode, raw in tqdm(
        raw_moments.items(),
        total=len(raw_moments),
        desc="fit PCA and probes",
        unit="split",
        dynamic_ncols=True,
    ):
        train_n = int(raw.counts[0])
        center = raw.sum_x[0] / train_n
        train_centered_gram = centered_gram(raw, 0, center)
        train_centered_gram = (train_centered_gram + train_centered_gram.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(train_centered_gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        total_variance = float(eigenvalues.sum())
        max_components = pca_sizes[-1]
        pca_projection = eigenvectors[:, :max_components]
        pca_scale = np.sqrt(np.maximum(eigenvalues[:max_components] / train_n, 1e-24))
        np.savez(
            args.output_dir / f"pca_{mode}.npz",
            mean=center,
            components=pca_projection.T,
            explained_variance=eigenvalues[:max_components] / train_n,
            explained_variance_ratio=eigenvalues[:max_components] / total_variance,
            singular_values=np.sqrt(eigenvalues[:max_components]),
            train_rows=np.asarray(train_n),
        )
        cumulative = np.cumsum(eigenvalues) / total_variance
        for component in range(1, max_components + 1):
            pca_rows.append(
                {
                    "split_mode": mode,
                    "component": component,
                    "explained_variance": eigenvalues[component - 1] / train_n,
                    "explained_variance_ratio": eigenvalues[component - 1] / total_variance,
                    "cumulative_explained_variance_ratio": cumulative[component - 1],
                }
            )

        full_scale = np.sqrt(
            np.maximum(np.diag(train_centered_gram) / train_n, 1e-24)
        )
        spaces = {
            "full": feature_moments(raw, "full", center, full_scale, None)
        }
        pca_max = feature_moments(
            raw, f"pca_{max_components}", center, pca_scale, pca_projection
        )
        for size in pca_sizes:
            spaces[f"pca_{size}"] = FeatureMoments(
                name=f"pca_{size}",
                center=center,
                scale=pca_max.scale[:size],
                projection=pca_max.projection[:, :size],
                counts=pca_max.counts,
                sum_z=pca_max.sum_z[:, :size],
                gram_z=pca_max.gram_z[:, :size, :size],
                sum_t=pca_max.sum_t,
                sum_t2=pca_max.sum_t2,
                cross_zt=pca_max.cross_zt[:, :size],
                target_names=pca_max.target_names,
            )
        feature_spaces[mode] = spaces
        fitted_probes[mode] = {}
        for space_name, space in spaces.items():
            fitted_probes[mode][space_name] = {}
            eigendecomposition = feature_eigendecomposition(space)
            for target_name in target_names:
                metrics, parameters = fit_feature_target(
                    space, target_name, eigendecomposition
                )
                fitted_probes[mode][space_name][target_name] = parameters
                safe_target = target_name.replace("log_solubility", "logS")
                np.savez(
                    args.output_dir / f"probe_{mode}_{space_name}_{safe_target}.npz",
                    **parameters,
                )
                row = {
                    "split_mode": mode,
                    "representation": space_name,
                    "target": target_name,
                    **metrics,
                }
                if target_name == "residual_log_solubility":
                    for split_id, split_name in enumerate(SPLIT_NAMES):
                        original_tss = target_tss(y, splits[mode], split_id)
                        residual_sse = float(parameters[f"sse_{split_name}"])
                        row[f"combined_r2_{split_name}"] = 1.0 - residual_sse / original_tss
                metric_rows.append(row)

    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(args.output_dir / "probe_metrics.csv", index=False)
    solubility_rows = []
    for mode in splits:
        for representation in feature_spaces[mode]:
            raw_row = metric_frame[
                (metric_frame.split_mode == mode)
                & (metric_frame.representation == representation)
                & (metric_frame.target == "log_solubility")
            ].iloc[0]
            residual_row = metric_frame[
                (metric_frame.split_mode == mode)
                & (metric_frame.representation == representation)
                & (metric_frame.target == "residual_log_solubility")
            ].iloc[0]
            for split_name in SPLIT_NAMES:
                solubility_rows.append(
                    {
                        "split_mode": mode,
                        "representation": representation,
                        "evaluation_split": split_name,
                        "raw_r2": raw_row[f"r2_{split_name}"],
                        "residual_r2": residual_row[f"r2_{split_name}"],
                        "combined_r2": residual_row[f"combined_r2_{split_name}"],
                        "confound_only_r2": confound_results[mode]["metrics"][
                            f"r2_{split_name}"
                        ],
                        "raw_alpha": raw_row["alpha"],
                        "residual_alpha": residual_row["alpha"],
                        "confound_alpha": confound_results[mode]["metrics"]["alpha"],
                    }
                )
    pd.DataFrame(solubility_rows).to_csv(
        args.output_dir / "solubility_r2_summary.csv", index=False
    )
    pd.DataFrame(pca_rows).to_csv(
        args.output_dir / "pca_explained_variance.csv", index=False
    )

    rng = np.random.default_rng(args.seed)
    sample_indexes = np.sort(
        rng.choice(rows, size=args.prediction_sample_size, replace=False)
    )
    sample_x = np.asarray(latents[sample_indexes], dtype=np.float64)
    sample_confounds = confounds[sample_indexes]
    for mode, split in splits.items():
        sample = pd.DataFrame(
            {
                "rowid": sample_indexes + 1,
                "split": np.asarray(SPLIT_NAMES)[split[sample_indexes]],
                "predicted_log_solubility": y[sample_indexes],
                "confound_prediction": confound_predictions[mode][sample_indexes],
                "residual_log_solubility": residuals[mode][sample_indexes],
                **{
                    name: sample_confounds[:, index]
                    for index, name in enumerate(CONFOUND_NAMES)
                },
            }
        )
        for space_name, space in feature_spaces[mode].items():
            raw_parameters = fitted_probes[mode][space_name]["log_solubility"]
            residual_parameters = fitted_probes[mode][space_name][
                "residual_log_solubility"
            ]
            raw_prediction = predict_feature(sample_x, space, raw_parameters)
            residual_prediction = predict_feature(sample_x, space, residual_parameters)
            sample[f"{space_name}_raw_prediction"] = raw_prediction
            sample[f"{space_name}_residual_prediction"] = residual_prediction
            sample[f"{space_name}_combined_prediction"] = (
                sample["confound_prediction"].to_numpy() + residual_prediction
            )
        sample.to_csv(
            args.output_dir / f"prediction_sample_{mode}.csv.gz",
            index=False,
            compression="gzip",
        )
        pca = feature_spaces[mode][f"pca_{pca_sizes[-1]}"]
        pca_scores = ((sample_x - pca.center) @ pca.projection) / pca.scale
        np.savez(
            args.output_dir / f"pca_sample_{mode}.npz",
            rowids=sample_indexes.astype(np.int64) + 1,
            split=split[sample_indexes],
            log_solubility=y[sample_indexes],
            residual_log_solubility=residuals[mode][sample_indexes],
            confounds=sample_confounds,
            standardized_scores=pca_scores[:, : min(8, pca_scores.shape[1])].astype(
                np.float32
            ),
            confound_names=np.asarray(CONFOUND_NAMES),
        )
        correlation_columns = [
            "predicted_log_solubility",
            "residual_log_solubility",
            *CONFOUND_NAMES,
        ]
        correlation_frame = sample[correlation_columns].copy()
        for index in range(min(8, pca_scores.shape[1])):
            correlation_frame[f"PC{index + 1}"] = pca_scores[:, index]
        correlation_frame.corr(method="pearson").to_csv(
            args.output_dir / f"sample_pearson_correlations_{mode}.csv"
        )
        correlation_frame.corr(method="spearman").to_csv(
            args.output_dir / f"sample_spearman_correlations_{mode}.csv"
        )

    split_summaries = {
        mode: {
            split_name: {
                "rows": int((split == split_id).sum()),
                "target_mean": float(y[split == split_id].mean()),
                "target_std": float(y[split == split_id].std()),
            }
            for split_id, split_name in enumerate(SPLIT_NAMES)
        }
        for mode, split in splits.items()
    }
    confound_report = {
        mode: result["metrics"] for mode, result in confound_results.items()
    }
    report = {
        "format_version": 1,
        "rows": rows,
        "latent_dim": latent_dim,
        "database": str(args.database),
        "database_sha256": database_hash,
        "latent_manifest": manifest,
        "split_modes": list(splits),
        "split_summaries": split_summaries,
        "scaffold_group_count": int(len(np.unique(groups))),
        "seed": args.seed,
        "ridge_alphas": ALPHAS.tolist(),
        "pca_components": pca_sizes,
        "pca_fit": "training rows only; centered unscaled latent covariance",
        "confounds": list(CONFOUND_NAMES),
        "confound_only_probes": confound_report,
        "probe_metrics": metric_rows,
        "prediction_sample_rows": len(sample_indexes),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    json_dump(report, args.output_dir / "analysis_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
