"""Auditable ChemSpacE protocols on the frozen MolecuLA latent interface.

Two experiments are deliberately separated:

1. ``method_faithful`` follows the released ZINC ChemSpacE workflow: score
   valid prior decodes, apply the mean +/- one-standard-deviation margin,
   select tail classes, fit a linear SVM, and traverse 200 prior seeds over
   21 points in [-1, 1].
2. ``paired_local`` holds MolecuLA's training rows, 50 encoded seeds, decoder,
   and local candidate grid fixed while comparing the continuous OLS direction
   with an adapted extreme-SVM direction.

The module never trains or changes the VAE.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.svm import SVC

import chemspace_comparison_utils as legacy


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "protocol_outputs"
METHOD_FAITHFUL_DIR = OUTPUT_ROOT / "method_faithful"
PAIRED_DIR = OUTPUT_ROOT / "paired_local"
PAIRED_DISPLACEMENT_DIR = OUTPUT_ROOT / "paired_displacement"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TRAINING_CANONICAL = (
    HERE.parent / "rebuttal_outputs" / "training_canonical_smiles.csv"
)

PROPERTIES = legacy.PROPERTIES
CONFOUNDS = legacy.CONFOUNDS
LOCAL_ALPHAS = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
PAIRED_DISPLACEMENT_ALPHAS = np.linspace(-150.0, 150.0, 21, dtype=np.float64)
OFFICIAL_TRAIN_CANDIDATES = 500
OFFICIAL_VALID_LABEL_POOL = 400
OFFICIAL_EVALUATION_SEEDS = 200
OFFICIAL_PRIOR_TRAIN_SEED = 123
OFFICIAL_PRIOR_EVALUATION_SEED = 124
OFFICIAL_CHOSEN_RATIO = 0.10
FIT_RATIO = 0.70
OFFICIAL_EPSILON = 0.05
OFFICIAL_GAMMA = 0.05
DECODER_MAX_SELFIES_TOKENS = 154
PAIRED_SEED = 42
PAIRED_EXTREMES_PER_CLASS = 200
PUBLISHED_SIMILARITY_CUTOFFS = (0.0, 0.2, 0.4, 0.6)

METHOD_CHEMSPACE_OFFICIAL = "ChemSpacE released-ZINC protocol"
METHOD_CHEMSPACE_PAIRED = "ChemSpacE adapted extreme-SVM"
METHOD_OLS_PAIRED = "MolecuLA continuous OLS"
SCALE_NATIVE = "native_l2"
SCALE_STANDARDIZED = "train_std_unit"


def _rau() -> Any:
    return legacy._rebuttal_utils()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def _atomic_npy(values: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)
    return path


def _atomic_npz(values: Mapping[str, np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)
    return path


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _unit(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Direction must have a finite nonzero norm")
    return vector / norm


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_values = np.asarray(list(x), dtype=np.float64)
    y_values = np.asarray(list(y), dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if (
        finite.sum() < 2
        or np.unique(x_values[finite]).size < 2
        or np.unique(y_values[finite]).size < 2
    ):
        return float("nan")
    result = spearmanr(x_values[finite], y_values[finite])
    return float(result.statistic)


def _safe_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x_values = np.asarray(list(x), dtype=np.float64)
    y_values = np.asarray(list(y), dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if finite.sum() < 2 or np.unique(x_values[finite]).size < 2:
        return float("nan")
    if float(np.ptp(y_values[finite])) == 0.0:
        return 0.0
    return float(np.polyfit(x_values[finite], y_values[finite], 1)[0])


@dataclass(frozen=True)
class BoundaryResult:
    direction: np.ndarray
    summary: Mapping[str, Any]
    selection: pd.DataFrame


@dataclass(frozen=True)
class DirectionJob:
    method: str
    property_name: str
    scale: str
    direction: np.ndarray


def fit_released_zinc_margin_boundary(
    latents: np.ndarray,
    scores: np.ndarray,
    *,
    property_name: str,
    seed: int = OFFICIAL_PRIOR_TRAIN_SEED,
    chosen_ratio: float = OFFICIAL_CHOSEN_RATIO,
    fit_ratio: float = FIT_RATIO,
) -> BoundaryResult:
    """Reproduce ``train_boundary_zinc.py`` + ``manipulator_margin.py``.

    The official release has no deterministic seed. This adaptation fixes the
    shuffle seed and records it; all other selection steps follow the released
    ZINC entry point.
    """

    z = np.asarray(latents, dtype=np.float32)
    y = np.asarray(scores, dtype=np.float64).reshape(-1)
    if z.ndim != 2 or len(z) != len(y):
        raise ValueError("latents and scores must be row-aligned")
    if not np.isfinite(y).all():
        raise ValueError(f"{property_name}: scores contain non-finite values")

    mean = float(y.mean())
    std = float(y.std(ddof=0))
    low_margin = y <= mean - std
    high_margin = y >= mean + std
    margin_mask = low_margin | high_margin
    margin_rows = np.flatnonzero(margin_mask)
    if len(margin_rows) < 4:
        raise ValueError(f"{property_name}: insufficient mean +/- std tails")
    order = np.lexsort((margin_rows, -y[margin_rows]))
    ordered_rows = margin_rows[order]
    chosen = min(int(len(ordered_rows) * chosen_ratio), len(ordered_rows) // 2)
    if chosen < 2:
        raise ValueError(f"{property_name}: official tail selection chose <2 rows")
    high_rows = ordered_rows[:chosen].copy()
    low_rows = ordered_rows[-chosen:].copy()
    if not high_margin[high_rows].all() or not low_margin[low_rows].all():
        raise AssertionError(f"{property_name}: tail classes crossed the margin")

    rng = np.random.RandomState(int(seed))
    rng.shuffle(high_rows)
    rng.shuffle(low_rows)
    n_fit = int(chosen * fit_ratio)
    if not 0 < n_fit < chosen:
        raise ValueError(f"{property_name}: invalid fit/validation split")
    fit_rows = np.concatenate((high_rows[:n_fit], low_rows[:n_fit]))
    fit_labels = np.concatenate(
        (np.ones(n_fit, dtype=np.int8), np.zeros(n_fit, dtype=np.int8))
    )
    validation_rows = np.concatenate((high_rows[n_fit:], low_rows[n_fit:]))
    validation_labels = np.concatenate(
        (
            np.ones(chosen - n_fit, dtype=np.int8),
            np.zeros(chosen - n_fit, dtype=np.int8),
        )
    )
    fit_start = time.perf_counter()
    classifier = SVC(kernel="linear", C=1.0)
    classifier.fit(z[fit_rows], fit_labels)
    fit_seconds = time.perf_counter() - fit_start
    coefficient = np.asarray(classifier.coef_[0], dtype=np.float64)
    direction = _unit(coefficient)
    if np.median(classifier.decision_function(z[high_rows])) <= np.median(
        classifier.decision_function(z[low_rows])
    ):
        raise AssertionError(f"{property_name}: SVM direction orientation failed")

    fit_set = set(int(value) for value in fit_rows)
    selection_rows = []
    for label_name, label, rows in (
        ("high", 1, high_rows),
        ("low", 0, low_rows),
    ):
        for shuffled_rank, row in enumerate(rows):
            selection_rows.append(
                {
                    "property": property_name,
                    "tail": label_name,
                    "class_label": label,
                    "prior_pool_row": int(row),
                    "property_value": float(y[row]),
                    "rank_after_seeded_shuffle": int(shuffled_rank),
                    "used_for_svm_fit": int(row) in fit_set,
                }
            )
    summary = {
        "property": property_name,
        "protocol": "released ZINC margin implementation",
        "oracle_label_pool": int(len(y)),
        "mean": mean,
        "population_std": std,
        "low_margin_rows": int(low_margin.sum()),
        "high_margin_rows": int(high_margin.sum()),
        "margin_union_rows": int(margin_mask.sum()),
        "chosen_ratio_after_margin": float(chosen_ratio),
        "selected_per_class": int(chosen),
        "svm_fit_rows": int(2 * n_fit),
        "svm_validation_rows": int(2 * (chosen - n_fit)),
        "split_ratio": float(fit_ratio),
        "shuffle_seed_added_for_reproducibility": int(seed),
        "svm_C": 1.0,
        "latent_standardization": False,
        "fit_accuracy": float(classifier.score(z[fit_rows], fit_labels)),
        "validation_accuracy": float(
            classifier.score(z[validation_rows], validation_labels)
        ),
        "support_vectors": int(classifier.n_support_.sum()),
        "svm_fit_seconds": fit_seconds,
        "direction_l2": float(np.linalg.norm(direction)),
        "direction_sha256": _array_hash(direction),
    }
    return BoundaryResult(
        direction=direction,
        summary=summary,
        selection=pd.DataFrame(selection_rows),
    )


def fit_paired_extreme_boundary(
    latents: np.ndarray,
    scores: np.ndarray,
    eligible_rows: np.ndarray,
    *,
    property_name: str,
    n_per_class: int = PAIRED_EXTREMES_PER_CLASS,
    seed: int = PAIRED_SEED,
) -> BoundaryResult:
    """Adapt ChemSpacE to the same MolecuLA training-label pool as OLS."""

    z = np.asarray(latents)
    y = np.asarray(scores, dtype=np.float64)
    rows = np.asarray(eligible_rows, dtype=np.int64)
    ranked_high = np.lexsort((rows, -y[rows]))
    ranked_low = np.lexsort((rows, y[rows]))
    high_rows = rows[ranked_high[:n_per_class]].copy()
    low_rows = rows[ranked_low[:n_per_class]].copy()
    if np.intersect1d(high_rows, low_rows).size:
        raise AssertionError(f"{property_name}: paired tail groups overlap")
    rng = np.random.default_rng(int(seed))
    high_rows = rng.permutation(high_rows)
    low_rows = rng.permutation(low_rows)
    n_fit = int(n_per_class * FIT_RATIO)
    fit_rows = np.concatenate((high_rows[:n_fit], low_rows[:n_fit]))
    fit_labels = np.concatenate(
        (np.ones(n_fit, dtype=np.int8), np.zeros(n_fit, dtype=np.int8))
    )
    validation_rows = np.concatenate((high_rows[n_fit:], low_rows[n_fit:]))
    validation_labels = np.concatenate(
        (
            np.ones(n_per_class - n_fit, dtype=np.int8),
            np.zeros(n_per_class - n_fit, dtype=np.int8),
        )
    )
    fit_start = time.perf_counter()
    classifier = SVC(kernel="linear", C=1.0)
    classifier.fit(np.asarray(z[fit_rows]), fit_labels)
    fit_seconds = time.perf_counter() - fit_start
    direction = _unit(np.asarray(classifier.coef_[0], dtype=np.float64))
    fit_set = set(int(value) for value in fit_rows)
    selection = pd.DataFrame(
        [
            {
                "property": property_name,
                "tail": tail,
                "class_label": label,
                "global_row_index": int(row),
                "property_value": float(y[row]),
                "used_for_svm_fit": int(row) in fit_set,
            }
            for tail, label, tail_rows in (
                ("high", 1, high_rows),
                ("low", 0, low_rows),
            )
            for row in tail_rows
        ]
    )
    summary = {
        "property": property_name,
        "protocol": "paired adaptation; fixed top/bottom 200",
        "labels_inspected_for_ranking": int(len(rows)),
        "selected_per_class": int(n_per_class),
        "selected_rows": int(2 * n_per_class),
        "svm_fit_rows": int(len(fit_rows)),
        "svm_validation_rows": int(len(validation_rows)),
        "split_ratio": FIT_RATIO,
        "shuffle_seed": int(seed),
        "svm_C": 1.0,
        "fit_accuracy": float(classifier.score(z[fit_rows], fit_labels)),
        "validation_accuracy": float(
            classifier.score(z[validation_rows], validation_labels)
        ),
        "support_vectors": int(classifier.n_support_.sum()),
        "svm_fit_seconds": fit_seconds,
        "direction_l2": float(np.linalg.norm(direction)),
        "direction_sha256": _array_hash(direction),
    }
    return BoundaryResult(direction, summary, selection)


def prepare_prior_training_pool(
    *,
    bundle: Any,
    loaded_model: Any,
    output_dir: Path = METHOD_FAITHFUL_DIR,
    batch_size: int = 64,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Decode 500 deterministic prior candidates and keep the first 400 valid."""

    selected_path = output_dir / "prior_training_pool.csv"
    attempt_path = output_dir / "prior_training_decode_attempts.csv"
    latent_path = output_dir / "prior_training_pool_latents.npy"
    if selected_path.exists() and attempt_path.exists() and latent_path.exists():
        selected = pd.read_csv(selected_path)
        attempts = pd.read_csv(attempt_path)
        selected_latents = np.load(latent_path)
        if (
            len(selected) == OFFICIAL_VALID_LABEL_POOL
            and selected_latents.shape == (OFFICIAL_VALID_LABEL_POOL, 256)
            and selected["is_rdkit_valid"].astype(bool).all()
        ):
            return selected_latents, selected, attempts
        raise AssertionError("Prior training cache exists but failed validation")

    rau = _rau()
    prior = rau.sample_prior_latents(
        OFFICIAL_TRAIN_CANDIDATES,
        256,
        seed=OFFICIAL_PRIOR_TRAIN_SEED,
        verify_known_torch_2_5_1=False,
    )
    decoded = rau.decode_latents_adaptive(
        loaded_model, prior, bundle.id2tok, batch_size=batch_size
    )
    attempts = pd.DataFrame(decoded)
    attempts.insert(0, "prior_candidate_id", np.arange(len(attempts)))
    valid = rau.valid_canonical_mask(attempts)
    valid_indices = np.flatnonzero(valid.to_numpy())
    if len(valid_indices) < OFFICIAL_VALID_LABEL_POOL:
        raise RuntimeError(
            f"Only {len(valid_indices)}/{len(attempts)} prior candidates were "
            "valid; increase OFFICIAL_TRAIN_CANDIDATES explicitly"
        )
    selected_indices = valid_indices[:OFFICIAL_VALID_LABEL_POOL]
    attempts["selected_for_property_oracle"] = False
    attempts.loc[selected_indices, "selected_for_property_oracle"] = True
    selected = attempts.loc[selected_indices].copy().reset_index(drop=True)
    feature_rows = [
        rau.generated_features(row, (*PROPERTIES,))
        for row in selected.to_dict(orient="records")
    ]
    features = pd.DataFrame(feature_rows)
    for column in ("canonical_smiles", "is_rdkit_valid", *PROPERTIES, *CONFOUNDS):
        selected[column] = features[column]
    selected.insert(0, "prior_pool_row", np.arange(len(selected)))
    selected_latents = (
        prior.detach().cpu().numpy()[selected_indices].astype(np.float32, copy=False)
    )
    _atomic_csv(attempts, attempt_path)
    _atomic_csv(selected, selected_path)
    _atomic_npy(selected_latents, latent_path)
    return selected_latents, selected, attempts


def fit_method_faithful_directions(
    prior_latents: np.ndarray,
    prior_pool: pd.DataFrame,
    *,
    output_dir: Path = METHOD_FAITHFUL_DIR,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    direction_path = output_dir / "chemspace_released_zinc_directions.npz"
    summary_path = output_dir / "direction_training_summary.csv"
    selection_path = output_dir / "direction_training_rows.csv"
    if direction_path.exists() and summary_path.exists() and selection_path.exists():
        payload = np.load(direction_path)
        directions = {name: np.asarray(payload[name]) for name in payload.files}
        if tuple(directions) == PROPERTIES:
            return (
                directions,
                pd.read_csv(summary_path),
                pd.read_csv(selection_path),
            )
        raise AssertionError("Method-faithful direction cache has wrong properties")

    directions: dict[str, np.ndarray] = {}
    summaries = []
    selections = []
    for property_name in PROPERTIES:
        result = fit_released_zinc_margin_boundary(
            prior_latents,
            prior_pool[property_name].to_numpy(),
            property_name=property_name,
        )
        directions[property_name] = result.direction
        summaries.append(result.summary)
        selections.append(result.selection)
    summary = pd.DataFrame(summaries)
    selection = pd.concat(selections, ignore_index=True)
    _atomic_npz(directions, direction_path)
    _atomic_csv(summary, summary_path)
    _atomic_csv(selection, selection_path)
    return directions, summary, selection


def _chunked_training_std(
    latents: np.ndarray,
    rows: np.ndarray,
    *,
    chunk_size: int = 25_000,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(latents.shape[1], dtype=np.float64)
    total_sq = np.zeros(latents.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(rows), chunk_size):
        block = np.asarray(latents[rows[start : start + chunk_size]], dtype=np.float64)
        total += block.sum(axis=0)
        total_sq += np.square(block).sum(axis=0)
        count += len(block)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise AssertionError("Training latent standard deviations are invalid")
    return mean, std


def fit_paired_directions(
    inputs: legacy.CachedInputs,
    *,
    output_dir: Path = PAIRED_DIR,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    direction_path = output_dir / "paired_native_directions.npz"
    summary_path = output_dir / "direction_training_summary.csv"
    selection_path = output_dir / "chemspace_direction_training_rows.csv"
    std_path = output_dir / "training_latent_std.npy"
    if all(path.exists() for path in (direction_path, summary_path, selection_path, std_path)):
        payload = np.load(direction_path)
        directions = {
            METHOD_OLS_PAIRED: {
                prop: np.asarray(payload[f"ols::{prop}"]) for prop in PROPERTIES
            },
            METHOD_CHEMSPACE_PAIRED: {
                prop: np.asarray(payload[f"chemspace::{prop}"])
                for prop in PROPERTIES
            },
        }
        return (
            directions,
            pd.read_csv(summary_path),
            pd.read_csv(selection_path),
            np.load(std_path),
        )

    train_rows = np.flatnonzero(np.asarray(inputs.split_codes) == 0)
    eligible = np.setdiff1d(
        train_rows, np.asarray(inputs.registry.row_indices), assume_unique=False
    )
    _, train_std = _chunked_training_std(inputs.latents, eligible)

    ols_cache = legacy.OUTPUTS / "raw_ols_seed_excluded_directions.npz"
    if not ols_cache.exists():
        legacy.fit_seed_excluded_raw_ols_directions(inputs)
    ols_payload = np.load(ols_cache)
    ols = {prop: _unit(ols_payload[prop]) for prop in PROPERTIES}

    chemspace: dict[str, np.ndarray] = {}
    summaries = []
    selections = []
    for property_name in PROPERTIES:
        scores = inputs.panel[:, inputs.panel_index[property_name]]
        result = fit_paired_extreme_boundary(
            inputs.latents,
            scores,
            eligible,
            property_name=property_name,
        )
        chemspace[property_name] = result.direction
        summaries.append({"method": METHOD_CHEMSPACE_PAIRED, **result.summary})
        selections.append(result.selection)
        summaries.append(
            {
                "method": METHOD_OLS_PAIRED,
                "property": property_name,
                "protocol": "continuous OLS on all eligible training rows",
                "labels_inspected_for_ranking": int(len(eligible)),
                "selected_rows": int(len(eligible)),
                "svm_fit_rows": np.nan,
                "svm_validation_rows": np.nan,
                "split_ratio": np.nan,
                "shuffle_seed": np.nan,
                "svm_C": np.nan,
                "fit_accuracy": np.nan,
                "validation_accuracy": np.nan,
                "support_vectors": np.nan,
                "svm_fit_seconds": np.nan,
                "direction_l2": float(np.linalg.norm(ols[property_name])),
                "direction_sha256": _array_hash(ols[property_name]),
            }
        )
    summary = pd.DataFrame(summaries)
    selection = pd.concat(selections, ignore_index=True)
    payload = {
        **{f"ols::{prop}": ols[prop] for prop in PROPERTIES},
        **{f"chemspace::{prop}": chemspace[prop] for prop in PROPERTIES},
    }
    _atomic_npz(payload, direction_path)
    _atomic_csv(summary, summary_path)
    _atomic_csv(selection, selection_path)
    _atomic_npy(train_std, std_path)
    return (
        {METHOD_OLS_PAIRED: ols, METHOD_CHEMSPACE_PAIRED: chemspace},
        summary,
        selection,
        train_std,
    )


def _scaled_jobs(
    native: Mapping[str, Mapping[str, np.ndarray]],
    train_std: np.ndarray | None,
    *,
    scales: Sequence[str],
) -> tuple[list[DirectionJob], pd.DataFrame]:
    jobs = []
    records = []
    for method, by_property in native.items():
        for property_name, native_direction in by_property.items():
            unit = _unit(native_direction)
            native_std_norm = (
                float(np.linalg.norm(unit / train_std))
                if train_std is not None
                else float("nan")
            )
            for scale in scales:
                if scale == SCALE_NATIVE:
                    direction = unit
                elif scale == SCALE_STANDARDIZED:
                    if train_std is None:
                        raise ValueError("train_std is required for standardized scale")
                    direction = unit / native_std_norm
                else:
                    raise ValueError(f"Unknown scale: {scale}")
                jobs.append(DirectionJob(method, property_name, scale, direction))
                records.append(
                    {
                        "method": method,
                        "property": property_name,
                        "scale": scale,
                        "native_direction_l2": float(np.linalg.norm(unit)),
                        "native_direction_train_std_norm": native_std_norm,
                        "traversal_vector_l2": float(np.linalg.norm(direction)),
                        "traversal_vector_train_std_norm": (
                            float(np.linalg.norm(direction / train_std))
                            if train_std is not None
                            else float("nan")
                        ),
                        "traversal_vector_sha256": _array_hash(direction),
                    }
                )
    return jobs, pd.DataFrame(records)


def _reference_identity_from_centers(frame: pd.DataFrame) -> pd.DataFrame:
    centers = frame[np.isclose(frame["alpha"], 0.0)].copy()
    if centers["seed_id"].nunique() != frame["seed_id"].nunique():
        raise AssertionError("Every prior path must contain alpha=0")
    columns = ["seed_id", "canonical_smiles", *PROPERTIES]
    centers = centers[columns].drop_duplicates("seed_id").sort_values("seed_id")
    centers.rename(
        columns={"canonical_smiles": "seed_canonical_smiles"}, inplace=True
    )
    for property_name in PROPERTIES:
        centers.rename(
            columns={property_name: f"seed::{property_name}"}, inplace=True
        )
    return centers.reset_index(drop=True)


def _attach_seed_metrics(
    frame: pd.DataFrame,
    seed_identity: pd.DataFrame,
) -> pd.DataFrame:
    rau = _rau()
    DataStructs = importlib.import_module("rdkit.DataStructs")
    seed_lookup = seed_identity.set_index("seed_id").to_dict(orient="index")
    seed_features = {
        int(seed_id): rau._structure_identity_features(
            record.get("seed_canonical_smiles")
        )
        for seed_id, record in seed_lookup.items()
    }
    valid_mask = rau.valid_canonical_mask(frame)
    generated = {
        str(smiles): rau._structure_identity_features(str(smiles))
        for smiles in frame.loc[valid_mask, "canonical_smiles"].dropna().unique()
        if str(smiles).strip()
    }
    seed_canonical = []
    seed_scaffold = []
    generated_scaffold = []
    similarities = []
    scaffold_evaluable = []
    scaffold_retained = []
    changed = []
    for row in frame.itertuples(index=False):
        seed_feature = seed_features[int(row.seed_id)]
        generated_feature = (
            generated.get(str(row.canonical_smiles))
            if bool(row.is_rdkit_valid)
            and pd.notna(row.canonical_smiles)
            and str(row.canonical_smiles).strip()
            else None
        )
        seed_canonical.append(seed_feature["canonical"])
        seed_scaffold.append(seed_feature["scaffold"])
        generated_scaffold.append(
            generated_feature["scaffold"] if generated_feature else None
        )
        if (
            generated_feature is not None
            and seed_feature["fingerprint"] is not None
        ):
            similarities.append(
                float(
                    DataStructs.TanimotoSimilarity(
                        seed_feature["fingerprint"],
                        generated_feature["fingerprint"],
                    )
                )
            )
        else:
            similarities.append(float("nan"))
        evaluable = bool(
            generated_feature is not None
            and seed_feature["scaffold"] is not None
            and generated_feature["scaffold"] is not None
        )
        scaffold_evaluable.append(evaluable)
        scaffold_retained.append(
            bool(seed_feature["scaffold"] == generated_feature["scaffold"])
            if evaluable
            else None
        )
        changed.append(
            bool(
                generated_feature is not None
                and seed_feature["canonical"] is not None
                and generated_feature["canonical"] != seed_feature["canonical"]
            )
        )
    enriched = frame.copy()
    enriched["seed_canonical_smiles"] = seed_canonical
    enriched["seed_scaffold"] = seed_scaffold
    enriched["generated_scaffold"] = generated_scaffold
    enriched["seed_similarity_tanimoto"] = similarities
    enriched["scaffold_evaluable"] = scaffold_evaluable
    enriched["scaffold_retained"] = pd.array(
        scaffold_retained, dtype="boolean"
    )
    enriched["changed_from_seed"] = changed
    return enriched


def decode_direction_jobs(
    jobs: Sequence[DirectionJob],
    seed_latents: np.ndarray,
    *,
    bundle: Any,
    loaded_model: Any,
    output_dir: Path,
    seed_identity: pd.DataFrame | None,
    batch_size: int = 64,
    alpha_grid: np.ndarray = LOCAL_ALPHAS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode local paths resumably, one method/property/scale shard at a time."""

    rau = _rau()
    seed_values = np.asarray(seed_latents, dtype=np.float32)
    alphas = np.asarray(alpha_grid, dtype=np.float64)
    if seed_values.ndim != 2 or seed_values.shape[1] != 256:
        raise ValueError("seed_latents must have shape (n, 256)")
    if alphas.ndim != 1 or len(alphas) < 2 or np.isclose(alphas, 0.0).sum() != 1:
        raise ValueError("alpha_grid must be one-dimensional and contain one zero")
    timing_path = output_dir / "decode_timing.csv"
    prior_timing = (
        pd.read_csv(timing_path)
        if timing_path.exists()
        else pd.DataFrame()
    )
    timing_rows = []
    frames = []
    for job in jobs:
        safe_method = (
            "chemspace"
            if job.method.startswith("ChemSpacE")
            else "molecula_ols"
        )
        shard = output_dir / (
            f"decoded__{safe_method}__{job.scale}__{job.property_name}.csv"
        )
        direction_hash = _array_hash(np.asarray(job.direction))
        expected_rows = len(seed_values) * len(alphas)
        if shard.exists():
            cached = pd.read_csv(shard)
            valid_cache = (
                len(cached) == expected_rows
                and cached["method"].eq(job.method).all()
                and cached["property"].eq(job.property_name).all()
                and cached["scale"].eq(job.scale).all()
                and cached["direction_sha256"].eq(direction_hash).all()
                and np.allclose(
                    cached[cached["seed_id"].eq(0)]["alpha"].to_numpy(),
                    alphas,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            if not valid_cache:
                raise AssertionError(f"Invalid local traversal cache: {shard}")
            frames.append(cached)
            prior = prior_timing[
                prior_timing.get("method", pd.Series(dtype=str)).eq(job.method)
                & prior_timing.get("property", pd.Series(dtype=str)).eq(
                    job.property_name
                )
                & prior_timing.get("scale", pd.Series(dtype=str)).eq(job.scale)
            ]
            timing_rows.append(
                {
                    "method": job.method,
                    "property": job.property_name,
                    "scale": job.scale,
                    "loaded_from_cache": True,
                    "candidates": expected_rows,
                    "decoder_seconds": (
                        float(prior.iloc[0]["decoder_seconds"])
                        if len(prior)
                        and pd.notna(prior.iloc[0]["decoder_seconds"])
                        else np.nan
                    ),
                    "descriptor_and_identity_seconds": (
                        float(prior.iloc[0]["descriptor_and_identity_seconds"])
                        if len(prior)
                        and pd.notna(
                            prior.iloc[0]["descriptor_and_identity_seconds"]
                        )
                        else np.nan
                    ),
                }
            )
            continue

        grid = (
            seed_values[:, None, :]
            + alphas.astype(np.float32)[None, :, None]
            * np.asarray(job.direction, dtype=np.float32)[None, None, :]
        ).reshape(-1, 256)
        decode_start = time.perf_counter()
        decoded = rau.decode_latents_adaptive(
            loaded_model, grid, bundle.id2tok, batch_size=batch_size
        )
        decode_seconds = time.perf_counter() - decode_start
        feature_start = time.perf_counter()
        frame = pd.DataFrame(decoded)
        features = pd.DataFrame(
            [
                rau.generated_features(row, (*PROPERTIES,))
                for row in decoded
            ]
        )
        for column in ("canonical_smiles", "is_rdkit_valid", *PROPERTIES, *CONFOUNDS):
            frame[column] = features[column]
        frame["is_rdkit_valid"] = rau.valid_canonical_mask(frame)
        frame.insert(0, "alpha", np.tile(alphas, len(seed_values)))
        frame.insert(
            0,
            "alpha_index",
            np.tile(np.arange(len(alphas)), len(seed_values)),
        )
        frame.insert(0, "seed_id", np.repeat(np.arange(len(seed_values)), len(alphas)))
        frame.insert(0, "direction_sha256", direction_hash)
        frame.insert(0, "direction_l2", float(np.linalg.norm(job.direction)))
        frame.insert(0, "scale", job.scale)
        frame.insert(0, "property", job.property_name)
        frame.insert(0, "method", job.method)
        identity = (
            _reference_identity_from_centers(frame)
            if seed_identity is None
            else seed_identity
        )
        frame = _attach_seed_metrics(frame, identity)
        frame["seed_property_value"] = frame["seed_id"].map(
            identity.set_index("seed_id")[f"seed::{job.property_name}"]
        )
        frame["property_delta"] = (
            frame[job.property_name] - frame["seed_property_value"]
        )
        frame["expected_signed_improvement"] = (
            np.sign(frame["alpha"]) * frame["property_delta"]
        )
        feature_seconds = time.perf_counter() - feature_start
        _atomic_csv(frame, shard)
        frames.append(frame)
        timing_rows.append(
            {
                "method": job.method,
                "property": job.property_name,
                "scale": job.scale,
                "loaded_from_cache": False,
                "candidates": expected_rows,
                "decoder_seconds": decode_seconds,
                "descriptor_and_identity_seconds": feature_seconds,
            }
        )
    candidates = pd.concat(frames, ignore_index=True)
    timing = pd.DataFrame(timing_rows)
    _atomic_csv(candidates, output_dir / "local_traversal_candidates.csv")
    _atomic_csv(timing, timing_path)
    return candidates, timing


def _training_canonical_set() -> set[str]:
    values = pd.read_csv(
        TRAINING_CANONICAL,
        usecols=["canonical_smiles"],
        dtype={"canonical_smiles": "string"},
    )["canonical_smiles"]
    return set(values.dropna().astype(str))


def _official_fp_generator() -> Any:
    generator_module = importlib.import_module(
        "rdkit.Chem.rdFingerprintGenerator"
    )
    return generator_module.GetMorganGenerator(
        radius=4,
        fpSize=2048,
        includeChirality=False,
        useBondTypes=True,
    )


def _synthesis_score_lookup(canonical_smiles: Iterable[str]) -> dict[str, tuple[float, float]]:
    """Use the reference RDKit Contrib SA_Score and NP_Score implementations."""

    RDConfig = importlib.import_module("rdkit.RDConfig")
    Chem = importlib.import_module("rdkit.Chem")
    sa_dir = str(Path(RDConfig.RDContribDir) / "SA_Score")
    np_dir = str(Path(RDConfig.RDContribDir) / "NP_Score")
    for path in (sa_dir, np_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    sascorer = importlib.import_module("sascorer")
    npscorer = importlib.import_module("npscorer")
    np_model = npscorer.readNPModel(
        str(Path(np_dir) / "publicnp.model.gz")
    )
    lookup = {}
    for smiles in sorted(set(str(value) for value in canonical_smiles)):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        lookup[smiles] = (
            float(sascorer.calculateScore(molecule)),
            float(npscorer.scoreMol(molecule, np_model)),
        )
    return lookup


def _attach_synthesis_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    valid_mask = candidates["is_rdkit_valid"].astype(bool)
    lookup = _synthesis_score_lookup(
        candidates.loc[valid_mask, "canonical_smiles"].dropna().astype(str)
    )
    scored = candidates.copy()
    scored["sa_score"] = scored["canonical_smiles"].map(
        {smiles: values[0] for smiles, values in lookup.items()}
    )
    scored["np_likeness_score"] = scored["canonical_smiles"].map(
        {smiles: values[1] for smiles, values in lookup.items()}
    )
    center = scored[np.isclose(scored["alpha"], 0.0)][
        ["method", "scale", "property", "seed_id", "sa_score", "np_likeness_score"]
    ].rename(
        columns={
            "sa_score": "seed_sa_score",
            "np_likeness_score": "seed_np_likeness_score",
        }
    )
    scored = scored.merge(
        center,
        on=["method", "scale", "property", "seed_id"],
        how="left",
        validate="many_to_one",
    )
    scored["sa_score_delta"] = scored["sa_score"] - scored["seed_sa_score"]
    scored["np_likeness_delta"] = (
        scored["np_likeness_score"] - scored["seed_np_likeness_score"]
    )
    return scored


def _path_quality_row(
    group: pd.DataFrame,
    *,
    property_name: str,
    global_property_range: float,
    fingerprint_lookup: Mapping[str, Any],
) -> dict[str, Any]:
    valid = group[
        group["is_rdkit_valid"].astype(bool)
        & group["canonical_smiles"].notna()
        & np.isfinite(group[property_name])
    ].sort_values("alpha")
    values = valid[property_name].to_numpy(dtype=np.float64)
    smiles = valid["canonical_smiles"].astype(str).tolist()
    requested = len(group)
    if len(valid) < 2:
        return {
            "requested_points": requested,
            "valid_points": len(valid),
            "full_path_valid": False,
            "property_monotonic_either": False,
            "property_monotonic_expected": False,
            "structure_distance_monotonic": False,
            "raw_path_diverse": False,
            "canonical_path_diverse": False,
            "ssr_official_code_compatible": False,
            "ssr_expected_all_valid": False,
            "rsr_local_official_code_compatible": False,
            "rsr_global_official_code_compatible": False,
        }
    differences = np.diff(values)
    monotonic_up = bool(np.all(differences >= 0))
    monotonic_down = bool(np.all(differences <= 0))
    property_monotonic_either = monotonic_up or monotonic_down
    fingerprints = [fingerprint_lookup[value] for value in smiles]
    DataStructs = importlib.import_module("rdkit.DataStructs")
    first = fingerprints[0]
    distances = np.asarray(
        [
            1.0 - float(DataStructs.TanimotoSimilarity(first, fingerprint))
            for fingerprint in fingerprints
        ]
    )
    distance_steps = np.diff(distances)
    structure_monotonic = bool(np.all(distance_steps >= 0))
    raw_smiles = valid["decoded_smiles"].fillna("").astype(str).tolist()
    raw_diverse = len(set(raw_smiles)) != 1
    canonical_diverse = len(set(smiles)) != 1
    local_range = float(np.ptp(values))
    local_tolerance = OFFICIAL_EPSILON * local_range
    global_tolerance = OFFICIAL_EPSILON * global_property_range
    relaxed_local = bool(
        np.all(differences >= -local_tolerance)
        or np.all(differences <= local_tolerance)
    )
    relaxed_global = bool(
        np.all(differences >= -global_tolerance)
        or np.all(differences <= global_tolerance)
    )
    relaxed_structure = bool(np.all(distance_steps >= -OFFICIAL_GAMMA))
    full_path_valid = len(valid) == requested
    return {
        "requested_points": requested,
        "valid_points": len(valid),
        "full_path_valid": full_path_valid,
        "property_monotonic_either": property_monotonic_either,
        "property_monotonic_expected": monotonic_up,
        "structure_distance_monotonic": structure_monotonic,
        "raw_path_diverse": raw_diverse,
        "canonical_path_diverse": canonical_diverse,
        "ssr_official_code_compatible": bool(
            property_monotonic_either and structure_monotonic and raw_diverse
        ),
        "ssr_expected_all_valid": bool(
            full_path_valid
            and monotonic_up
            and structure_monotonic
            and canonical_diverse
        ),
        "rsr_local_official_code_compatible": bool(
            relaxed_local and relaxed_structure and raw_diverse
        ),
        "rsr_global_official_code_compatible": bool(
            relaxed_global and relaxed_structure and raw_diverse
        ),
    }


def build_protocol_metrics(
    candidates: pd.DataFrame,
    *,
    full_training_property_ranges: Mapping[str, float],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute official path metrics plus MolecuLA identity metrics."""

    training = _training_canonical_set()
    valid_mask = candidates["is_rdkit_valid"].astype(bool)
    candidates = _attach_synthesis_scores(candidates)
    candidates["training_novel"] = (
        valid_mask
        & ~candidates["canonical_smiles"].fillna("").astype(str).isin(training)
    )

    Chem = importlib.import_module("rdkit.Chem")
    fp_generator = _official_fp_generator()
    fingerprint_lookup = {}
    for smiles in candidates.loc[valid_mask, "canonical_smiles"].dropna().unique():
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is not None:
            fingerprint_lookup[str(smiles)] = fp_generator.GetFingerprint(molecule)

    path_rows = []
    grouping = ["method", "scale", "property", "seed_id"]
    for key, group in candidates.groupby(grouping, sort=False):
        method, scale, property_name, seed_id = key
        valid = group[group["is_rdkit_valid"].astype(bool)]
        quality = _path_quality_row(
            group,
            property_name=property_name,
            global_property_range=full_training_property_ranges[property_name],
            fingerprint_lookup=fingerprint_lookup,
        )
        expected = valid[~np.isclose(valid["alpha"], 0.0)]
        scaffold = valid[valid["scaffold_evaluable"].astype(bool)]
        path_rows.append(
            {
                "method": method,
                "scale": scale,
                "property": property_name,
                "seed_id": int(seed_id),
                **quality,
                "spearman_alpha_property": _safe_spearman(
                    valid["alpha"], valid[property_name]
                ),
                "slope_property_per_alpha": _safe_slope(
                    valid["alpha"], valid[property_name]
                ),
                "spearman_alpha_sa_score": _safe_spearman(
                    valid["alpha"], valid["sa_score"]
                ),
                "slope_sa_score_per_alpha": _safe_slope(
                    valid["alpha"], valid["sa_score"]
                ),
                "spearman_alpha_np_likeness": _safe_spearman(
                    valid["alpha"], valid["np_likeness_score"]
                ),
                "slope_np_likeness_per_alpha": _safe_slope(
                    valid["alpha"], valid["np_likeness_score"]
                ),
                "expected_sign_fraction_valid_nonzero": (
                    float((expected["expected_signed_improvement"] > 0).mean())
                    if len(expected)
                    else float("nan")
                ),
                "changed_from_seed_fraction_all": float(
                    group["changed_from_seed"].astype(bool).mean()
                ),
                "changed_from_seed_fraction_valid": (
                    float(valid["changed_from_seed"].astype(bool).mean())
                    if len(valid)
                    else float("nan")
                ),
                "median_seed_similarity": float(
                    valid["seed_similarity_tanimoto"].median()
                ),
                "scaffold_evaluable_fraction": float(
                    valid["scaffold_evaluable"].astype(bool).mean()
                )
                if len(valid)
                else float("nan"),
                "scaffold_retention_fraction": (
                    float(scaffold["scaffold_retained"].astype(bool).mean())
                    if len(scaffold)
                    else float("nan")
                ),
            }
        )
    per_seed = pd.DataFrame(path_rows)

    alpha_rows = []
    for key, group in candidates.groupby(
        ["method", "scale", "property", "alpha"], sort=False
    ):
        method, scale, property_name, alpha = key
        valid = group[group["is_rdkit_valid"].astype(bool)]
        scaffold = valid[valid["scaffold_evaluable"].astype(bool)]
        alpha_rows.append(
            {
                "method": method,
                "scale": scale,
                "property": property_name,
                "alpha": float(alpha),
                "requested_candidates": len(group),
                "valid_fraction": float(group["is_rdkit_valid"].astype(bool).mean()),
                "unique_canonical_fraction_total": (
                    float(valid["canonical_smiles"].nunique() / len(group))
                ),
                "changed_from_seed_fraction_all": float(
                    group["changed_from_seed"].astype(bool).mean()
                ),
                "median_property": float(valid[property_name].median()),
                "median_property_delta": float(valid["property_delta"].median()),
                "median_sa_score": float(valid["sa_score"].median()),
                "median_sa_score_delta": float(valid["sa_score_delta"].median()),
                "median_np_likeness_score": float(
                    valid["np_likeness_score"].median()
                ),
                "median_np_likeness_delta": float(
                    valid["np_likeness_delta"].median()
                ),
                "fraction_at_decoder_token_budget": float(
                    group["selfies_len_tokens"]
                    .eq(DECODER_MAX_SELFIES_TOKENS)
                    .mean()
                ),
                "expected_sign_fraction_valid": (
                    float((valid["expected_signed_improvement"] > 0).mean())
                    if len(valid) and not np.isclose(alpha, 0.0)
                    else float("nan")
                ),
                "median_seed_similarity": float(
                    valid["seed_similarity_tanimoto"].median()
                ),
                "q25_seed_similarity": float(
                    valid["seed_similarity_tanimoto"].quantile(0.25)
                ),
                "q75_seed_similarity": float(
                    valid["seed_similarity_tanimoto"].quantile(0.75)
                ),
                "scaffold_retention_fraction": (
                    float(scaffold["scaffold_retained"].astype(bool).mean())
                    if len(scaffold)
                    else float("nan")
                ),
            }
        )
    per_alpha = pd.DataFrame(alpha_rows)

    summary_rows = []
    for key, group in candidates.groupby(
        ["method", "scale", "property"], sort=False
    ):
        method, scale, property_name = key
        valid = group[group["is_rdkit_valid"].astype(bool)]
        seed_group = per_seed[
            per_seed["method"].eq(method)
            & per_seed["scale"].eq(scale)
            & per_seed["property"].eq(property_name)
        ]
        summary_rows.append(
            {
                "method": method,
                "scale": scale,
                "property": property_name,
                "requested_candidates": len(group),
                "valid_count": len(valid),
                "valid_fraction": float(len(valid) / len(group)),
                "raw_unique_count": int(valid["decoded_smiles"].nunique()),
                "raw_unique_fraction_total": float(
                    valid["decoded_smiles"].nunique() / len(group)
                ),
                "canonical_unique_count": int(
                    valid["canonical_smiles"].nunique()
                ),
                "canonical_unique_fraction_total": float(
                    valid["canonical_smiles"].nunique() / len(group)
                ),
                "training_novel_fraction_total": float(
                    group["training_novel"].astype(bool).mean()
                ),
                "training_novel_fraction_valid": (
                    float(valid["training_novel"].astype(bool).mean())
                    if len(valid)
                    else float("nan")
                ),
                "full_path_valid_fraction": float(
                    seed_group["full_path_valid"].mean()
                ),
                "ssr_official_code_compatible": float(
                    seed_group["ssr_official_code_compatible"].mean()
                ),
                "ssr_expected_all_valid": float(
                    seed_group["ssr_expected_all_valid"].mean()
                ),
                "rsr_local_official_code_compatible": float(
                    seed_group["rsr_local_official_code_compatible"].mean()
                ),
                "rsr_global_official_code_compatible": float(
                    seed_group["rsr_global_official_code_compatible"].mean()
                ),
                "median_per_seed_spearman": float(
                    seed_group["spearman_alpha_property"].median()
                ),
                "median_per_seed_slope": float(
                    seed_group["slope_property_per_alpha"].median()
                ),
                "fraction_seeds_positive_slope": float(
                    (seed_group["slope_property_per_alpha"] > 0).mean()
                ),
                "median_expected_sign_fraction": float(
                    seed_group["expected_sign_fraction_valid_nonzero"].median()
                ),
                "changed_from_seed_fraction_all": float(
                    group["changed_from_seed"].astype(bool).mean()
                ),
                "median_seed_similarity": float(
                    valid["seed_similarity_tanimoto"].median()
                ),
                "scaffold_evaluable_fraction": float(
                    valid["scaffold_evaluable"].astype(bool).mean()
                )
                if len(valid)
                else float("nan"),
                "fraction_at_decoder_token_budget": float(
                    group["selfies_len_tokens"]
                    .eq(DECODER_MAX_SELFIES_TOKENS)
                    .mean()
                ),
                "median_sa_score": float(valid["sa_score"].median()),
                "median_np_likeness_score": float(
                    valid["np_likeness_score"].median()
                ),
                "median_per_seed_spearman_sa_score": float(
                    seed_group["spearman_alpha_sa_score"].median()
                ),
                "median_per_seed_spearman_np_likeness": float(
                    seed_group["spearman_alpha_np_likeness"].median()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    _atomic_csv(candidates, output_dir / "local_traversal_candidates.csv")
    _atomic_csv(per_seed, output_dir / "path_metrics_per_seed.csv")
    _atomic_csv(per_alpha, output_dir / "metrics_per_signed_alpha.csv")
    _atomic_csv(summary, output_dir / "compact_summary.csv")
    return per_seed, per_alpha, summary


def selfies_confound_audit(
    candidates: pd.DataFrame,
    *,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit decoded SELFIES diagnostics for every method/scale/path."""

    rows = []
    grouping = ["method", "scale", "property", "seed_id"]
    for key, group in candidates.groupby(grouping, sort=False):
        method, scale, property_name, seed_id = key
        for confound in CONFOUNDS:
            rows.append(
                {
                    "method": method,
                    "scale": scale,
                    "property": property_name,
                    "seed_id": int(seed_id),
                    "confound": confound,
                    "spearman_alpha_confound": _safe_spearman(
                        group["alpha"], group[confound]
                    ),
                    "spearman_abs_alpha_confound": _safe_spearman(
                        group["alpha"].abs(), group[confound]
                    ),
                    "slope_confound_per_alpha": _safe_slope(
                        group["alpha"], group[confound]
                    ),
                    "slope_confound_per_abs_alpha": _safe_slope(
                        group["alpha"].abs(), group[confound]
                    ),
                }
            )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(
            ["method", "scale", "property", "confound"], as_index=False
        )
        .agg(
            seeds=("seed_id", "size"),
            median_spearman_alpha_confound=(
                "spearman_alpha_confound",
                "median",
            ),
            median_abs_spearman_alpha_confound=(
                "spearman_alpha_confound",
                lambda values: values.abs().median(),
            ),
            median_spearman_abs_alpha_confound=(
                "spearman_abs_alpha_confound",
                "median",
            ),
            median_abs_spearman_abs_alpha_confound=(
                "spearman_abs_alpha_confound",
                lambda values: values.abs().median(),
            ),
            fraction_positive_spearman=(
                "spearman_alpha_confound",
                lambda values: (values > 0).mean(),
            ),
            median_slope_confound_per_alpha=(
                "slope_confound_per_alpha",
                "median",
            ),
            median_slope_confound_per_abs_alpha=(
                "slope_confound_per_abs_alpha",
                "median",
            ),
        )
    )
    _atomic_csv(per_seed, output_dir / "selfies_confound_per_seed.csv")
    _atomic_csv(summary, output_dir / "selfies_confound_summary.csv")
    return per_seed, summary


def matched_seed_similarity(
    candidates: pd.DataFrame,
    *,
    output_dir: Path = PAIRED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    methods = (METHOD_OLS_PAIRED, METHOD_CHEMSPACE_PAIRED)
    for scale in (SCALE_NATIVE, SCALE_STANDARDIZED):
        for property_name in PROPERTIES:
            for seed_id in range(50):
                for sign_name, sign in (("negative", -1), ("positive", 1)):
                    selected = {}
                    for method in methods:
                        group = candidates[
                            candidates["method"].eq(method)
                            & candidates["scale"].eq(scale)
                            & candidates["property"].eq(property_name)
                            & candidates["seed_id"].eq(seed_id)
                            & (np.sign(candidates["alpha"]) == sign)
                            & candidates["is_rdkit_valid"].astype(bool)
                        ].sort_values(
                            ["seed_similarity_tanimoto", "alpha"],
                            ascending=[False, sign < 0],
                        )
                        selected[method] = group.reset_index(drop=True)
                    n_pairs = min(len(selected[method]) for method in methods)
                    for rank in range(n_pairs):
                        ours = selected[METHOD_OLS_PAIRED].iloc[rank]
                        chem = selected[METHOD_CHEMSPACE_PAIRED].iloc[rank]
                        rows.append(
                            {
                                "scale": scale,
                                "property": property_name,
                                "seed_id": seed_id,
                                "alpha_sign": sign_name,
                                "similarity_rank": rank,
                                "ols_alpha": ours["alpha"],
                                "chemspace_alpha": chem["alpha"],
                                "ols_seed_similarity": ours[
                                    "seed_similarity_tanimoto"
                                ],
                                "chemspace_seed_similarity": chem[
                                    "seed_similarity_tanimoto"
                                ],
                                "absolute_similarity_gap": abs(
                                    ours["seed_similarity_tanimoto"]
                                    - chem["seed_similarity_tanimoto"]
                                ),
                                "ols_expected_improvement": ours[
                                    "expected_signed_improvement"
                                ],
                                "chemspace_expected_improvement": chem[
                                    "expected_signed_improvement"
                                ],
                            }
                        )
    pairs = pd.DataFrame(rows)
    summary = (
        pairs.groupby(["scale", "property", "alpha_sign"], as_index=False)
        .agg(
            pairs=("similarity_rank", "size"),
            median_absolute_similarity_gap=("absolute_similarity_gap", "median"),
            p95_absolute_similarity_gap=(
                "absolute_similarity_gap",
                lambda values: values.quantile(0.95),
            ),
            median_ols_expected_improvement=(
                "ols_expected_improvement",
                "median",
            ),
            median_chemspace_expected_improvement=(
                "chemspace_expected_improvement",
                "median",
            ),
        )
    )
    _atomic_csv(pairs, output_dir / "matched_seed_similarity_pairs.csv")
    _atomic_csv(summary, output_dir / "matched_seed_similarity_summary.csv")
    return pairs, summary


def matched_latent_displacement(
    candidates: pd.DataFrame,
    *,
    output_dir: Path = PAIRED_DISPLACEMENT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair the two learners at identical seed, alpha, and direction norm."""

    value_columns = [
        "is_rdkit_valid",
        "changed_from_seed",
        "property_delta",
        "expected_signed_improvement",
        "seed_similarity_tanimoto",
        "scaffold_evaluable",
        "scaffold_retained",
        "sa_score_delta",
        "np_likeness_delta",
        "selfies_len_tokens",
    ]
    index = ["scale", "property", "seed_id", "alpha"]
    left = candidates[candidates["method"].eq(METHOD_OLS_PAIRED)][
        index + value_columns
    ].copy()
    right = candidates[candidates["method"].eq(METHOD_CHEMSPACE_PAIRED)][
        index + value_columns
    ].copy()
    pairs = left.merge(
        right,
        on=index,
        how="inner",
        validate="one_to_one",
        suffixes=("_ols", "_chemspace"),
    )
    pairs["alpha_sign"] = np.where(
        pairs["alpha"] < 0,
        "negative",
        np.where(pairs["alpha"] > 0, "positive", "zero"),
    )
    pairs["absolute_latent_displacement"] = pairs["alpha"].abs()
    pairs["improvement_difference_chemspace_minus_ols"] = (
        pairs["expected_signed_improvement_chemspace"]
        - pairs["expected_signed_improvement_ols"]
    )
    pairs["sa_delta_difference_chemspace_minus_ols"] = (
        pairs["sa_score_delta_chemspace"] - pairs["sa_score_delta_ols"]
    )
    pairs["similarity_difference_chemspace_minus_ols"] = (
        pairs["seed_similarity_tanimoto_chemspace"]
        - pairs["seed_similarity_tanimoto_ols"]
    )
    nonzero = pairs[~pairs["alpha_sign"].eq("zero")]
    summary = (
        nonzero.groupby(["scale", "property", "alpha_sign"], as_index=False)
        .agg(
            pairs=("seed_id", "size"),
            both_valid=(
                "is_rdkit_valid_ols",
                lambda values: int(
                    (
                        values.astype(bool)
                        & nonzero.loc[values.index, "is_rdkit_valid_chemspace"]
                        .astype(bool)
                    ).sum()
                ),
            ),
            median_ols_expected_improvement=(
                "expected_signed_improvement_ols",
                "median",
            ),
            median_chemspace_expected_improvement=(
                "expected_signed_improvement_chemspace",
                "median",
            ),
            median_improvement_difference_chemspace_minus_ols=(
                "improvement_difference_chemspace_minus_ols",
                "median",
            ),
            median_ols_seed_similarity=(
                "seed_similarity_tanimoto_ols",
                "median",
            ),
            median_chemspace_seed_similarity=(
                "seed_similarity_tanimoto_chemspace",
                "median",
            ),
            median_similarity_difference_chemspace_minus_ols=(
                "similarity_difference_chemspace_minus_ols",
                "median",
            ),
            median_ols_sa_delta=("sa_score_delta_ols", "median"),
            median_chemspace_sa_delta=("sa_score_delta_chemspace", "median"),
            median_sa_delta_difference_chemspace_minus_ols=(
                "sa_delta_difference_chemspace_minus_ols",
                "median",
            ),
            ols_fraction_at_decoder_token_budget=(
                "selfies_len_tokens_ols",
                lambda values: values.eq(DECODER_MAX_SELFIES_TOKENS).mean(),
            ),
            chemspace_fraction_at_decoder_token_budget=(
                "selfies_len_tokens_chemspace",
                lambda values: values.eq(DECODER_MAX_SELFIES_TOKENS).mean(),
            ),
        )
    )
    _atomic_csv(pairs, output_dir / "matched_latent_displacement_pairs.csv")
    _atomic_csv(
        summary, output_dir / "matched_latent_displacement_summary.csv"
    )
    return pairs, summary


def constrained_design_tables(
    candidates: pd.DataFrame,
    *,
    output_dir: Path = PAIRED_DISPLACEMENT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse traversal candidates at ChemSpacE's published similarity cutoffs."""

    rows = []
    grouping = ["method", "scale", "property", "seed_id"]
    for key, group in candidates.groupby(grouping, sort=False):
        method, scale, property_name, seed_id = key
        for objective, multiplier in (("maximize", 1.0), ("minimize", -1.0)):
            for cutoff in PUBLISHED_SIMILARITY_CUTOFFS:
                eligible = group[
                    group["is_rdkit_valid"].astype(bool)
                    & group["changed_from_seed"].astype(bool)
                    & ~np.isclose(group["alpha"], 0.0)
                    & group["seed_similarity_tanimoto"].ge(cutoff)
                ].copy()
                eligible["objective_improvement"] = (
                    multiplier * eligible["property_delta"]
                )
                if len(eligible):
                    selected = eligible.loc[
                        eligible["objective_improvement"].idxmax()
                    ]
                    best = float(selected["objective_improvement"])
                    selected_alpha = float(selected["alpha"])
                    selected_similarity = float(
                        selected["seed_similarity_tanimoto"]
                    )
                    selected_sa_delta = float(selected["sa_score_delta"])
                else:
                    best = float("nan")
                    selected_alpha = float("nan")
                    selected_similarity = float("nan")
                    selected_sa_delta = float("nan")
                rows.append(
                    {
                        "method": method,
                        "scale": scale,
                        "property": property_name,
                        "seed_id": int(seed_id),
                        "objective": objective,
                        "seed_similarity_cutoff": float(cutoff),
                        "candidate_budget": int(len(group)),
                        "eligible_valid_changed_candidates": int(len(eligible)),
                        "success": bool(best > 0)
                        if np.isfinite(best)
                        else False,
                        "best_objective_improvement": best,
                        "selected_alpha": selected_alpha,
                        "selected_seed_similarity": selected_similarity,
                        "selected_sa_score_delta": selected_sa_delta,
                    }
                )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(
            [
                "method",
                "scale",
                "property",
                "objective",
                "seed_similarity_cutoff",
            ],
            as_index=False,
        )
        .agg(
            seeds=("seed_id", "size"),
            successful_seeds=("success", "sum"),
            success_fraction=("success", "mean"),
            median_best_improvement=("best_objective_improvement", "median"),
            median_selected_similarity=("selected_seed_similarity", "median"),
            median_selected_alpha=("selected_alpha", "median"),
            median_selected_sa_score_delta=(
                "selected_sa_score_delta",
                "median",
            ),
            median_eligible_candidates=(
                "eligible_valid_changed_candidates",
                "median",
            ),
        )
    )
    _atomic_csv(per_seed, output_dir / "constrained_design_per_seed.csv")
    _atomic_csv(summary, output_dir / "constrained_design_summary.csv")
    return per_seed, summary


def direction_cosines(
    directions: Mapping[str, Mapping[str, np.ndarray]],
    train_std: np.ndarray,
    *,
    output_dir: Path = PAIRED_DIR,
) -> pd.DataFrame:
    rows = []
    for property_name in PROPERTIES:
        ols = _unit(directions[METHOD_OLS_PAIRED][property_name])
        chem = _unit(directions[METHOD_CHEMSPACE_PAIRED][property_name])
        ols_standard = _unit(ols / train_std)
        chem_standard = _unit(chem / train_std)
        rows.append(
            {
                "property": property_name,
                "raw_latent_cosine": float(ols @ chem),
                "train_standardized_metric_cosine": float(
                    ols_standard @ chem_standard
                ),
            }
        )
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, output_dir / "direction_cosines.csv")
    return frame


def _full_training_property_ranges(
    inputs: legacy.CachedInputs,
) -> dict[str, float]:
    train = np.asarray(inputs.split_codes) == 0
    return {
        property_name: float(
            np.ptp(inputs.panel[train, inputs.panel_index[property_name]])
        )
        for property_name in PROPERTIES
    }


def create_figures(
    faithful_alpha: pd.DataFrame,
    faithful_summary: pd.DataFrame,
    paired_alpha: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, property_name in zip(axes.flat, PROPERTIES):
        group = faithful_alpha[faithful_alpha["property"].eq(property_name)]
        axis.plot(group["alpha"], group["median_property_delta"], marker="o", ms=3)
        axis.axhline(0, color="0.7", lw=1)
        axis.set_title(property_name)
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel("median decoded change")
    figure.savefig(FIGURE_DIR / "method_faithful_property_trajectories.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    styles = {
        (METHOD_OLS_PAIRED, SCALE_NATIVE): ("#1f77b4", "-"),
        (METHOD_CHEMSPACE_PAIRED, SCALE_NATIVE): ("#ff7f0e", "-"),
        (METHOD_OLS_PAIRED, SCALE_STANDARDIZED): ("#1f77b4", "--"),
        (METHOD_CHEMSPACE_PAIRED, SCALE_STANDARDIZED): ("#ff7f0e", "--"),
    }
    for axis, property_name in zip(axes.flat, PROPERTIES):
        for (method, scale), (color, linestyle) in styles.items():
            group = paired_alpha[
                paired_alpha["property"].eq(property_name)
                & paired_alpha["method"].eq(method)
                & paired_alpha["scale"].eq(scale)
            ].sort_values("alpha")
            axis.plot(
                group["alpha"],
                group["median_property_delta"],
                color=color,
                linestyle=linestyle,
                label=f"{method}; {scale}",
            )
        axis.axhline(0, color="0.7", lw=1)
        axis.set_title(property_name)
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel("median decoded change")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    figure.savefig(
        FIGURE_DIR / "paired_displacement_property_trajectories.png", dpi=220
    )
    plt.close(figure)

    for score_column, delta_column, label, filename in (
        (
            "median_sa_score",
            "median_sa_score_delta",
            "median ΔSA (lower is easier)",
            "paired_displacement_sa_score_trajectories.png",
        ),
        (
            "median_np_likeness_score",
            "median_np_likeness_delta",
            "median ΔNP-likeness",
            "paired_displacement_np_score_trajectories.png",
        ),
    ):
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for axis, property_name in zip(axes.flat, PROPERTIES):
            for (method, scale), (color, linestyle) in styles.items():
                group = paired_alpha[
                    paired_alpha["property"].eq(property_name)
                    & paired_alpha["method"].eq(method)
                    & paired_alpha["scale"].eq(scale)
                ].sort_values("alpha")
                axis.plot(
                    group["alpha"],
                    group[delta_column],
                    color=color,
                    linestyle=linestyle,
                    label=f"{method}; {scale}",
                )
            axis.axhline(0, color="0.7", lw=1)
            axis.set_title(property_name)
            axis.set_xlabel(r"$\alpha$")
            axis.set_ylabel(label)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
        figure.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight")
        plt.close(figure)

    quality = pd.concat(
        [
            faithful_summary.assign(experiment="method-faithful"),
            paired_summary.assign(experiment="paired-local"),
        ],
        ignore_index=True,
    )
    _atomic_csv(quality, OUTPUT_ROOT / "all_compact_summaries.csv")

    native = paired_summary[paired_summary["scale"].eq(SCALE_NATIVE)].copy()
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    metric_specs = (
        ("valid_fraction", "RDKit-valid fraction"),
        ("median_seed_similarity", "median seed similarity"),
        (
            "fraction_at_decoder_token_budget",
            "fraction at 154-token budget",
        ),
        ("scaffold_evaluable_fraction", "scaffold-evaluable fraction"),
    )
    positions = np.arange(len(PROPERTIES))
    width = 0.36
    for axis, (metric, label) in zip(axes.flat, metric_specs):
        for offset, method, color in (
            (-width / 2, METHOD_OLS_PAIRED, "#1f77b4"),
            (width / 2, METHOD_CHEMSPACE_PAIRED, "#ff7f0e"),
        ):
            values = (
                native[native["method"].eq(method)]
                .set_index("property")
                .reindex(PROPERTIES)[metric]
                .to_numpy()
            )
            axis.bar(
                positions + offset,
                values,
                width,
                color=color,
                label=method,
            )
        axis.set_xticks(positions, ["Fsp3", "HBA", "TPSA", "cLogP", "Bertz", "HAC"])
        axis.set_ylim(0, 1.02)
        axis.set_ylabel(label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    figure.savefig(
        FIGURE_DIR / "paired_displacement_decoder_quality.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    constrained_path = PAIRED_DISPLACEMENT_DIR / "constrained_design_summary.csv"
    if constrained_path.exists():
        constrained = pd.read_csv(constrained_path)
        constrained = constrained[
            constrained["scale"].eq(SCALE_NATIVE)
            & constrained["objective"].eq("maximize")
        ]
        figure, axes = plt.subplots(
            2, 3, figsize=(12, 7), constrained_layout=True
        )
        for axis, property_name in zip(axes.flat, PROPERTIES):
            for method, color in (
                (METHOD_OLS_PAIRED, "#1f77b4"),
                (METHOD_CHEMSPACE_PAIRED, "#ff7f0e"),
            ):
                group = constrained[
                    constrained["method"].eq(method)
                    & constrained["property"].eq(property_name)
                ].sort_values("seed_similarity_cutoff")
                axis.plot(
                    group["seed_similarity_cutoff"],
                    group["success_fraction"],
                    marker="o",
                    color=color,
                    label=method,
                )
            axis.set_title(property_name)
            axis.set_xlabel("published similarity cutoff")
            axis.set_ylabel("maximization success")
            axis.set_ylim(-0.02, 1.02)
        axes.flat[0].legend(loc="upper right", fontsize=7)
        figure.savefig(
            FIGURE_DIR / "paired_displacement_constrained_success.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)


def load_completed_results() -> dict[str, pd.DataFrame]:
    """Load completed protocol tables without model loading or decoding."""

    def read(directory: Path, filename: str) -> pd.DataFrame:
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; execute run_all() to create the experiment"
            )
        return pd.read_csv(path)

    return {
        "faithful_training": read(
            METHOD_FAITHFUL_DIR, "direction_training_summary.csv"
        ),
        "faithful_selection": read(
            METHOD_FAITHFUL_DIR, "direction_training_rows.csv"
        ),
        "faithful_prior_pool": read(
            METHOD_FAITHFUL_DIR, "prior_training_pool.csv"
        ),
        "faithful_prior_attempts": read(
            METHOD_FAITHFUL_DIR, "prior_training_decode_attempts.csv"
        ),
        "faithful_candidates": read(
            METHOD_FAITHFUL_DIR, "local_traversal_candidates.csv"
        ),
        "faithful_per_seed": read(
            METHOD_FAITHFUL_DIR, "path_metrics_per_seed.csv"
        ),
        "faithful_per_alpha": read(
            METHOD_FAITHFUL_DIR, "metrics_per_signed_alpha.csv"
        ),
        "faithful_summary": read(METHOD_FAITHFUL_DIR, "compact_summary.csv"),
        "faithful_timing": read(METHOD_FAITHFUL_DIR, "decode_timing.csv"),
        "faithful_confound_per_seed": read(
            METHOD_FAITHFUL_DIR, "selfies_confound_per_seed.csv"
        ),
        "faithful_confound_summary": read(
            METHOD_FAITHFUL_DIR, "selfies_confound_summary.csv"
        ),
        "paired_training": read(PAIRED_DIR, "direction_training_summary.csv"),
        "paired_selection": read(
            PAIRED_DIR, "chemspace_direction_training_rows.csv"
        ),
        "paired_local_candidates": read(
            PAIRED_DIR, "local_traversal_candidates.csv"
        ),
        "paired_local_per_seed": read(PAIRED_DIR, "path_metrics_per_seed.csv"),
        "paired_local_per_alpha": read(
            PAIRED_DIR, "metrics_per_signed_alpha.csv"
        ),
        "paired_local_summary": read(PAIRED_DIR, "compact_summary.csv"),
        "paired_local_timing": read(PAIRED_DIR, "decode_timing.csv"),
        "paired_local_confound_per_seed": read(
            PAIRED_DIR, "selfies_confound_per_seed.csv"
        ),
        "paired_local_confound_summary": read(
            PAIRED_DIR, "selfies_confound_summary.csv"
        ),
        "paired_candidates": read(
            PAIRED_DISPLACEMENT_DIR, "local_traversal_candidates.csv"
        ),
        "paired_per_seed": read(
            PAIRED_DISPLACEMENT_DIR, "path_metrics_per_seed.csv"
        ),
        "paired_per_alpha": read(
            PAIRED_DISPLACEMENT_DIR, "metrics_per_signed_alpha.csv"
        ),
        "paired_summary": read(
            PAIRED_DISPLACEMENT_DIR, "compact_summary.csv"
        ),
        "paired_timing": read(PAIRED_DISPLACEMENT_DIR, "decode_timing.csv"),
        "paired_confound_per_seed": read(
            PAIRED_DISPLACEMENT_DIR, "selfies_confound_per_seed.csv"
        ),
        "paired_confound_summary": read(
            PAIRED_DISPLACEMENT_DIR, "selfies_confound_summary.csv"
        ),
        "matched_pairs": read(
            PAIRED_DISPLACEMENT_DIR, "matched_seed_similarity_pairs.csv"
        ),
        "matched_summary": read(
            PAIRED_DISPLACEMENT_DIR, "matched_seed_similarity_summary.csv"
        ),
        "matched_displacement_pairs": read(
            PAIRED_DISPLACEMENT_DIR, "matched_latent_displacement_pairs.csv"
        ),
        "matched_displacement_summary": read(
            PAIRED_DISPLACEMENT_DIR, "matched_latent_displacement_summary.csv"
        ),
        "constrained_per_seed": read(
            PAIRED_DISPLACEMENT_DIR, "constrained_design_per_seed.csv"
        ),
        "constrained_summary": read(
            PAIRED_DISPLACEMENT_DIR, "constrained_design_summary.csv"
        ),
        "cosines": read(PAIRED_DISPLACEMENT_DIR, "direction_cosines.csv"),
    }


def run_all(*, device: str = "cuda", batch_size: int = 64) -> dict[str, Any]:
    """Execute both experiments with shard-level resumption."""

    inputs = legacy.load_cached_inputs()
    bundle, loaded_model = legacy.load_frozen_model(device=device)
    full_ranges = _full_training_property_ranges(inputs)

    prior_latents, prior_pool, prior_attempts = prepare_prior_training_pool(
        bundle=bundle,
        loaded_model=loaded_model,
        batch_size=batch_size,
    )
    faithful_directions, faithful_training, faithful_selection = (
        fit_method_faithful_directions(prior_latents, prior_pool)
    )
    faithful_jobs, faithful_scaling = _scaled_jobs(
        {METHOD_CHEMSPACE_OFFICIAL: faithful_directions},
        None,
        scales=(SCALE_NATIVE,),
    )
    prior_eval = _rau().sample_prior_latents(
        OFFICIAL_EVALUATION_SEEDS,
        256,
        seed=OFFICIAL_PRIOR_EVALUATION_SEED,
        verify_known_torch_2_5_1=False,
    ).detach().cpu().numpy()
    faithful_candidates, faithful_timing = decode_direction_jobs(
        faithful_jobs,
        prior_eval,
        bundle=bundle,
        loaded_model=loaded_model,
        output_dir=METHOD_FAITHFUL_DIR,
        seed_identity=None,
        batch_size=batch_size,
    )
    faithful_seed, faithful_alpha, faithful_summary = build_protocol_metrics(
        faithful_candidates,
        full_training_property_ranges=full_ranges,
        output_dir=METHOD_FAITHFUL_DIR,
    )
    faithful_scored = pd.read_csv(
        METHOD_FAITHFUL_DIR / "local_traversal_candidates.csv"
    )
    faithful_confound_seed, faithful_confound_summary = selfies_confound_audit(
        faithful_scored, output_dir=METHOD_FAITHFUL_DIR
    )
    _atomic_csv(faithful_scaling, METHOD_FAITHFUL_DIR / "direction_scaling.csv")

    paired_directions, paired_training, paired_selection, train_std = (
        fit_paired_directions(inputs)
    )
    paired_jobs, paired_scaling = _scaled_jobs(
        paired_directions,
        train_std,
        scales=(SCALE_NATIVE, SCALE_STANDARDIZED),
    )
    paired_identity = legacy._seed_identity_table(inputs)
    paired_candidates, paired_timing = decode_direction_jobs(
        paired_jobs,
        inputs.registry.seed_latents,
        bundle=bundle,
        loaded_model=loaded_model,
        output_dir=PAIRED_DIR,
        seed_identity=paired_identity,
        batch_size=batch_size,
    )
    paired_local_seed, paired_local_alpha, paired_local_summary = build_protocol_metrics(
        paired_candidates,
        full_training_property_ranges=full_ranges,
        output_dir=PAIRED_DIR,
    )
    paired_local_scored = pd.read_csv(
        PAIRED_DIR / "local_traversal_candidates.csv"
    )
    paired_local_confound_seed, paired_local_confound_summary = (
        selfies_confound_audit(paired_local_scored, output_dir=PAIRED_DIR)
    )
    _atomic_csv(paired_scaling, PAIRED_DIR / "direction_scaling.csv")

    paired_displacement_candidates, paired_displacement_timing = (
        decode_direction_jobs(
            paired_jobs,
            inputs.registry.seed_latents,
            bundle=bundle,
            loaded_model=loaded_model,
            output_dir=PAIRED_DISPLACEMENT_DIR,
            seed_identity=paired_identity,
            batch_size=batch_size,
            alpha_grid=PAIRED_DISPLACEMENT_ALPHAS,
        )
    )
    paired_seed, paired_alpha, paired_summary = build_protocol_metrics(
        paired_displacement_candidates,
        full_training_property_ranges=full_ranges,
        output_dir=PAIRED_DISPLACEMENT_DIR,
    )
    paired_scored = pd.read_csv(
        PAIRED_DISPLACEMENT_DIR / "local_traversal_candidates.csv"
    )
    paired_confound_seed, paired_confound_summary = selfies_confound_audit(
        paired_scored, output_dir=PAIRED_DISPLACEMENT_DIR
    )
    _atomic_csv(
        paired_scaling, PAIRED_DISPLACEMENT_DIR / "direction_scaling.csv"
    )
    matched_pairs, matched_summary = matched_seed_similarity(
        paired_displacement_candidates,
        output_dir=PAIRED_DISPLACEMENT_DIR,
    )
    matched_displacement_pairs, matched_displacement_summary = (
        matched_latent_displacement(paired_scored)
    )
    constrained_per_seed, constrained_summary = constrained_design_tables(
        paired_scored
    )
    cosines = direction_cosines(
        paired_directions, train_std, output_dir=PAIRED_DISPLACEMENT_DIR
    )
    create_figures(
        faithful_alpha,
        faithful_summary,
        paired_alpha,
        paired_summary,
    )
    return {
        "faithful_training": faithful_training,
        "faithful_selection": faithful_selection,
        "faithful_prior_pool": prior_pool,
        "faithful_prior_attempts": prior_attempts,
        "faithful_candidates": faithful_candidates,
        "faithful_per_seed": faithful_seed,
        "faithful_per_alpha": faithful_alpha,
        "faithful_summary": faithful_summary,
        "faithful_timing": faithful_timing,
        "faithful_confound_per_seed": faithful_confound_seed,
        "faithful_confound_summary": faithful_confound_summary,
        "paired_training": paired_training,
        "paired_selection": paired_selection,
        "paired_local_candidates": paired_candidates,
        "paired_local_per_seed": paired_local_seed,
        "paired_local_per_alpha": paired_local_alpha,
        "paired_local_summary": paired_local_summary,
        "paired_local_timing": paired_timing,
        "paired_local_confound_per_seed": paired_local_confound_seed,
        "paired_local_confound_summary": paired_local_confound_summary,
        "paired_candidates": paired_displacement_candidates,
        "paired_per_seed": paired_seed,
        "paired_per_alpha": paired_alpha,
        "paired_summary": paired_summary,
        "paired_timing": paired_displacement_timing,
        "paired_confound_per_seed": paired_confound_seed,
        "paired_confound_summary": paired_confound_summary,
        "matched_pairs": matched_pairs,
        "matched_summary": matched_summary,
        "matched_displacement_pairs": matched_displacement_pairs,
        "matched_displacement_summary": matched_displacement_summary,
        "constrained_per_seed": constrained_per_seed,
        "constrained_summary": constrained_summary,
        "cosines": cosines,
    }


def verify_protocol_outputs(
    *, output_root: Path = OUTPUT_ROOT
) -> pd.DataFrame:
    """Deterministic, result-level acceptance checks for the notebook test."""

    checks = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    faithful = pd.read_csv(
        output_root / "method_faithful" / "local_traversal_candidates.csv"
    )
    paired = pd.read_csv(
        output_root / "paired_displacement" / "local_traversal_candidates.csv"
    )
    paired_local = pd.read_csv(
        output_root / "paired_local" / "local_traversal_candidates.csv"
    )
    faithful_summary = pd.read_csv(
        output_root / "method_faithful" / "compact_summary.csv"
    )
    paired_summary = pd.read_csv(
        output_root / "paired_displacement" / "compact_summary.csv"
    )
    direction_training = pd.read_csv(
        output_root / "method_faithful" / "direction_training_summary.csv"
    )
    paired_training = pd.read_csv(
        output_root / "paired_local" / "direction_training_summary.csv"
    )
    displacement_pairs = pd.read_csv(
        output_root
        / "paired_displacement"
        / "matched_latent_displacement_pairs.csv"
    )
    constrained = pd.read_csv(
        output_root / "paired_displacement" / "constrained_design_summary.csv"
    )
    record(
        "faithful candidate count",
        len(faithful) == 6 * 200 * 21,
        len(faithful),
    )
    record(
        "paired candidate count",
        len(paired) == 6 * 2 * 2 * 50 * 21,
        len(paired),
    )
    record(
        "paired unit-grid diagnostic candidate count",
        len(paired_local) == 6 * 2 * 2 * 50 * 21,
        len(paired_local),
    )
    record(
        "exact local alpha grid",
        all(
            np.allclose(
                group.sort_values("alpha_index")["alpha"].to_numpy(),
                LOCAL_ALPHAS,
                rtol=0.0,
                atol=1e-12,
            )
            for _, group in faithful.groupby(["property", "seed_id"])
        )
        and all(
            np.allclose(
                group.sort_values("alpha_index")["alpha"].to_numpy(),
                PAIRED_DISPLACEMENT_ALPHAS,
                rtol=0.0,
                atol=1e-12,
            )
            for _, group in paired.groupby(
                ["method", "scale", "property", "seed_id"]
            )
        ),
        "faithful [-1,1] and paired [-150,150], 21 points including zero",
    )
    record(
        "synthesis scores attached",
        {"sa_score", "np_likeness_score"}.issubset(faithful.columns)
        and {"sa_score", "np_likeness_score"}.issubset(paired.columns),
        "RDKit Contrib SA_Score and NP_Score",
    )
    record(
        "exact matched displacement pairs",
        len(displacement_pairs) == 6 * 2 * 50 * 21,
        len(displacement_pairs),
    )
    record(
        "published constrained-design cutoffs only",
        set(constrained["seed_similarity_cutoff"].unique())
        == set(PUBLISHED_SIMILARITY_CUTOFFS)
        and constrained["median_eligible_candidates"].between(0, 21).all(),
        sorted(constrained["seed_similarity_cutoff"].unique()),
    )
    record(
        "faithful prior oracle pool",
        direction_training["oracle_label_pool"].eq(400).all(),
        direction_training["oracle_label_pool"].tolist(),
    )
    record(
        "paired label accounting",
        paired_training[
            paired_training["method"].eq(METHOD_CHEMSPACE_PAIRED)
        ]["labels_inspected_for_ranking"]
        .eq(635_483)
        .all(),
        paired_training["labels_inspected_for_ranking"].dropna().unique(),
    )
    record(
        "actual SSR columns",
        {
            "ssr_official_code_compatible",
            "ssr_expected_all_valid",
            "rsr_local_official_code_compatible",
            "rsr_global_official_code_compatible",
        }.issubset(faithful_summary.columns)
        and {
            "ssr_official_code_compatible",
            "ssr_expected_all_valid",
        }.issubset(paired_summary.columns),
        faithful_summary.columns.tolist(),
    )
    record(
        "validity bounded",
        faithful_summary["valid_fraction"].between(0, 1).all()
        and paired_summary["valid_fraction"].between(0, 1).all(),
        "all summary rows",
    )
    record(
        "canonical uniqueness bounded",
        faithful_summary["canonical_unique_fraction_total"].between(0, 1).all()
        and paired_summary["canonical_unique_fraction_total"].between(0, 1).all(),
        "all summary rows",
    )
    record(
        "all center points present",
        faithful.groupby(["property", "seed_id"])["alpha"]
        .apply(lambda values: np.isclose(values, 0.0).sum() == 1)
        .all()
        and paired.groupby(["method", "scale", "property", "seed_id"])["alpha"]
        .apply(lambda values: np.isclose(values, 0.0).sum() == 1)
        .all(),
        "one alpha=0 per path",
    )
    record(
        "test rejects old broad-grid success surrogate",
        "success_fraction" not in faithful_summary.columns
        and "success_fraction" not in paired_summary.columns,
        "no any-improving-candidate metric",
    )
    return pd.DataFrame(checks)
