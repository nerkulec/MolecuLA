"""Task-aligned ChemSpacE baseline for the frozen MolecuLA latent interface.

The implementation is intentionally narrow: extreme-group linear SVM
directions, the frozen historical traversal registry, cached MolecuLA inputs,
and reviewer-facing comparison tables. It does not train or alter the VAE.
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
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
PROBEVAE = HERE.parent
PROJECT_ROOT = PROBEVAE.parent
REBUTTAL_OUTPUTS = PROBEVAE / "rebuttal_outputs"
OUTPUTS = HERE / "outputs"

PROPERTIES = (
    "FractionCSP3",
    "HBA",
    "TPSA",
    "cLogP",
    "BertzCT",
    "HeavyAtomCount",
)
CONFOUNDS = (
    "selfies_len_tokens",
    "branch_token_count",
    "ring_token_count",
    "token_entropy",
)
PANEL_COLUMNS = (
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "cLogP",
    "TPSA",
    "HBD",
    "HBA",
    "NumRotatableBonds",
    "RingCount",
    "AromaticRingCount",
    "FractionCSP3",
    "NumSpiroAtoms",
    "NumBridgeheadAtoms",
    "BertzCT",
    "QED",
    *CONFOUNDS,
)
ALPHAS = np.linspace(-150.0, 150.0, 100)
METHOD_CHEMSPACE = "ChemSpacE extreme-SVM"
METHOD_OURS = "MolecuLA raw OLS (seed-excluded)"
SIMILARITY_CUTOFFS = (0.0, 0.2, 0.4, 0.6)
N_EXTREME_PER_CLASS = 200
FIT_RATIO = 0.70
SEED = 42


def _rebuttal_utils() -> Any:
    """Import the existing utility module under its pickle-compatible name."""

    probevae_text = str(PROBEVAE)
    if probevae_text not in sys.path:
        sys.path.insert(0, probevae_text)
    return importlib.import_module("rebuttal_analysis_utils")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedInputs:
    latents: np.ndarray
    panel: np.ndarray
    split_codes: np.ndarray
    registry: Any
    panel_index: Mapping[str, int]


def load_cached_inputs() -> CachedInputs:
    """Load and validate the existing row-aligned MolecuLA analysis caches."""

    rau = _rebuttal_utils()
    latents = np.load(REBUTTAL_OUTPUTS / "latents_mu.npy", mmap_mode="r")
    panel = np.load(
        REBUTTAL_OUTPUTS / "numeric_property_confound_panel.npy", mmap_mode="r"
    )
    split_codes = np.load(REBUTTAL_OUTPUTS / "split_codes.npy", mmap_mode="r")
    if latents.shape != (794_403, 256) or latents.dtype != np.float32:
        raise AssertionError(f"Unexpected latent cache: {latents.shape}, {latents.dtype}")
    if panel.shape != (794_403, len(PANEL_COLUMNS)):
        raise AssertionError(f"Unexpected numeric panel: {panel.shape}")
    if split_codes.shape != (794_403,) or split_codes.dtype != np.uint8:
        raise AssertionError(
            f"Unexpected split cache: {split_codes.shape}, {split_codes.dtype}"
        )
    registry = rau.make_traversal_registry(
        latents, n_seeds=50, seed=SEED, alphas=ALPHAS
    )
    expected_rows = np.array(
        [
            319669, 407691, 357774, 70896, 618350, 771121, 511472, 556492,
            219855, 354054, 281634, 538540, 343969, 53955, 624425, 418215,
            397466, 602221, 160039, 667072, 440554, 653586, 131256, 553961,
            705329, 709496, 348626, 73191, 682034, 736200, 501789, 519962,
            584456, 357837, 775000, 569940, 50695, 604624, 352240, 681818,
            433279, 180515, 68271, 620859, 614795, 145013, 294552, 101769,
            74811, 657462,
        ],
        dtype=np.int64,
    )
    if not np.array_equal(registry.row_indices, expected_rows):
        raise AssertionError("Historical 50-seed registry changed")
    if not np.array_equal(registry.alphas, ALPHAS):
        raise AssertionError("Historical alpha grid changed")
    return CachedInputs(
        latents=latents,
        panel=panel,
        split_codes=split_codes,
        registry=registry,
        panel_index={name: index for index, name in enumerate(PANEL_COLUMNS)},
    )


def seed_split_audit(inputs: CachedInputs) -> pd.DataFrame:
    names = {0: "train", 1: "validation", 2: "test"}
    rows = []
    for seed_id, row_index in enumerate(inputs.registry.row_indices):
        code = int(inputs.split_codes[row_index])
        rows.append(
            {
                "seed_id": seed_id,
                "seed_row_index": int(row_index),
                "split_code": code,
                "split": names[code],
                "excluded_from_chemspace_fit": True,
            }
        )
    return pd.DataFrame(rows)


def _rank_extremes(
    global_rows: np.ndarray,
    scores: np.ndarray,
    n_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank scores with global row ID as an explicit deterministic tie-break."""

    if len(global_rows) != len(scores):
        raise ValueError("global_rows and scores must be row aligned")
    if 2 * n_per_class > len(global_rows):
        raise ValueError("Extreme groups would overlap")
    high_order = np.lexsort((global_rows, -scores))
    low_order = np.lexsort((global_rows, scores))
    high = global_rows[high_order[:n_per_class]]
    low = global_rows[low_order[:n_per_class]]
    if np.intersect1d(high, low).size:
        raise AssertionError("High and low extreme groups overlap")
    return high, low


def fit_chemspace_boundary(
    latents: np.ndarray,
    scores: np.ndarray,
    eligible_rows: np.ndarray,
    *,
    property_name: str,
    n_per_class: int = N_EXTREME_PER_CLASS,
    fit_ratio: float = FIT_RATIO,
    seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    """Fit the published ChemSpacE extreme-group linear-SVM normal."""

    total_start = time.perf_counter()
    eligible_rows = np.asarray(eligible_rows, dtype=np.int64)
    eligible_scores = np.asarray(scores[eligible_rows], dtype=np.float64)
    if not np.isfinite(eligible_scores).all():
        raise ValueError(f"{property_name} contains non-finite eligible labels")
    high_rows, low_rows = _rank_extremes(
        eligible_rows, eligible_scores, n_per_class
    )
    rng = np.random.default_rng(int(seed))
    high_rows = rng.permutation(high_rows)
    low_rows = rng.permutation(low_rows)
    n_fit = int(math.floor(n_per_class * fit_ratio))
    if not 0 < n_fit < n_per_class:
        raise ValueError("fit_ratio must retain both fit and validation rows")
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
    classifier = SVC(kernel="linear", C=1.0)
    fit_start = time.perf_counter()
    classifier.fit(np.asarray(latents[fit_rows]), fit_labels)
    fit_seconds = time.perf_counter() - fit_start
    coefficient = np.asarray(classifier.coef_[0], dtype=np.float64)
    coefficient_norm = float(np.linalg.norm(coefficient))
    if not np.isfinite(coefficient_norm) or coefficient_norm == 0:
        raise AssertionError(f"{property_name}: invalid SVM normal")
    direction = coefficient / coefficient_norm
    high_decision = classifier.decision_function(np.asarray(latents[high_rows]))
    low_decision = classifier.decision_function(np.asarray(latents[low_rows]))
    if np.median(high_decision) <= np.median(low_decision):
        raise AssertionError(f"{property_name}: SVM normal orientation is reversed")

    selection_rows = []
    fit_set = set(int(value) for value in fit_rows)
    for class_name, label, rows in (
        ("high", 1, high_rows),
        ("low", 0, low_rows),
    ):
        for rank_after_shuffle, row_index in enumerate(rows):
            selection_rows.append(
                {
                    "property": property_name,
                    "extreme_class": class_name,
                    "class_label": label,
                    "rank_after_seeded_shuffle": rank_after_shuffle,
                    "global_row_index": int(row_index),
                    "property_value": float(scores[row_index]),
                    "used_for_svm_fit": int(row_index) in fit_set,
                }
            )
    details = {
        "property": property_name,
        "eligible_training_rows": int(len(eligible_rows)),
        "high_extreme_rows": int(n_per_class),
        "low_extreme_rows": int(n_per_class),
        "cached_labels_consumed": int(2 * n_per_class),
        "incremental_training_oracle_calls": 0,
        "svm_fit_rows": int(len(fit_rows)),
        "svm_validation_rows": int(len(validation_rows)),
        "svm_C": 1.0,
        "latent_standardization": False,
        "svm_fit_accuracy": float(classifier.score(latents[fit_rows], fit_labels)),
        "svm_validation_accuracy": float(
            classifier.score(latents[validation_rows], validation_labels)
        ),
        "support_vectors": int(classifier.n_support_.sum()),
        "coefficient_l2_before_normalization": coefficient_norm,
        "direction_l2": float(np.linalg.norm(direction)),
        "direction_sha256": _array_sha256(direction),
        "svm_fit_seconds": fit_seconds,
        "direction_learning_total_seconds": time.perf_counter() - total_start,
        "sklearn_version": importlib.import_module("sklearn").__version__,
    }
    return direction, details, pd.DataFrame(selection_rows)


def fit_all_chemspace_directions(
    inputs: CachedInputs,
    *,
    output_dir: Path = OUTPUTS,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Fit six deterministic ChemSpacE directions and persist compact outputs."""

    train_rows = np.flatnonzero(np.asarray(inputs.split_codes) == 0)
    eligible_rows = np.setdiff1d(
        train_rows, inputs.registry.row_indices, assume_unique=False
    )
    directions: dict[str, np.ndarray] = {}
    detail_rows = []
    selection_frames = []
    for property_name in PROPERTIES:
        scores = inputs.panel[:, inputs.panel_index[property_name]]
        direction, details, selected = fit_chemspace_boundary(
            inputs.latents,
            scores,
            eligible_rows,
            property_name=property_name,
        )
        directions[property_name] = direction
        detail_rows.append(details)
        selection_frames.append(selected)
    training = pd.DataFrame(detail_rows)
    selected_rows = pd.concat(selection_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "chemspace_directions.npz",
        **{name: directions[name] for name in PROPERTIES},
    )
    _atomic_csv(training, output_dir / "direction_training_summary.csv")
    _atomic_csv(selected_rows, output_dir / "direction_training_rows.csv")
    _atomic_csv(seed_split_audit(inputs), output_dir / "seed_split_audit.csv")
    return directions, training, selected_rows


def fit_seed_excluded_raw_ols_directions(
    inputs: CachedInputs,
    *,
    output_dir: Path = OUTPUTS,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Refit the original multi-target OLS after excluding all 50 seeds."""

    rau = _rebuttal_utils()
    selector = np.asarray(inputs.split_codes) == 0
    selector = selector.copy()
    selector[inputs.registry.row_indices] = False
    property_indices = [inputs.panel_index[name] for name in PROPERTIES]
    targets = np.asarray(inputs.panel[:, property_indices])
    start = time.perf_counter()
    model = rau.fit_chunked_standardized_ols(
        inputs.latents,
        targets,
        selector=selector,
        feature_names=tuple(f"z{index}" for index in range(inputs.latents.shape[1])),
        target_names=PROPERTIES,
    )
    elapsed = time.perf_counter() - start
    units = model.unit_directions()
    directions = {
        property_name: rau.unit_directions(direction)
        for property_name, direction in zip(PROPERTIES, units)
    }
    historical = load_reference_directions()
    rows = []
    for property_name in PROPERTIES:
        direction = directions[property_name]
        rows.append(
            {
                "property": property_name,
                "eligible_training_rows": int(selector.sum()),
                "excluded_seed_rows_total": 50,
                "excluded_seed_rows_from_training_split": int(
                    np.count_nonzero(
                        np.asarray(inputs.split_codes[inputs.registry.row_indices])
                        == 0
                    )
                ),
                "fit_method": "chunked StandardScaler + multi-target LinearRegression",
                "direction_l2": float(np.linalg.norm(direction)),
                "direction_sha256": _array_sha256(direction),
                "cosine_vs_historical_raw_ols": float(
                    direction @ historical[f"raw::{property_name}"]
                ),
                "joint_fit_seconds_all_six_targets": elapsed,
            }
        )
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "raw_ols_seed_excluded_directions.npz",
        **{name: directions[name] for name in PROPERTIES},
    )
    _atomic_csv(summary, output_dir / "raw_ols_seed_excluded_training.csv")
    return directions, summary


def load_reference_directions() -> dict[str, np.ndarray]:
    """Load the existing OLS raw/residual/confound directions without refitting."""

    rau = _rebuttal_utils()
    payload = joblib.load(REBUTTAL_OUTPUTS / "residualization_models.joblib")
    latent_model = payload["joint_latent_standardized_ols"]
    units = latent_model.unit_directions()
    return {
        name: rau.unit_directions(direction)
        for name, direction in zip(latent_model.target_names, units)
    }


def direction_cosine_table(
    chemspace: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    confound_matrix = np.column_stack(
        [reference[f"confound::{name}"] for name in CONFOUNDS]
    )
    q_basis, singular_values, _ = np.linalg.svd(
        confound_matrix, full_matrices=False
    )
    tolerance = (
        np.finfo(np.float64).eps
        * max(confound_matrix.shape)
        * singular_values[0]
    )
    q_basis = q_basis[:, singular_values > tolerance]
    rows = []
    for property_name in PROPERTIES:
        direction = chemspace[property_name]
        row = {
            "property": property_name,
            "cosine_chemspace_vs_raw_ols": float(
                direction @ reference[f"raw::{property_name}"]
            ),
            "cosine_chemspace_vs_residual_ols": float(
                direction @ reference[f"residual::{property_name}"]
            ),
            "chemspace_confound_subspace_overlap": float(
                np.linalg.norm(q_basis.T @ direction)
            ),
        }
        for confound in CONFOUNDS:
            row[f"cosine_chemspace_vs_{confound}"] = float(
                direction @ reference[f"confound::{confound}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _seed_identity_table(inputs: CachedInputs) -> pd.DataFrame:
    columns = [
        "seed_id",
        "seed_row_index",
        "seed_canonical_smiles",
        "seed_scaffold",
    ]
    identity = pd.read_parquet(
        REBUTTAL_OUTPUTS / "reviewer_traversal_identity_metrics.parquet",
        columns=columns,
    ).drop_duplicates()
    identity = identity.sort_values("seed_id").reset_index(drop=True)
    if len(identity) != 50:
        raise AssertionError("Expected exactly 50 seed identities")
    if not np.array_equal(
        identity["seed_row_index"].to_numpy(), inputs.registry.row_indices
    ):
        raise AssertionError("Cached seed identities do not match the registry")
    for property_name in PROPERTIES:
        identity[f"seed::{property_name}"] = inputs.panel[
            inputs.registry.row_indices, inputs.panel_index[property_name]
        ]
    return identity


def _attach_structure_metrics(
    frame: pd.DataFrame,
    seed_identity: pd.DataFrame,
) -> pd.DataFrame:
    rau = _rebuttal_utils()
    DataStructs = importlib.import_module("rdkit.DataStructs")
    seed_lookup = seed_identity.set_index("seed_id").to_dict(orient="index")
    seed_features = {
        seed_id: rau._structure_identity_features(row["seed_canonical_smiles"])
        for seed_id, row in seed_lookup.items()
    }
    valid = rau.valid_canonical_mask(frame)
    generated = {
        str(smiles): rau._structure_identity_features(str(smiles))
        for smiles in frame.loc[valid, "canonical_smiles"].dropna().unique()
        if str(smiles).strip()
    }
    seed_canonical = []
    seed_scaffold = []
    generated_scaffold = []
    similarity = []
    scaffold_evaluable = []
    scaffold_retained = []
    for row in frame.itertuples(index=False):
        seed_id = int(row.seed_id)
        seed_feature = seed_features[seed_id]
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
        if generated_feature is not None:
            similarity.append(
                float(
                    DataStructs.TanimotoSimilarity(
                        seed_feature["fingerprint"],
                        generated_feature["fingerprint"],
                    )
                )
            )
        else:
            similarity.append(float("nan"))
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
    enriched = frame.copy()
    enriched["seed_canonical_smiles"] = seed_canonical
    enriched["seed_scaffold"] = seed_scaffold
    enriched["generated_scaffold"] = generated_scaffold
    enriched["seed_similarity_tanimoto"] = similarity
    enriched["scaffold_evaluable"] = scaffold_evaluable
    enriched["scaffold_retained"] = pd.array(
        scaffold_retained, dtype="boolean"
    )
    enriched["changed_from_seed"] = (
        rau.valid_canonical_mask(enriched)
        & enriched["canonical_smiles"].astype("string").ne(
            enriched["seed_canonical_smiles"].astype("string")
        )
    )
    return enriched


def load_frozen_model(*, device: str = "cuda") -> tuple[Any, Any]:
    """Run the existing compatibility gate and load the frozen model once."""

    rau = _rebuttal_utils()
    bundle = rau.validate_normalized_dataset(
        split_cache_path=REBUTTAL_OUTPUTS / "split_codes.npy"
    )
    loaded_model = rau.load_verified_ar_model(bundle, device=device)
    return bundle, loaded_model


def _decode_direction_set(
    inputs: CachedInputs,
    directions: Mapping[str, np.ndarray],
    *,
    bundle: Any,
    loaded_model: Any,
    method: str,
    shard_prefix: str,
    aggregate_filename: str,
    timing_filename: str,
    output_dir: Path = OUTPUTS,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode six 5,000-candidate direction traversals, resumably."""

    rau = _rebuttal_utils()
    seed_identity = _seed_identity_table(inputs)
    frames = []
    timing_rows = []
    prior_timing_path = output_dir / timing_filename
    prior_timing = (
        pd.read_csv(prior_timing_path).set_index("property")
        if prior_timing_path.exists()
        else pd.DataFrame()
    )
    for property_name in PROPERTIES:
        path = output_dir / f"{shard_prefix}__{property_name}.csv"
        direction_hash = _array_sha256(directions[property_name])
        if path.exists():
            frame = pd.read_csv(path)
            valid_cache = (
                len(frame) == 5_000
                and frame["method"].eq(method).all()
                and frame["property"].eq(property_name).all()
                and frame["direction_sha256"].eq(direction_hash).all()
                and np.array_equal(
                    frame["seed_row_index"].drop_duplicates().to_numpy(),
                    inputs.registry.row_indices,
                )
                and np.allclose(
                    frame.loc[frame["seed_id"].eq(0), "alpha"].to_numpy(),
                    ALPHAS,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            if not valid_cache:
                raise AssertionError(f"Traversal cache failed validation: {path}")
            frames.append(frame)
            previous = (
                prior_timing.loc[property_name]
                if not prior_timing.empty
                and property_name in prior_timing.index
                else None
            )
            timing_rows.append(
                {
                    "property": property_name,
                    "loaded_from_cache": True,
                    "candidate_count": 5_000,
                    "decoder_seconds": (
                        float(previous["decoder_seconds"])
                        if previous is not None
                        and pd.notna(previous["decoder_seconds"])
                        else float("nan")
                    ),
                    "descriptor_and_structure_seconds": (
                        float(previous["descriptor_and_structure_seconds"])
                        if previous is not None
                        and pd.notna(
                            previous["descriptor_and_structure_seconds"]
                        )
                        else float("nan")
                    ),
                    "valid_property_oracle_calls": int(
                        frame["is_rdkit_valid"].astype(bool).sum()
                    ),
                }
            )
            continue

        grid = rau.traversal_latent_grid(
            inputs.registry, directions[property_name]
        )
        decode_start = time.perf_counter()
        decoded = rau.decode_latents_adaptive(
            loaded_model,
            grid,
            bundle.id2tok,
            batch_size=batch_size,
        )
        decode_seconds = time.perf_counter() - decode_start
        descriptor_start = time.perf_counter()
        feature_rows = [
            rau.generated_features(row, (*PROPERTIES,)) for row in decoded
        ]
        frame = pd.DataFrame(decoded)
        features = pd.DataFrame(feature_rows)
        for column in (
            "canonical_smiles",
            "is_rdkit_valid",
            *PROPERTIES,
            *CONFOUNDS,
        ):
            frame[column] = features[column]
        frame["is_rdkit_valid"] = rau.valid_canonical_mask(frame)
        frame.insert(0, "alpha", np.tile(ALPHAS, 50))
        frame.insert(
            0, "seed_row_index", np.repeat(inputs.registry.row_indices, 100)
        )
        frame.insert(0, "seed_id", np.repeat(np.arange(50), 100))
        frame.insert(0, "direction_sha256", direction_hash)
        frame.insert(0, "direction_l2", float(np.linalg.norm(directions[property_name])))
        frame.insert(0, "latent_displacement", np.abs(frame["alpha"]))
        frame.insert(0, "property", property_name)
        frame.insert(0, "method", method)
        frame = _attach_structure_metrics(frame, seed_identity)
        frame["seed_property_value"] = frame["seed_id"].map(
            seed_identity.set_index("seed_id")[f"seed::{property_name}"]
        )
        frame["property_delta"] = (
            frame[property_name] - frame["seed_property_value"]
        )
        frame["expected_signed_improvement"] = (
            np.sign(frame["alpha"]) * frame["property_delta"]
        )
        descriptor_seconds = time.perf_counter() - descriptor_start
        _atomic_csv(frame, path)
        frames.append(frame)
        timing_rows.append(
            {
                "property": property_name,
                "loaded_from_cache": False,
                "candidate_count": 5_000,
                "decoder_seconds": decode_seconds,
                "descriptor_and_structure_seconds": descriptor_seconds,
                "valid_property_oracle_calls": int(
                    frame["is_rdkit_valid"].astype(bool).sum()
                ),
            }
        )
    all_candidates = pd.concat(frames, ignore_index=True)
    timing = pd.DataFrame(timing_rows)
    _atomic_csv(all_candidates, output_dir / aggregate_filename)
    _atomic_csv(timing, output_dir / timing_filename)
    return all_candidates, timing


def decode_chemspace_directions(
    inputs: CachedInputs,
    directions: Mapping[str, np.ndarray],
    *,
    bundle: Any,
    loaded_model: Any,
    output_dir: Path = OUTPUTS,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _decode_direction_set(
        inputs,
        directions,
        bundle=bundle,
        loaded_model=loaded_model,
        method=METHOD_CHEMSPACE,
        shard_prefix="decoded_candidates",
        aggregate_filename="chemspace_traversal_candidates.csv",
        timing_filename="decode_timing_chemspace.csv",
        output_dir=output_dir,
        batch_size=batch_size,
    )


def decode_seed_excluded_raw_ols_directions(
    inputs: CachedInputs,
    directions: Mapping[str, np.ndarray],
    *,
    bundle: Any,
    loaded_model: Any,
    output_dir: Path = OUTPUTS,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _decode_direction_set(
        inputs,
        directions,
        bundle=bundle,
        loaded_model=loaded_model,
        method=METHOD_OURS,
        shard_prefix="decoded_raw_ols_seed_excluded",
        aggregate_filename="raw_ols_seed_excluded_traversal_candidates.csv",
        timing_filename="decode_timing_raw_ols_seed_excluded.csv",
        output_dir=output_dir,
        batch_size=batch_size,
    )


def load_raw_ols_candidates(inputs: CachedInputs) -> pd.DataFrame:
    """Load historical raw-OLS decodes (diagnostic only, not strict baseline)."""

    rau = _rebuttal_utils()
    reference_directions = load_reference_directions()
    frame = pd.read_parquet(
        REBUTTAL_OUTPUTS / "reviewer_traversal_identity_metrics.parquet"
    )
    frame = frame[frame["direction_kind"].eq("raw_property")].copy()
    frame.rename(columns={"focus_property": "property"}, inplace=True)
    frame.insert(0, "method", "MolecuLA raw OLS (historical cached)")
    frame["latent_displacement"] = frame["alpha"].abs()
    frame["direction_l2"] = 1.0
    frame["direction_sha256"] = frame["property"].map(
        {
            property_name: _array_sha256(
                reference_directions[f"raw::{property_name}"]
            )
            for property_name in PROPERTIES
        }
    )
    frame["is_rdkit_valid"] = rau.valid_canonical_mask(frame)
    frame["changed_from_seed"] = (
        frame["is_rdkit_valid"]
        & frame["canonical_smiles"].astype("string").ne(
            frame["seed_canonical_smiles"].astype("string")
        )
    )
    for property_name in PROPERTIES:
        selector = frame["property"].eq(property_name)
        seed_values = inputs.panel[
            frame.loc[selector, "seed_row_index"].to_numpy(dtype=np.int64),
            inputs.panel_index[property_name],
        ]
        frame.loc[selector, "seed_property_value"] = seed_values
        frame.loc[selector, "property_delta"] = (
            frame.loc[selector, property_name].to_numpy(dtype=float)
            - seed_values
        )
    frame["expected_signed_improvement"] = (
        np.sign(frame["alpha"]) * frame["property_delta"]
    )
    return frame


COMPARISON_COLUMNS = (
    "method",
    "property",
    "seed_id",
    "seed_row_index",
    "alpha_index",
    "alpha",
    "latent_displacement",
    "direction_l2",
    "direction_sha256",
    "decoded_selfies",
    "decoded_smiles",
    "canonical_smiles",
    "is_rdkit_valid",
    "seed_canonical_smiles",
    "changed_from_seed",
    "seed_similarity_tanimoto",
    "seed_scaffold",
    "generated_scaffold",
    "scaffold_evaluable",
    "scaffold_retained",
    *PROPERTIES,
    *CONFOUNDS,
    "seed_property_value",
    "property_delta",
    "expected_signed_improvement",
)


def combine_candidates(
    raw: pd.DataFrame,
    chemspace: pd.DataFrame,
    *,
    output_dir: Path = OUTPUTS,
) -> pd.DataFrame:
    normalized_frames = []
    for name, source in (("raw", raw), ("ChemSpacE", chemspace)):
        frame = source.copy()
        alpha_values = frame["alpha"].to_numpy(dtype=np.float64)
        alpha_index = np.abs(
            alpha_values[:, None] - ALPHAS[None, :]
        ).argmin(axis=1)
        if not np.allclose(
            alpha_values, ALPHAS[alpha_index], rtol=0.0, atol=1e-12
        ):
            raise AssertionError(f"{name} candidates do not use the fixed alpha grid")
        frame["alpha_index"] = alpha_index
        frame["alpha"] = ALPHAS[alpha_index]
        frame["latent_displacement"] = np.abs(frame["alpha"])
        missing = set(COMPARISON_COLUMNS) - set(frame.columns)
        if missing:
            raise AssertionError(f"{name} candidates miss columns: {sorted(missing)}")
        normalized_frames.append(frame)
    combined = pd.concat(
        [frame[list(COMPARISON_COLUMNS)] for frame in normalized_frames],
        ignore_index=True,
    )
    if len(combined) != 60_000:
        raise AssertionError(f"Expected 60,000 candidates, got {len(combined)}")
    _atomic_csv(combined, output_dir / "full_traversal_candidates.csv")
    return combined


def _finite_fraction(values: pd.Series, predicate: Any) -> tuple[float, int]:
    finite = values[np.isfinite(values)]
    return (
        float(predicate(finite).mean()) if len(finite) else float("nan"),
        int(len(finite)),
    )


def traversal_tables(
    candidates: pd.DataFrame,
    *,
    output_dir: Path = OUTPUTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build signed-alpha, per-seed, and compact traversal tables."""

    rau = _rebuttal_utils()
    per_alpha_rows = []
    per_seed_rows = []
    for (method, property_name), group in candidates.groupby(
        ["method", "property"], sort=False
    ):
        for alpha, alpha_group in group.groupby("alpha", sort=True):
            valid = alpha_group["is_rdkit_valid"].astype(bool)
            valid_group = alpha_group[valid]
            scaffold = valid_group.loc[
                valid_group["scaffold_evaluable"].astype(bool),
                "scaffold_retained",
            ].astype(bool)
            per_alpha_rows.append(
                {
                    "method": method,
                    "property": property_name,
                    "alpha": float(alpha),
                    "latent_displacement": abs(float(alpha)),
                    "requested_candidates": int(len(alpha_group)),
                    "valid_count": int(valid.sum()),
                    "valid_fraction": float(valid.mean()),
                    "changed_from_seed_count": int(
                        alpha_group["changed_from_seed"].astype(bool).sum()
                    ),
                    "changed_from_seed_fraction_all": float(
                        alpha_group["changed_from_seed"].astype(bool).mean()
                    ),
                    "changed_from_seed_fraction_valid": (
                        float(
                            valid_group["changed_from_seed"].astype(bool).mean()
                        )
                        if len(valid_group)
                        else float("nan")
                    ),
                    "median_property": float(valid_group[property_name].median()),
                    "median_property_delta": float(
                        valid_group["property_delta"].median()
                    ),
                    "median_expected_signed_improvement": float(
                        valid_group["expected_signed_improvement"].median()
                    ),
                    "expected_sign_fraction_valid": (
                        float(
                            (
                                valid_group["expected_signed_improvement"] > 0
                            ).mean()
                        )
                        if len(valid_group)
                        else float("nan")
                    ),
                    "median_seed_similarity": float(
                        valid_group["seed_similarity_tanimoto"].median()
                    ),
                    "q25_seed_similarity": float(
                        valid_group["seed_similarity_tanimoto"].quantile(0.25)
                    ),
                    "q75_seed_similarity": float(
                        valid_group["seed_similarity_tanimoto"].quantile(0.75)
                    ),
                    "scaffold_evaluable_count": int(len(scaffold)),
                    "scaffold_retention_fraction": (
                        float(scaffold.mean())
                        if len(scaffold)
                        else float("nan")
                    ),
                }
            )
        for seed_id, seed_group in group.groupby("seed_id", sort=True):
            valid = seed_group["is_rdkit_valid"].astype(bool)
            valid_group = seed_group[valid]
            scaffold = valid_group.loc[
                valid_group["scaffold_evaluable"].astype(bool),
                "scaffold_retained",
            ].astype(bool)
            rho = rau.safe_spearman(
                seed_group["alpha"], seed_group[property_name]
            )
            slope = rau.safe_linear_slope(
                seed_group["alpha"], seed_group[property_name]
            )
            per_seed_rows.append(
                {
                    "method": method,
                    "property": property_name,
                    "seed_id": int(seed_id),
                    "seed_row_index": int(seed_group["seed_row_index"].iloc[0]),
                    "spearman_alpha_property": rho,
                    "linear_slope_property_per_alpha": slope,
                    "seed_moves_expected_direction": bool(slope > 0)
                    if np.isfinite(slope)
                    else pd.NA,
                    "expected_sign_fraction_valid_candidates": (
                        float(
                            (
                                valid_group["expected_signed_improvement"] > 0
                            ).mean()
                        )
                        if len(valid_group)
                        else float("nan")
                    ),
                    "valid_fraction": float(valid.mean()),
                    "changed_from_seed_fraction_all": float(
                        seed_group["changed_from_seed"].astype(bool).mean()
                    ),
                    "changed_from_seed_fraction_valid": (
                        float(
                            valid_group["changed_from_seed"].astype(bool).mean()
                        )
                        if len(valid_group)
                        else float("nan")
                    ),
                    "median_seed_similarity": float(
                        valid_group["seed_similarity_tanimoto"].median()
                    ),
                    "scaffold_retention_fraction": (
                        float(scaffold.mean())
                        if len(scaffold)
                        else float("nan")
                    ),
                    "best_increase": float(valid_group["property_delta"].max()),
                    "best_decrease": float(-valid_group["property_delta"].min()),
                }
            )
    per_alpha = pd.DataFrame(per_alpha_rows)
    per_seed = pd.DataFrame(per_seed_rows)
    summary_rows = []
    for (method, property_name), group in per_seed.groupby(
        ["method", "property"], sort=False
    ):
        seed_direction = group["seed_moves_expected_direction"].dropna().astype(bool)
        source = candidates[
            candidates["method"].eq(method)
            & candidates["property"].eq(property_name)
        ]
        valid = source["is_rdkit_valid"].astype(bool)
        valid_source = source[valid]
        scaffold = valid_source.loc[
            valid_source["scaffold_evaluable"].astype(bool),
            "scaffold_retained",
        ].astype(bool)
        summary_rows.append(
            {
                "method": method,
                "property": property_name,
                "requested_candidates": int(len(source)),
                "valid_count": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "changed_from_seed_fraction_all": float(
                    source["changed_from_seed"].astype(bool).mean()
                ),
                "changed_from_seed_fraction_valid": float(
                    valid_source["changed_from_seed"].astype(bool).mean()
                ),
                "median_seed_similarity": float(
                    valid_source["seed_similarity_tanimoto"].median()
                ),
                "q25_seed_similarity": float(
                    valid_source["seed_similarity_tanimoto"].quantile(0.25)
                ),
                "q75_seed_similarity": float(
                    valid_source["seed_similarity_tanimoto"].quantile(0.75)
                ),
                "scaffold_retention_fraction": float(scaffold.mean())
                if len(scaffold)
                else float("nan"),
                "median_per_seed_spearman": float(
                    group["spearman_alpha_property"].median()
                ),
                "median_per_seed_slope": float(
                    group["linear_slope_property_per_alpha"].median()
                ),
                "fraction_seeds_moving_expected_direction": float(
                    seed_direction.mean()
                )
                if len(seed_direction)
                else float("nan"),
                "median_per_seed_expected_sign_fraction": float(
                    group["expected_sign_fraction_valid_candidates"].median()
                ),
                "median_best_increase": float(group["best_increase"].median()),
                "median_best_decrease": float(group["best_decrease"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    _atomic_csv(per_alpha, output_dir / "traversal_per_signed_alpha.csv")
    _atomic_csv(per_seed, output_dir / "traversal_per_seed.csv")
    _atomic_csv(summary, output_dir / "traversal_compact_summary.csv")
    return per_alpha, per_seed, summary


def matched_latent_displacement_tables(
    candidates: pd.DataFrame,
    *,
    output_dir: Path = OUTPUTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "property",
        "seed_id",
        "seed_row_index",
        "alpha_index",
        "alpha",
        "latent_displacement",
        "is_rdkit_valid",
        "property_delta",
        "expected_signed_improvement",
        "seed_similarity_tanimoto",
        "scaffold_retained",
    ]
    raw = candidates[candidates["method"].eq(METHOD_OURS)][columns]
    chemspace = candidates[candidates["method"].eq(METHOD_CHEMSPACE)][columns]
    pairs = raw.merge(
        chemspace,
        on=[
            "property",
            "seed_id",
            "seed_row_index",
            "alpha_index",
        ],
        suffixes=("_raw_ols", "_chemspace"),
        validate="one_to_one",
    )
    if len(pairs) != 30_000:
        raise AssertionError("Latent-displacement pairing is incomplete")
    if not np.allclose(
        pairs["alpha_raw_ols"], pairs["alpha_chemspace"], rtol=0.0, atol=0.0
    ) or not np.allclose(
        pairs["latent_displacement_raw_ols"],
        pairs["latent_displacement_chemspace"],
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("Canonical alpha/displacement values disagree")
    pairs["alpha"] = pairs.pop("alpha_raw_ols")
    pairs.drop(columns=["alpha_chemspace"], inplace=True)
    pairs["latent_displacement"] = pairs.pop(
        "latent_displacement_raw_ols"
    )
    pairs.drop(columns=["latent_displacement_chemspace"], inplace=True)
    pairs["expected_improvement_difference_chemspace_minus_raw"] = (
        pairs["expected_signed_improvement_chemspace"]
        - pairs["expected_signed_improvement_raw_ols"]
    )
    pairs["seed_similarity_difference_chemspace_minus_raw"] = (
        pairs["seed_similarity_tanimoto_chemspace"]
        - pairs["seed_similarity_tanimoto_raw_ols"]
    )
    summary = (
        pairs.groupby(["property", "alpha", "latent_displacement"], as_index=False)
        .agg(
            paired_seeds=("seed_id", "size"),
            raw_valid_fraction=("is_rdkit_valid_raw_ols", "mean"),
            chemspace_valid_fraction=("is_rdkit_valid_chemspace", "mean"),
            raw_median_expected_improvement=(
                "expected_signed_improvement_raw_ols",
                "median",
            ),
            chemspace_median_expected_improvement=(
                "expected_signed_improvement_chemspace",
                "median",
            ),
            median_improvement_difference_chemspace_minus_raw=(
                "expected_improvement_difference_chemspace_minus_raw",
                "median",
            ),
            raw_median_seed_similarity=(
                "seed_similarity_tanimoto_raw_ols",
                "median",
            ),
            chemspace_median_seed_similarity=(
                "seed_similarity_tanimoto_chemspace",
                "median",
            ),
        )
    )
    _atomic_csv(pairs, output_dir / "matched_latent_displacement_pairs.csv")
    _atomic_csv(
        summary, output_dir / "matched_latent_displacement_summary.csv"
    )
    return pairs, summary


def matched_seed_similarity_tables(
    candidates: pd.DataFrame,
    *,
    output_dir: Path = OUTPUTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-to-one rank matching within property, seed, and alpha sign."""

    rows = []
    for property_name in PROPERTIES:
        for seed_id in range(50):
            for sign_name, sign in (("negative", -1), ("positive", 1)):
                groups = {}
                for method in (METHOD_OURS, METHOD_CHEMSPACE):
                    group = candidates[
                        candidates["property"].eq(property_name)
                        & candidates["seed_id"].eq(seed_id)
                        & candidates["method"].eq(method)
                        & (np.sign(candidates["alpha"]) == sign)
                        & candidates["is_rdkit_valid"].astype(bool)
                        & candidates["seed_similarity_tanimoto"].notna()
                    ].copy()
                    group.sort_values(
                        ["seed_similarity_tanimoto", "latent_displacement", "alpha"],
                        ascending=[False, True, True],
                        inplace=True,
                    )
                    groups[method] = group.reset_index(drop=True)
                pair_count = min(len(groups[METHOD_OURS]), len(groups[METHOD_CHEMSPACE]))
                for rank in range(pair_count):
                    raw = groups[METHOD_OURS].iloc[rank]
                    chem = groups[METHOD_CHEMSPACE].iloc[rank]
                    rows.append(
                        {
                            "property": property_name,
                            "seed_id": seed_id,
                            "alpha_sign": sign_name,
                            "similarity_rank": rank,
                            "raw_alpha": raw["alpha"],
                            "chemspace_alpha": chem["alpha"],
                            "raw_latent_displacement": raw["latent_displacement"],
                            "chemspace_latent_displacement": chem[
                                "latent_displacement"
                            ],
                            "raw_seed_similarity": raw[
                                "seed_similarity_tanimoto"
                            ],
                            "chemspace_seed_similarity": chem[
                                "seed_similarity_tanimoto"
                            ],
                            "absolute_similarity_mismatch": abs(
                                raw["seed_similarity_tanimoto"]
                                - chem["seed_similarity_tanimoto"]
                            ),
                            "raw_expected_signed_improvement": raw[
                                "expected_signed_improvement"
                            ],
                            "chemspace_expected_signed_improvement": chem[
                                "expected_signed_improvement"
                            ],
                            "improvement_difference_chemspace_minus_raw": (
                                chem["expected_signed_improvement"]
                                - raw["expected_signed_improvement"]
                            ),
                        }
                    )
    pairs = pd.DataFrame(rows)
    summary = (
        pairs.groupby(["property", "alpha_sign"], as_index=False)
        .agg(
            matched_pairs=("similarity_rank", "size"),
            median_absolute_similarity_mismatch=(
                "absolute_similarity_mismatch",
                "median",
            ),
            p95_absolute_similarity_mismatch=(
                "absolute_similarity_mismatch",
                lambda values: values.quantile(0.95),
            ),
            raw_median_expected_improvement=(
                "raw_expected_signed_improvement",
                "median",
            ),
            chemspace_median_expected_improvement=(
                "chemspace_expected_signed_improvement",
                "median",
            ),
            median_improvement_difference_chemspace_minus_raw=(
                "improvement_difference_chemspace_minus_raw",
                "median",
            ),
            raw_median_latent_displacement=(
                "raw_latent_displacement",
                "median",
            ),
            chemspace_median_latent_displacement=(
                "chemspace_latent_displacement",
                "median",
            ),
        )
    )
    _atomic_csv(pairs, output_dir / "matched_seed_similarity_pairs.csv")
    _atomic_csv(summary, output_dir / "matched_seed_similarity_summary.csv")
    return pairs, summary


def constrained_design_tables(
    candidates: pd.DataFrame,
    *,
    output_dir: Path = OUTPUTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (method, property_name, seed_id), group in candidates.groupby(
        ["method", "property", "seed_id"], sort=False
    ):
        for objective, multiplier in (("maximize", 1.0), ("minimize", -1.0)):
            for cutoff in SIMILARITY_CUTOFFS:
                eligible = group[
                    group["is_rdkit_valid"].astype(bool)
                    & group["changed_from_seed"].astype(bool)
                    & group["seed_similarity_tanimoto"].ge(cutoff)
                ].copy()
                eligible["objective_improvement"] = (
                    multiplier * eligible["property_delta"]
                )
                if len(eligible):
                    selected = eligible.loc[eligible["objective_improvement"].idxmax()]
                    best = float(selected["objective_improvement"])
                    selected_alpha = float(selected["alpha"])
                    selected_similarity = float(selected["seed_similarity_tanimoto"])
                    selected_scaffold = selected["scaffold_retained"]
                    selected_smiles = selected["canonical_smiles"]
                else:
                    best = float("nan")
                    selected_alpha = float("nan")
                    selected_similarity = float("nan")
                    selected_scaffold = pd.NA
                    selected_smiles = pd.NA
                rows.append(
                    {
                        "method": method,
                        "property": property_name,
                        "seed_id": int(seed_id),
                        "objective": objective,
                        "seed_similarity_cutoff": cutoff,
                        "candidate_budget": int(len(group)),
                        "eligible_valid_changed_candidates": int(len(eligible)),
                        "success": bool(best > 0) if np.isfinite(best) else False,
                        "best_objective_improvement": best,
                        "selected_alpha": selected_alpha,
                        "selected_seed_similarity": selected_similarity,
                        "selected_scaffold_retained": selected_scaffold,
                        "selected_canonical_smiles": selected_smiles,
                    }
                )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(
            ["method", "property", "objective", "seed_similarity_cutoff"],
            as_index=False,
        )
        .agg(
            seeds=("seed_id", "size"),
            successful_seeds=("success", "sum"),
            success_fraction=("success", "mean"),
            median_best_improvement=("best_objective_improvement", "median"),
            median_selected_similarity=("selected_seed_similarity", "median"),
            median_selected_alpha=("selected_alpha", "median"),
            median_eligible_candidates=(
                "eligible_valid_changed_candidates",
                "median",
            ),
        )
    )
    _atomic_csv(per_seed, output_dir / "constrained_design_per_seed.csv")
    _atomic_csv(summary, output_dir / "constrained_design_compact_summary.csv")
    return per_seed, summary


def selfies_confound_tables(
    candidates: pd.DataFrame,
    cosines: pd.DataFrame,
    *,
    output_dir: Path = OUTPUTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rau = _rebuttal_utils()
    rows = []
    for (method, property_name, seed_id), group in candidates.groupby(
        ["method", "property", "seed_id"], sort=False
    ):
        for confound in CONFOUNDS:
            rows.append(
                {
                    "method": method,
                    "property": property_name,
                    "seed_id": int(seed_id),
                    "confound": confound,
                    "spearman_alpha_confound": rau.safe_spearman(
                        group["alpha"], group[confound]
                    ),
                    "linear_slope_confound_per_alpha": rau.safe_linear_slope(
                        group["alpha"], group[confound]
                    ),
                }
            )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(["method", "property", "confound"], as_index=False)
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
            fraction_positive_spearman=(
                "spearman_alpha_confound",
                lambda values: (values.dropna() > 0).mean(),
            ),
            median_linear_slope_confound=(
                "linear_slope_confound_per_alpha",
                "median",
            ),
        )
    )
    _atomic_csv(cosines, output_dir / "direction_cosines.csv")
    _atomic_csv(per_seed, output_dir / "selfies_confound_per_seed.csv")
    _atomic_csv(summary, output_dir / "selfies_confound_compact_summary.csv")
    return per_seed, summary


def accounting_tables(
    training: pd.DataFrame,
    timing: pd.DataFrame,
    chemspace_candidates: pd.DataFrame,
    *,
    raw_training: pd.DataFrame | None = None,
    raw_timing: pd.DataFrame | None = None,
    raw_candidates: pd.DataFrame | None = None,
    output_dir: Path = OUTPUTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oracle_rows = []
    runtime_rows = []
    for property_name in PROPERTIES:
        train_row = training[training["property"].eq(property_name)].iloc[0]
        time_row = timing[timing["property"].eq(property_name)].iloc[0]
        candidates = chemspace_candidates[
            chemspace_candidates["property"].eq(property_name)
        ]
        valid_count = int(candidates["is_rdkit_valid"].astype(bool).sum())
        oracle_rows.append(
            {
                "method": METHOD_CHEMSPACE,
                "property": property_name,
                "cached_training_labels_consumed": int(
                    train_row["cached_labels_consumed"]
                ),
                "incremental_training_property_oracle_calls": 0,
                "decoded_candidates": int(len(candidates)),
                "descriptor_bundle_attempts": int(len(candidates)),
                "valid_decoded_property_oracle_calls": valid_count,
                "additional_constrained_design_oracle_calls": 0,
            }
        )
        runtime_rows.append(
            {
                "method": METHOD_CHEMSPACE,
                "property": property_name,
                "direction_svm_fit_seconds": train_row["svm_fit_seconds"],
                "direction_learning_total_seconds": train_row[
                    "direction_learning_total_seconds"
                ],
                "frozen_decoder_seconds": time_row["decoder_seconds"],
                "descriptor_and_structure_seconds": time_row[
                    "descriptor_and_structure_seconds"
                ],
                "decoder_loaded_from_cache": time_row["loaded_from_cache"],
            }
        )
        if (
            raw_training is not None
            and raw_timing is not None
            and raw_candidates is not None
        ):
            raw_train_row = raw_training[
                raw_training["property"].eq(property_name)
            ].iloc[0]
            raw_time_row = raw_timing[
                raw_timing["property"].eq(property_name)
            ].iloc[0]
            raw_property_candidates = raw_candidates[
                raw_candidates["property"].eq(property_name)
            ]
            oracle_rows.append(
                {
                    "method": METHOD_OURS,
                    "property": property_name,
                    "cached_training_labels_consumed": int(
                        raw_train_row["eligible_training_rows"]
                    ),
                    "incremental_training_property_oracle_calls": 0,
                    "decoded_candidates": int(len(raw_property_candidates)),
                    "descriptor_bundle_attempts": int(len(raw_property_candidates)),
                    "valid_decoded_property_oracle_calls": int(
                        raw_property_candidates["is_rdkit_valid"].astype(bool).sum()
                    ),
                    "additional_constrained_design_oracle_calls": 0,
                }
            )
            runtime_rows.append(
                {
                    "method": METHOD_OURS,
                    "property": property_name,
                    "direction_svm_fit_seconds": float("nan"),
                    "direction_learning_total_seconds": raw_train_row[
                        "joint_fit_seconds_all_six_targets"
                    ],
                    "frozen_decoder_seconds": raw_time_row["decoder_seconds"],
                    "descriptor_and_structure_seconds": raw_time_row[
                        "descriptor_and_structure_seconds"
                    ],
                    "decoder_loaded_from_cache": raw_time_row[
                        "loaded_from_cache"
                    ],
                }
            )
    oracle = pd.DataFrame(oracle_rows)
    runtime = pd.DataFrame(runtime_rows)
    _atomic_csv(oracle, output_dir / "property_oracle_accounting.csv")
    _atomic_csv(runtime, output_dir / "runtime_accounting.csv")
    return oracle, runtime


def create_figures(
    per_alpha: pd.DataFrame,
    constrained_summary: pd.DataFrame,
    cosines: pd.DataFrame,
    confound_summary: pd.DataFrame,
    *,
    output_dir: Path = OUTPUTS,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    colors = {METHOD_OURS: "#2457A7", METHOD_CHEMSPACE: "#D95F02"}

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for axis, property_name in zip(axes.flat, PROPERTIES):
        for method in (METHOD_OURS, METHOD_CHEMSPACE):
            group = per_alpha[
                per_alpha["property"].eq(property_name)
                & per_alpha["method"].eq(method)
            ]
            axis.plot(
                group["alpha"],
                group["median_property_delta"],
                color=colors[method],
                label=method,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(property_name)
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel("median decoded change")
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        output_dir / "traversal_property_curves.png", dpi=250, bbox_inches="tight"
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for axis, property_name in zip(axes.flat, PROPERTIES):
        for method in (METHOD_OURS, METHOD_CHEMSPACE):
            group = per_alpha[
                per_alpha["property"].eq(property_name)
                & per_alpha["method"].eq(method)
            ]
            axis.plot(
                group["alpha"],
                group["median_seed_similarity"],
                color=colors[method],
                label=method,
            )
        axis.set_title(property_name)
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel("median seed Tanimoto")
        axis.set_ylim(0, 1.02)
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        output_dir / "seed_similarity_curves.png", dpi=250, bbox_inches="tight"
    )
    plt.close(figure)

    plot_data = constrained_summary[
        constrained_summary["objective"].eq("maximize")
    ]
    grid = sns.catplot(
        data=plot_data,
        x="seed_similarity_cutoff",
        y="success_fraction",
        hue="method",
        col="property",
        col_wrap=3,
        kind="point",
        palette=colors,
        height=3.0,
        aspect=1.15,
    )
    grid.set(ylim=(0, 1.02))
    grid.set_axis_labels("seed-similarity cutoff", "success fraction")
    grid.figure.savefig(
        output_dir / "constrained_design_success.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(grid.figure)

    cosine_columns = [
        "cosine_chemspace_vs_raw_ols",
        "cosine_chemspace_vs_residual_ols",
        *[f"cosine_chemspace_vs_{name}" for name in CONFOUNDS],
    ]
    matrix = cosines.set_index("property")[cosine_columns]
    matrix.columns = [
        "raw OLS",
        "residual OLS",
        "SELFIES length",
        "branch tokens",
        "ring tokens",
        "token entropy",
    ]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        ax=axis,
    )
    axis.set_title("ChemSpacE direction cosines")
    figure.tight_layout()
    figure.savefig(
        output_dir / "direction_cosines.png", dpi=250, bbox_inches="tight"
    )
    plt.close(figure)

    chem_confound = confound_summary[
        confound_summary["method"].eq(METHOD_CHEMSPACE)
    ].pivot(
        index="property",
        columns="confound",
        values="median_spearman_alpha_confound",
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    sns.heatmap(
        chem_confound,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        ax=axis,
    )
    axis.set_title("Decoded SELFIES-confound trajectories: median seed rho")
    figure.tight_layout()
    figure.savefig(
        output_dir / "selfies_confound_audit.png", dpi=250, bbox_inches="tight"
    )
    plt.close(figure)


def build_all_reports(
    inputs: CachedInputs,
    directions: Mapping[str, np.ndarray],
    training: pd.DataFrame,
    chemspace_candidates: pd.DataFrame,
    timing: pd.DataFrame,
    *,
    raw_directions: Mapping[str, np.ndarray],
    raw_training: pd.DataFrame,
    raw_candidates: pd.DataFrame,
    raw_timing: pd.DataFrame,
    output_dir: Path = OUTPUTS,
) -> dict[str, pd.DataFrame]:
    reference = load_reference_directions()
    historical_raw = {
        property_name: reference[f"raw::{property_name}"]
        for property_name in PROPERTIES
    }
    for property_name in PROPERTIES:
        reference[f"raw::{property_name}"] = raw_directions[property_name]
    cosines = direction_cosine_table(directions, reference)
    raw_stability = pd.DataFrame(
        [
            {
                "property": property_name,
                "cosine_seed_excluded_vs_historical_raw_ols": float(
                    raw_directions[property_name] @ historical_raw[property_name]
                ),
            }
            for property_name in PROPERTIES
        ]
    )
    _atomic_csv(
        raw_stability, output_dir / "raw_ols_direction_stability.csv"
    )
    combined = combine_candidates(
        raw_candidates, chemspace_candidates, output_dir=output_dir
    )
    per_alpha, per_seed, traversal_summary = traversal_tables(
        combined, output_dir=output_dir
    )
    latent_pairs, latent_summary = matched_latent_displacement_tables(
        combined, output_dir=output_dir
    )
    similarity_pairs, similarity_summary = matched_seed_similarity_tables(
        combined, output_dir=output_dir
    )
    constrained_per_seed, constrained_summary = constrained_design_tables(
        combined, output_dir=output_dir
    )
    confound_per_seed, confound_summary = selfies_confound_tables(
        combined, cosines, output_dir=output_dir
    )
    oracle, runtime = accounting_tables(
        training,
        timing,
        chemspace_candidates,
        raw_training=raw_training,
        raw_timing=raw_timing,
        raw_candidates=raw_candidates,
        output_dir=output_dir,
    )
    create_figures(
        per_alpha,
        constrained_summary,
        cosines,
        confound_summary,
        output_dir=output_dir,
    )
    return {
        "candidates": combined,
        "per_alpha": per_alpha,
        "per_seed": per_seed,
        "traversal_summary": traversal_summary,
        "latent_pairs": latent_pairs,
        "latent_summary": latent_summary,
        "similarity_pairs": similarity_pairs,
        "similarity_summary": similarity_summary,
        "constrained_per_seed": constrained_per_seed,
        "constrained_summary": constrained_summary,
        "confound_per_seed": confound_per_seed,
        "confound_summary": confound_summary,
        "cosines": cosines,
        "oracle": oracle,
        "runtime": runtime,
        "raw_stability": raw_stability,
    }


def verify_outputs(*, output_dir: Path = OUTPUTS) -> pd.DataFrame:
    """Lightweight deterministic acceptance checks used by the test notebook."""

    directions = np.load(output_dir / "chemspace_directions.npz")
    checks = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    record(
        "properties",
        tuple(directions.files) == PROPERTIES,
        f"{directions.files}",
    )
    norms = np.array([np.linalg.norm(directions[name]) for name in PROPERTIES])
    record("unit directions", np.allclose(norms, 1.0), str(norms))
    training = pd.read_csv(output_dir / "direction_training_summary.csv")
    record(
        "400 labels per property",
        training["cached_labels_consumed"].eq(400).all(),
        training["cached_labels_consumed"].tolist().__str__(),
    )
    record(
        "280 SVM fit rows",
        training["svm_fit_rows"].eq(280).all(),
        training["svm_fit_rows"].tolist().__str__(),
    )
    candidates = pd.read_csv(output_dir / "full_traversal_candidates.csv")
    record("60,000 candidates", len(candidates) == 60_000, str(len(candidates)))
    counts = candidates.groupby(["method", "property"]).size()
    expected_method_property_pairs = {
        (method, property_name)
        for method in (METHOD_OURS, METHOD_CHEMSPACE)
        for property_name in PROPERTIES
    }
    record(
        "strict method labels",
        set(counts.index.tolist()) == expected_method_property_pairs,
        str(sorted(counts.index.tolist())),
    )
    record(
        "5,000 per method/property",
        counts.eq(5_000).all(),
        counts.to_dict().__str__(),
    )
    record(
        "signed alpha grid preserved",
        all(
            np.array_equal(
                group["alpha_index"].drop_duplicates().to_numpy(),
                np.arange(len(ALPHAS)),
            )
            and np.allclose(
                group.drop_duplicates("alpha_index")["alpha"].to_numpy(),
                ALPHAS,
                rtol=0.0,
                atol=1e-12,
            )
            for _, group in candidates.groupby(["method", "property"], sort=False)
        ),
        "100 exact signed values",
    )
    seed_audit = pd.read_csv(output_dir / "seed_split_audit.csv")
    split_counts = seed_audit["split"].value_counts().to_dict()
    record(
        "historical seed split disclosed",
        split_counts == {"train": 39, "test": 8, "validation": 3},
        str(split_counts),
    )
    seed_rows = set(seed_audit["seed_row_index"].astype(int))
    selected_rows = pd.read_csv(output_dir / "direction_training_rows.csv")
    selected_seed_overlap = seed_rows.intersection(
        selected_rows["global_row_index"].astype(int)
    )
    record(
        "ChemSpacE learning rows exclude all seeds",
        not selected_seed_overlap,
        str(sorted(selected_seed_overlap)),
    )
    raw_training = pd.read_csv(output_dir / "raw_ols_seed_excluded_training.csv")
    record(
        "raw OLS excludes registry",
        raw_training["excluded_seed_rows_total"].eq(50).all()
        and raw_training["excluded_seed_rows_from_training_split"].eq(39).all(),
        raw_training[
            [
                "excluded_seed_rows_total",
                "excluded_seed_rows_from_training_split",
            ]
        ]
        .drop_duplicates()
        .to_dict("records")
        .__str__(),
    )
    raw_stability = pd.read_csv(output_dir / "raw_ols_direction_stability.csv")
    record(
        "seed-excluded OLS direction stability",
        raw_stability[
            "cosine_seed_excluded_vs_historical_raw_ols"
        ].gt(0.9999999).all(),
        str(
            raw_stability[
                "cosine_seed_excluded_vs_historical_raw_ols"
            ].tolist()
        ),
    )
    constrained = pd.read_csv(
        output_dir / "constrained_design_compact_summary.csv"
    )
    record(
        "published similarity cutoffs",
        set(constrained["seed_similarity_cutoff"].unique())
        == set(SIMILARITY_CUTOFFS),
        str(sorted(constrained["seed_similarity_cutoff"].unique())),
    )
    return pd.DataFrame(checks)


def synthetic_boundary_test() -> pd.DataFrame:
    """Small method test: extreme selection, orientation, and unit norm."""

    rng = np.random.default_rng(7)
    latents = rng.normal(size=(1_000, 8)).astype(np.float32)
    scores = latents[:, 0] + 0.05 * rng.normal(size=1_000)
    direction, details, selected = fit_chemspace_boundary(
        latents,
        scores,
        np.arange(1_000),
        property_name="synthetic",
        n_per_class=40,
        fit_ratio=0.70,
        seed=42,
    )
    assertions = {
        "unit_norm": np.isclose(np.linalg.norm(direction), 1.0),
        "oriented_to_high_score": float(direction[0]) > 0,
        "balanced_extremes": selected["extreme_class"].value_counts().to_dict()
        == {"high": 40, "low": 40},
        "fit_count": details["svm_fit_rows"] == 56,
        "validation_count": details["svm_validation_rows"] == 24,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    return pd.DataFrame(
        [{"check": name, "passed": value} for name, value in assertions.items()]
    )
