"""Cache-only SA and NP-likeness analysis for the ChemSpacE comparison.

The scorer implementations and models are imported from the installed RDKit
Contrib package. No VAE encoding, training, or decoding is performed here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDConfig, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.NP_Score import npscorer
from rdkit.Contrib.SA_Score import sascorer


HERE = Path(__file__).resolve().parent
BASE_OUTPUTS = HERE / "outputs"
OUTPUTS = BASE_OUTPUTS / "chemical_quality"
SOURCE_CANDIDATES = BASE_OUTPUTS / "full_traversal_candidates.csv"

METHOD_CHEMSPACE = "ChemSpacE extreme-SVM"
METHOD_OLS = "MolecuLA raw OLS (seed-excluded)"
METHODS = (METHOD_OLS, METHOD_CHEMSPACE)
PROPERTIES = (
    "FractionCSP3",
    "HBA",
    "TPSA",
    "cLogP",
    "BertzCT",
    "HeavyAtomCount",
)
SIMILARITY_CUTOFFS = (0.0, 0.2, 0.4, 0.6)
QUALITY_RULES = (
    "none",
    "sa_nonworsening",
    "np_nondecreasing",
    "sa_and_np",
)
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    countSimulation=False,
    includeChirality=False,
    useBondTypes=True,
    onlyNonzeroInvariants=False,
    includeRingMembership=True,
    includeRedundantEnvironments=False,
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scorer_provenance() -> pd.DataFrame:
    """Record scorer/model identity without embedding local filesystem paths."""

    contrib = Path(RDConfig.RDContribDir)
    records = (
        (
            "sa_implementation",
            contrib / "SA_Score" / "sascorer.py",
            "https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py",
            "1=easier, 10=harder; RDKit Contrib implementation",
        ),
        (
            "sa_fragment_model",
            contrib / "SA_Score" / "fpscores.pkl.gz",
            "https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/fpscores.pkl.gz",
            "Radius-2 Morgan fragment contribution table",
        ),
        (
            "np_implementation",
            contrib / "NP_Score" / "npscorer.py",
            "https://github.com/rdkit/rdkit/blob/master/Contrib/NP_Score/npscorer.py",
            "Higher=more natural-product-like; confidence is fragment coverage",
        ),
        (
            "np_fragment_model",
            contrib / "NP_Score" / "publicnp.model.gz",
            "https://github.com/rdkit/rdkit/blob/master/Contrib/NP_Score/publicnp.model.gz",
            "Public natural-product versus ZINC fragment model",
        ),
    )
    rows = []
    for artifact, path, url, interpretation in records:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "artifact": artifact,
                "logical_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "rdkit_version": rdBase.rdkitVersion,
                "official_source": url,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def load_candidates(path: Path = SOURCE_CANDIDATES) -> pd.DataFrame:
    columns = [
        "method",
        "property",
        "seed_id",
        "seed_row_index",
        "alpha",
        "latent_displacement",
        "canonical_smiles",
        "is_rdkit_valid",
        "seed_canonical_smiles",
        "changed_from_seed",
        "seed_similarity_tanimoto",
        "scaffold_retained",
        "seed_property_value",
        "property_delta",
        "expected_signed_improvement",
    ]
    frame = pd.read_csv(path, usecols=columns)
    if len(frame) != 60_000:
        raise AssertionError(f"Expected 60,000 cached candidates, found {len(frame)}")
    expected_pairs = {
        (method, property_name)
        for method in METHODS
        for property_name in PROPERTIES
    }
    counts = frame.groupby(["method", "property"]).size()
    if set(counts.index) != expected_pairs or not counts.eq(5_000).all():
        raise AssertionError(f"Unexpected method/property counts: {counts.to_dict()}")
    frame["is_rdkit_valid"] = frame["is_rdkit_valid"].astype(bool)
    frame["changed_from_seed"] = frame["changed_from_seed"].astype(bool)
    frame["alpha_sign"] = np.where(frame["alpha"] < 0, "negative", "positive")
    return frame


def score_unique_structures(candidates: pd.DataFrame) -> pd.DataFrame:
    """Score each scorer-version canonical structure once, including every seed.

    The cached canonical string is retained as a source identifier.  A second
    canonical string is generated with the scoring RDKit version so equivalent
    source spellings cannot be counted as separate structures.
    """

    valid_smiles = candidates.loc[
        candidates["is_rdkit_valid"], "canonical_smiles"
    ].dropna()
    seed_smiles = candidates["seed_canonical_smiles"].dropna()
    seeds = set(seed_smiles.astype(str))
    source_smiles_values = sorted(set(valid_smiles.astype(str)).union(seeds))
    np_model = npscorer.readNPModel()

    source_to_scoring = {}
    scoring_molecules = {}
    for source_smiles in source_smiles_values:
        molecule = Chem.MolFromSmiles(source_smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise AssertionError(
                f"Canonical cache failed RDKit parsing: {source_smiles}"
            )
        scoring_canonical = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        source_to_scoring[source_smiles] = scoring_canonical
        scoring_molecules.setdefault(scoring_canonical, molecule)

    scores = {}
    for scoring_canonical, molecule in scoring_molecules.items():
        np_result = npscorer.scoreMolWConfidence(molecule, np_model)
        scores[scoring_canonical] = {
            "sa_score": float(sascorer.calculateScore(molecule)),
            "np_likeness": float(np_result.nplikeness),
            "np_fragment_confidence": float(np_result.confidence),
            "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        }

    rows = []
    for source_smiles, scoring_canonical in source_to_scoring.items():
        rows.append(
            {
                "source_canonical_smiles": source_smiles,
                "scoring_canonical_smiles": scoring_canonical,
                **scores[scoring_canonical],
                "source_matches_scoring_canonical": (
                    source_smiles == scoring_canonical
                ),
                "is_seed_structure": source_smiles in seeds,
            }
        )
    return pd.DataFrame(rows)


def attach_scores(
    candidates: pd.DataFrame, structure_scores: pd.DataFrame
) -> pd.DataFrame:
    candidate_scores = structure_scores.rename(
        columns={
            "scoring_canonical_smiles": "candidate_scoring_canonical_smiles",
            "sa_score": "candidate_sa_score",
            "np_likeness": "candidate_np_likeness",
            "np_fragment_confidence": "candidate_np_fragment_confidence",
            "heavy_atom_count": "candidate_heavy_atom_count",
            "source_matches_scoring_canonical": (
                "candidate_source_matches_scoring_canonical"
            ),
        }
    ).drop(columns=["is_seed_structure"])
    seed_scores = structure_scores.rename(
        columns={
            "source_canonical_smiles": "seed_canonical_smiles",
            "scoring_canonical_smiles": "seed_scoring_canonical_smiles",
            "sa_score": "seed_sa_score",
            "np_likeness": "seed_np_likeness",
            "np_fragment_confidence": "seed_np_fragment_confidence",
            "heavy_atom_count": "seed_heavy_atom_count",
            "source_matches_scoring_canonical": (
                "seed_source_matches_scoring_canonical"
            ),
        }
    ).drop(columns=["is_seed_structure"])

    frame = candidates.merge(
        candidate_scores,
        left_on="canonical_smiles",
        right_on="source_canonical_smiles",
        how="left",
        validate="many_to_one",
    )
    frame = frame.drop(columns=["source_canonical_smiles"])
    frame = frame.merge(
        seed_scores,
        on="seed_canonical_smiles",
        how="left",
        validate="many_to_one",
    )
    valid = frame["is_rdkit_valid"]
    candidate_columns = [
        "candidate_sa_score",
        "candidate_np_likeness",
        "candidate_np_fragment_confidence",
        "candidate_heavy_atom_count",
    ]
    if frame.loc[valid, candidate_columns].isna().any().any():
        raise AssertionError("A valid candidate is missing SA/NP scores")
    if frame.loc[~valid, candidate_columns].notna().any().any():
        raise AssertionError("An invalid candidate unexpectedly received SA/NP scores")
    if frame[
        [
            "seed_sa_score",
            "seed_np_likeness",
            "seed_np_fragment_confidence",
            "seed_heavy_atom_count",
        ]
    ].isna().any().any():
        raise AssertionError("A seed is missing SA/NP scores")

    frame["delta_sa_vs_seed"] = (
        frame["candidate_sa_score"] - frame["seed_sa_score"]
    )
    frame["delta_np_vs_seed"] = (
        frame["candidate_np_likeness"] - frame["seed_np_likeness"]
    )
    frame["delta_np_confidence_vs_seed"] = (
        frame["candidate_np_fragment_confidence"]
        - frame["seed_np_fragment_confidence"]
    )
    frame["sa_nonworsening"] = valid & frame["delta_sa_vs_seed"].le(0)
    frame["np_nondecreasing"] = valid & frame["delta_np_vs_seed"].ge(0)
    frame["sa_and_np"] = frame["sa_nonworsening"] & frame["np_nondecreasing"]
    return frame


def _quantile(series: pd.Series, probability: float) -> float:
    values = series.dropna()
    return float(values.quantile(probability)) if len(values) else float("nan")


def _absolute_score_record(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "denominator": int(len(frame)),
        "median_sa_score": _quantile(frame["candidate_sa_score"], 0.50),
        "q25_sa_score": _quantile(frame["candidate_sa_score"], 0.25),
        "q75_sa_score": _quantile(frame["candidate_sa_score"], 0.75),
        "median_np_likeness": _quantile(frame["candidate_np_likeness"], 0.50),
        "q25_np_likeness": _quantile(frame["candidate_np_likeness"], 0.25),
        "q75_np_likeness": _quantile(frame["candidate_np_likeness"], 0.75),
        "median_np_fragment_confidence": _quantile(
            frame["candidate_np_fragment_confidence"], 0.50
        ),
    }


def quality_summary_tables(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact denominator-aware summaries and per-seed summaries."""

    per_seed_rows = []
    compact_rows = []
    for (method, property_name), group in candidates.groupby(
        ["method", "property"], sort=False
    ):
        valid = group[group["is_rdkit_valid"]]
        changed = valid[valid["changed_from_seed"]]

        occurrence = {
            "method": method,
            "property": property_name,
            "weighting": "candidate_occurrence",
            **_absolute_score_record(valid),
            "requested_candidates": int(len(group)),
            "valid_candidates": int(len(valid)),
            "distinct_canonical_structures": int(
                valid["candidate_scoring_canonical_smiles"].nunique()
            ),
            "seed_count": int(group["seed_id"].nunique()),
            "delta_denominator_valid_changed": int(len(changed)),
            "median_delta_sa_vs_seed": _quantile(changed["delta_sa_vs_seed"], 0.50),
            "median_delta_np_vs_seed": _quantile(changed["delta_np_vs_seed"], 0.50),
            "median_delta_np_confidence_vs_seed": _quantile(
                changed["delta_np_confidence_vs_seed"], 0.50
            ),
            "sa_nonworsening_fraction_valid_changed": (
                float(changed["sa_nonworsening"].mean()) if len(changed) else np.nan
            ),
            "np_nondecreasing_fraction_valid_changed": (
                float(changed["np_nondecreasing"].mean()) if len(changed) else np.nan
            ),
            "sa_and_np_fraction_valid_changed": (
                float(changed["sa_and_np"].mean()) if len(changed) else np.nan
            ),
        }
        compact_rows.append(occurrence)

        unique = valid.drop_duplicates("candidate_scoring_canonical_smiles")
        compact_rows.append(
            {
                "method": method,
                "property": property_name,
                "weighting": "unique_structure",
                **_absolute_score_record(unique),
                "requested_candidates": int(len(group)),
                "valid_candidates": int(len(valid)),
                "distinct_canonical_structures": int(len(unique)),
                "seed_count": int(group["seed_id"].nunique()),
                "delta_denominator_valid_changed": np.nan,
                "median_delta_sa_vs_seed": np.nan,
                "median_delta_np_vs_seed": np.nan,
                "median_delta_np_confidence_vs_seed": np.nan,
                "sa_nonworsening_fraction_valid_changed": np.nan,
                "np_nondecreasing_fraction_valid_changed": np.nan,
                "sa_and_np_fraction_valid_changed": np.nan,
            }
        )

        seed_records = []
        for seed_id, seed_group in group.groupby("seed_id", sort=True):
            seed_valid = seed_group[seed_group["is_rdkit_valid"]]
            seed_changed = seed_valid[seed_valid["changed_from_seed"]]
            record = {
                "method": method,
                "property": property_name,
                "seed_id": int(seed_id),
                "requested_candidates": int(len(seed_group)),
                "valid_candidates": int(len(seed_valid)),
                "distinct_canonical_structures": int(
                    seed_valid["candidate_scoring_canonical_smiles"].nunique()
                ),
                "median_sa_score": _quantile(
                    seed_valid["candidate_sa_score"], 0.50
                ),
                "median_np_likeness": _quantile(
                    seed_valid["candidate_np_likeness"], 0.50
                ),
                "median_np_fragment_confidence": _quantile(
                    seed_valid["candidate_np_fragment_confidence"], 0.50
                ),
                "valid_changed_candidates": int(len(seed_changed)),
                "median_delta_sa_vs_seed": _quantile(
                    seed_changed["delta_sa_vs_seed"], 0.50
                ),
                "median_delta_np_vs_seed": _quantile(
                    seed_changed["delta_np_vs_seed"], 0.50
                ),
                "median_delta_np_confidence_vs_seed": _quantile(
                    seed_changed["delta_np_confidence_vs_seed"], 0.50
                ),
                "sa_nonworsening_fraction_valid_changed": (
                    float(seed_changed["sa_nonworsening"].mean())
                    if len(seed_changed)
                    else np.nan
                ),
                "np_nondecreasing_fraction_valid_changed": (
                    float(seed_changed["np_nondecreasing"].mean())
                    if len(seed_changed)
                    else np.nan
                ),
                "sa_and_np_fraction_valid_changed": (
                    float(seed_changed["sa_and_np"].mean())
                    if len(seed_changed)
                    else np.nan
                ),
            }
            seed_records.append(record)
            per_seed_rows.append(record)

        seed_frame = pd.DataFrame(seed_records)
        compact_rows.append(
            {
                "method": method,
                "property": property_name,
                "weighting": "seed_weighted",
                "denominator": int(len(seed_frame)),
                "median_sa_score": _quantile(seed_frame["median_sa_score"], 0.50),
                "q25_sa_score": _quantile(seed_frame["median_sa_score"], 0.25),
                "q75_sa_score": _quantile(seed_frame["median_sa_score"], 0.75),
                "median_np_likeness": _quantile(
                    seed_frame["median_np_likeness"], 0.50
                ),
                "q25_np_likeness": _quantile(
                    seed_frame["median_np_likeness"], 0.25
                ),
                "q75_np_likeness": _quantile(
                    seed_frame["median_np_likeness"], 0.75
                ),
                "median_np_fragment_confidence": _quantile(
                    seed_frame["median_np_fragment_confidence"], 0.50
                ),
                "requested_candidates": int(len(group)),
                "valid_candidates": int(len(valid)),
                "distinct_canonical_structures": int(
                    valid["candidate_scoring_canonical_smiles"].nunique()
                ),
                "seed_count": int(len(seed_frame)),
                "delta_denominator_valid_changed": int(
                    seed_frame["valid_changed_candidates"].gt(0).sum()
                ),
                "median_delta_sa_vs_seed": _quantile(
                    seed_frame["median_delta_sa_vs_seed"], 0.50
                ),
                "median_delta_np_vs_seed": _quantile(
                    seed_frame["median_delta_np_vs_seed"], 0.50
                ),
                "median_delta_np_confidence_vs_seed": _quantile(
                    seed_frame["median_delta_np_confidence_vs_seed"], 0.50
                ),
                "sa_nonworsening_fraction_valid_changed": _quantile(
                    seed_frame["sa_nonworsening_fraction_valid_changed"], 0.50
                ),
                "np_nondecreasing_fraction_valid_changed": _quantile(
                    seed_frame["np_nondecreasing_fraction_valid_changed"], 0.50
                ),
                "sa_and_np_fraction_valid_changed": _quantile(
                    seed_frame["sa_and_np_fraction_valid_changed"], 0.50
                ),
            }
        )

    return pd.DataFrame(compact_rows), pd.DataFrame(per_seed_rows)


def signed_alpha_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, property_name, alpha), group in candidates.groupby(
        ["method", "property", "alpha"], sort=False
    ):
        valid = group[group["is_rdkit_valid"]]
        changed = valid[valid["changed_from_seed"]]
        rows.append(
            {
                "method": method,
                "property": property_name,
                "alpha": float(alpha),
                "latent_displacement": abs(float(alpha)),
                "requested_candidates": int(len(group)),
                "valid_candidates": int(len(valid)),
                "valid_fraction": float(len(valid) / len(group)),
                "distinct_canonical_structures": int(
                    valid["candidate_scoring_canonical_smiles"].nunique()
                ),
                "median_sa_score": _quantile(
                    valid["candidate_sa_score"], 0.50
                ),
                "median_delta_sa_vs_seed": _quantile(
                    changed["delta_sa_vs_seed"], 0.50
                ),
                "median_np_likeness": _quantile(
                    valid["candidate_np_likeness"], 0.50
                ),
                "median_delta_np_vs_seed": _quantile(
                    changed["delta_np_vs_seed"], 0.50
                ),
                "median_np_fragment_confidence": _quantile(
                    valid["candidate_np_fragment_confidence"], 0.50
                ),
                "sa_nonworsening_fraction_valid_changed": (
                    float(changed["sa_nonworsening"].mean())
                    if len(changed)
                    else np.nan
                ),
                "np_nondecreasing_fraction_valid_changed": (
                    float(changed["np_nondecreasing"].mean())
                    if len(changed)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def matched_property_change_tables(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank-pair expected improvements; mismatch is always reported."""

    rows = []
    eligible = candidates[
        candidates["is_rdkit_valid"]
        & candidates["changed_from_seed"]
        & candidates["expected_signed_improvement"].gt(0)
    ]
    for (property_name, seed_id, alpha_sign), group in eligible.groupby(
        ["property", "seed_id", "alpha_sign"], sort=False
    ):
        by_method = {}
        for method in METHODS:
            subset = group[group["method"].eq(method)].sort_values(
                [
                    "expected_signed_improvement",
                    "latent_displacement",
                    "alpha",
                    "canonical_smiles",
                ],
                kind="mergesort",
            )
            by_method[method] = subset.reset_index(drop=True)
        pair_count = min(len(by_method[METHOD_OLS]), len(by_method[METHOD_CHEMSPACE]))
        for rank in range(pair_count):
            raw = by_method[METHOD_OLS].iloc[rank]
            chem = by_method[METHOD_CHEMSPACE].iloc[rank]
            rows.append(
                {
                    "property": property_name,
                    "seed_id": int(seed_id),
                    "alpha_sign": alpha_sign,
                    "rank": rank,
                    "raw_expected_improvement": raw[
                        "expected_signed_improvement"
                    ],
                    "chemspace_expected_improvement": chem[
                        "expected_signed_improvement"
                    ],
                    "absolute_improvement_mismatch": abs(
                        raw["expected_signed_improvement"]
                        - chem["expected_signed_improvement"]
                    ),
                    "raw_delta_sa_vs_seed": raw["delta_sa_vs_seed"],
                    "chemspace_delta_sa_vs_seed": chem["delta_sa_vs_seed"],
                    "chemspace_minus_raw_delta_sa": (
                        chem["delta_sa_vs_seed"] - raw["delta_sa_vs_seed"]
                    ),
                    "raw_delta_np_vs_seed": raw["delta_np_vs_seed"],
                    "chemspace_delta_np_vs_seed": chem["delta_np_vs_seed"],
                    "chemspace_minus_raw_delta_np": (
                        chem["delta_np_vs_seed"] - raw["delta_np_vs_seed"]
                    ),
                    "raw_np_fragment_confidence": raw[
                        "candidate_np_fragment_confidence"
                    ],
                    "chemspace_np_fragment_confidence": chem[
                        "candidate_np_fragment_confidence"
                    ],
                    "chemspace_minus_raw_seed_similarity": (
                        chem["seed_similarity_tanimoto"]
                        - raw["seed_similarity_tanimoto"]
                    ),
                    "raw_latent_displacement": raw["latent_displacement"],
                    "chemspace_latent_displacement": chem["latent_displacement"],
                }
            )
    pairs = pd.DataFrame(rows)
    summary = (
        pairs.groupby(["property", "alpha_sign"], as_index=False)
        .agg(
            matched_pairs=("rank", "size"),
            median_absolute_improvement_mismatch=(
                "absolute_improvement_mismatch",
                "median",
            ),
            p95_absolute_improvement_mismatch=(
                "absolute_improvement_mismatch",
                lambda values: values.quantile(0.95),
            ),
            median_chemspace_minus_raw_delta_sa=(
                "chemspace_minus_raw_delta_sa",
                "median",
            ),
            median_chemspace_minus_raw_delta_np=(
                "chemspace_minus_raw_delta_np",
                "median",
            ),
            median_chemspace_minus_raw_seed_similarity=(
                "chemspace_minus_raw_seed_similarity",
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
        .sort_values(["property", "alpha_sign"])
        .reset_index(drop=True)
    )
    return pairs, summary


def exact_matched_latent_quality_tables(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair methods exactly by property, seed, and signed alpha."""

    columns = [
        "property",
        "seed_id",
        "alpha",
        "alpha_sign",
        "latent_displacement",
        "is_rdkit_valid",
        "changed_from_seed",
        "expected_signed_improvement",
        "seed_similarity_tanimoto",
        "delta_sa_vs_seed",
        "delta_np_vs_seed",
        "candidate_np_fragment_confidence",
    ]
    raw = candidates[candidates["method"].eq(METHOD_OLS)][columns]
    chemspace = candidates[candidates["method"].eq(METHOD_CHEMSPACE)][columns]
    pairs = raw.merge(
        chemspace,
        on=["property", "seed_id", "alpha", "alpha_sign", "latent_displacement"],
        suffixes=("_raw_ols", "_chemspace"),
        validate="one_to_one",
    )
    if len(pairs) != 30_000:
        raise AssertionError(f"Expected 30,000 exact latent pairs, found {len(pairs)}")
    pairs["both_valid"] = (
        pairs["is_rdkit_valid_raw_ols"].astype(bool)
        & pairs["is_rdkit_valid_chemspace"].astype(bool)
    )
    pairs["both_valid_changed"] = (
        pairs["both_valid"]
        & pairs["changed_from_seed_raw_ols"].astype(bool)
        & pairs["changed_from_seed_chemspace"].astype(bool)
    )
    pairs["chemspace_minus_raw_delta_sa"] = (
        pairs["delta_sa_vs_seed_chemspace"]
        - pairs["delta_sa_vs_seed_raw_ols"]
    )
    pairs["chemspace_minus_raw_delta_np"] = (
        pairs["delta_np_vs_seed_chemspace"]
        - pairs["delta_np_vs_seed_raw_ols"]
    )
    pairs["chemspace_minus_raw_seed_similarity"] = (
        pairs["seed_similarity_tanimoto_chemspace"]
        - pairs["seed_similarity_tanimoto_raw_ols"]
    )
    pairs["chemspace_minus_raw_expected_improvement"] = (
        pairs["expected_signed_improvement_chemspace"]
        - pairs["expected_signed_improvement_raw_ols"]
    )

    rows = []
    for (property_name, alpha_sign), group in pairs.groupby(
        ["property", "alpha_sign"], sort=False
    ):
        evaluable = group[group["both_valid_changed"]]
        rows.append(
            {
                "property": property_name,
                "alpha_sign": alpha_sign,
                "requested_exact_pairs": int(len(group)),
                "both_valid_pairs": int(group["both_valid"].sum()),
                "both_valid_changed_pairs": int(len(evaluable)),
                "median_chemspace_minus_raw_delta_sa": _quantile(
                    evaluable["chemspace_minus_raw_delta_sa"], 0.50
                ),
                "q25_chemspace_minus_raw_delta_sa": _quantile(
                    evaluable["chemspace_minus_raw_delta_sa"], 0.25
                ),
                "q75_chemspace_minus_raw_delta_sa": _quantile(
                    evaluable["chemspace_minus_raw_delta_sa"], 0.75
                ),
                "chemspace_strictly_lower_delta_sa_fraction": (
                    float(
                        evaluable["delta_sa_vs_seed_chemspace"].lt(
                            evaluable["delta_sa_vs_seed_raw_ols"]
                        ).mean()
                    )
                    if len(evaluable)
                    else np.nan
                ),
                "equal_delta_sa_fraction": (
                    float(
                        evaluable["delta_sa_vs_seed_chemspace"].eq(
                            evaluable["delta_sa_vs_seed_raw_ols"]
                        ).mean()
                    )
                    if len(evaluable)
                    else np.nan
                ),
                "median_chemspace_minus_raw_delta_np": _quantile(
                    evaluable["chemspace_minus_raw_delta_np"], 0.50
                ),
                "median_chemspace_minus_raw_seed_similarity": _quantile(
                    evaluable["chemspace_minus_raw_seed_similarity"], 0.50
                ),
                "median_chemspace_minus_raw_expected_improvement": _quantile(
                    evaluable["chemspace_minus_raw_expected_improvement"], 0.50
                ),
            }
        )
    return pairs, pd.DataFrame(rows)


def matched_seed_similarity_quality_tables(
    candidates: pd.DataFrame,
    source_path: Path = BASE_OUTPUTS / "matched_seed_similarity_pairs.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach SA/NP scores to the existing deterministic similarity-rank pairs."""

    pairs = pd.read_csv(source_path)
    score_columns = [
        "property",
        "seed_id",
        "alpha",
        "delta_sa_vs_seed",
        "delta_np_vs_seed",
        "candidate_np_fragment_confidence",
        "changed_from_seed",
    ]
    raw = candidates[candidates["method"].eq(METHOD_OLS)][score_columns].rename(
        columns={
            "alpha": "raw_alpha",
            "delta_sa_vs_seed": "raw_delta_sa_vs_seed",
            "delta_np_vs_seed": "raw_delta_np_vs_seed",
            "candidate_np_fragment_confidence": "raw_np_fragment_confidence",
            "changed_from_seed": "raw_changed_from_seed",
        }
    )
    chemspace = candidates[
        candidates["method"].eq(METHOD_CHEMSPACE)
    ][score_columns].rename(
        columns={
            "alpha": "chemspace_alpha",
            "delta_sa_vs_seed": "chemspace_delta_sa_vs_seed",
            "delta_np_vs_seed": "chemspace_delta_np_vs_seed",
            "candidate_np_fragment_confidence": (
                "chemspace_np_fragment_confidence"
            ),
            "changed_from_seed": "chemspace_changed_from_seed",
        }
    )
    pairs = pairs.merge(
        raw,
        on=["property", "seed_id", "raw_alpha"],
        how="left",
        validate="one_to_one",
    )
    pairs = pairs.merge(
        chemspace,
        on=["property", "seed_id", "chemspace_alpha"],
        how="left",
        validate="one_to_one",
    )
    score_columns_attached = [
        "raw_delta_sa_vs_seed",
        "chemspace_delta_sa_vs_seed",
        "raw_delta_np_vs_seed",
        "chemspace_delta_np_vs_seed",
    ]
    if pairs[score_columns_attached].isna().any().any():
        raise AssertionError("A valid seed-similarity pair is missing SA/NP scores")
    pairs["chemspace_minus_raw_delta_sa"] = (
        pairs["chemspace_delta_sa_vs_seed"] - pairs["raw_delta_sa_vs_seed"]
    )
    pairs["chemspace_minus_raw_delta_np"] = (
        pairs["chemspace_delta_np_vs_seed"] - pairs["raw_delta_np_vs_seed"]
    )

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
            median_chemspace_minus_raw_delta_sa=(
                "chemspace_minus_raw_delta_sa",
                "median",
            ),
            q25_chemspace_minus_raw_delta_sa=(
                "chemspace_minus_raw_delta_sa",
                lambda values: values.quantile(0.25),
            ),
            q75_chemspace_minus_raw_delta_sa=(
                "chemspace_minus_raw_delta_sa",
                lambda values: values.quantile(0.75),
            ),
            chemspace_strictly_lower_delta_sa_fraction=(
                "chemspace_minus_raw_delta_sa",
                lambda values: values.lt(0).mean(),
            ),
            median_chemspace_minus_raw_delta_np=(
                "chemspace_minus_raw_delta_np",
                "median",
            ),
            median_improvement_difference_chemspace_minus_raw=(
                "improvement_difference_chemspace_minus_raw",
                "median",
            ),
        )
        .sort_values(["property", "alpha_sign"])
        .reset_index(drop=True)
    )
    return pairs, summary


def adjacent_transition_tables(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify local decode continuity between consecutive signed-alpha rows."""

    ordered = candidates.sort_values(
        ["method", "property", "seed_id", "alpha"], kind="mergesort"
    ).copy()
    groups = ordered.groupby(["method", "property", "seed_id"], sort=False)
    previous_columns = [
        "alpha",
        "is_rdkit_valid",
        "candidate_scoring_canonical_smiles",
        "candidate_sa_score",
        "candidate_np_likeness",
        "property_delta",
    ]
    for column in previous_columns:
        ordered[f"previous_{column}"] = groups[column].shift(1)
    transitions = ordered[ordered["previous_alpha"].notna()].copy()
    if len(transitions) != 59_400:
        raise AssertionError(
            f"Expected 59,400 adjacent transitions, found {len(transitions)}"
        )
    transitions["alpha_step"] = transitions["alpha"] - transitions["previous_alpha"]
    transitions["both_valid"] = (
        transitions["is_rdkit_valid"].astype(bool)
        & transitions["previous_is_rdkit_valid"].astype(bool)
    )
    transitions["structure_changed_across_step"] = (
        transitions["both_valid"]
        & transitions["candidate_scoring_canonical_smiles"].ne(
            transitions["previous_candidate_scoring_canonical_smiles"]
        )
    )

    scoring_smiles = sorted(
        set(
            transitions.loc[
                transitions["both_valid"], "candidate_scoring_canonical_smiles"
            ].dropna()
        ).union(
            set(
                transitions.loc[
                    transitions["both_valid"],
                    "previous_candidate_scoring_canonical_smiles",
                ].dropna()
            )
        )
    )
    fingerprints = {}
    for smiles in scoring_smiles:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise AssertionError(f"Scoring canonical failed parsing: {smiles}")
        fingerprints[str(smiles)] = MORGAN_GENERATOR.GetFingerprint(molecule)

    adjacent_similarity = []
    for row in transitions.itertuples(index=False):
        if not bool(row.both_valid):
            adjacent_similarity.append(np.nan)
            continue
        current = fingerprints[str(row.candidate_scoring_canonical_smiles)]
        previous = fingerprints[
            str(row.previous_candidate_scoring_canonical_smiles)
        ]
        adjacent_similarity.append(
            float(DataStructs.TanimotoSimilarity(previous, current))
        )
    transitions["adjacent_tanimoto"] = adjacent_similarity
    transitions["sa_step"] = (
        transitions["candidate_sa_score"]
        - transitions["previous_candidate_sa_score"]
    )
    transitions["absolute_sa_step"] = transitions["sa_step"].abs()
    transitions["np_step"] = (
        transitions["candidate_np_likeness"]
        - transitions["previous_candidate_np_likeness"]
    )
    transitions["absolute_np_step"] = transitions["np_step"].abs()
    transitions["property_step"] = (
        transitions["property_delta"] - transitions["previous_property_delta"]
    )
    transitions["absolute_property_step"] = transitions["property_step"].abs()

    per_seed_rows = []
    for (method, property_name, seed_id), group in transitions.groupby(
        ["method", "property", "seed_id"], sort=False
    ):
        evaluable = group[group["both_valid"]]
        changed = evaluable[evaluable["structure_changed_across_step"]]
        per_seed_rows.append(
            {
                "method": method,
                "property": property_name,
                "seed_id": int(seed_id),
                "requested_transitions": int(len(group)),
                "evaluable_transitions": int(len(evaluable)),
                "structure_changed_transitions": int(len(changed)),
                "structure_changed_fraction_evaluable": (
                    float(len(changed) / len(evaluable))
                    if len(evaluable)
                    else np.nan
                ),
                "median_adjacent_tanimoto": _quantile(
                    evaluable["adjacent_tanimoto"], 0.50
                ),
                "q25_adjacent_tanimoto": _quantile(
                    evaluable["adjacent_tanimoto"], 0.25
                ),
                "p05_adjacent_tanimoto": _quantile(
                    evaluable["adjacent_tanimoto"], 0.05
                ),
                "median_absolute_sa_step_changed": _quantile(
                    changed["absolute_sa_step"], 0.50
                ),
                "p95_absolute_sa_step_changed": _quantile(
                    changed["absolute_sa_step"], 0.95
                ),
                "median_absolute_np_step_changed": _quantile(
                    changed["absolute_np_step"], 0.50
                ),
                "p95_absolute_np_step_changed": _quantile(
                    changed["absolute_np_step"], 0.95
                ),
                "median_absolute_property_step_changed": _quantile(
                    changed["absolute_property_step"], 0.50
                ),
                "p95_absolute_property_step_changed": _quantile(
                    changed["absolute_property_step"], 0.95
                ),
            }
        )
    per_seed = pd.DataFrame(per_seed_rows)

    compact_rows = []
    for (method, property_name), group in per_seed.groupby(
        ["method", "property"], sort=False
    ):
        compact_rows.append(
            {
                "method": method,
                "property": property_name,
                "seed_count": int(len(group)),
                "requested_transitions": int(group["requested_transitions"].sum()),
                "evaluable_transitions": int(group["evaluable_transitions"].sum()),
                "median_seed_structure_changed_fraction": _quantile(
                    group["structure_changed_fraction_evaluable"], 0.50
                ),
                "median_seed_adjacent_tanimoto": _quantile(
                    group["median_adjacent_tanimoto"], 0.50
                ),
                "median_seed_q25_adjacent_tanimoto": _quantile(
                    group["q25_adjacent_tanimoto"], 0.50
                ),
                "median_seed_p05_adjacent_tanimoto": _quantile(
                    group["p05_adjacent_tanimoto"], 0.50
                ),
                "median_seed_median_absolute_sa_step_changed": _quantile(
                    group["median_absolute_sa_step_changed"], 0.50
                ),
                "median_seed_p95_absolute_sa_step_changed": _quantile(
                    group["p95_absolute_sa_step_changed"], 0.50
                ),
                "median_seed_median_absolute_np_step_changed": _quantile(
                    group["median_absolute_np_step_changed"], 0.50
                ),
                "median_seed_p95_absolute_np_step_changed": _quantile(
                    group["p95_absolute_np_step_changed"], 0.50
                ),
                "median_seed_median_absolute_property_step_changed": _quantile(
                    group["median_absolute_property_step_changed"], 0.50
                ),
                "median_seed_p95_absolute_property_step_changed": _quantile(
                    group["p95_absolute_property_step_changed"], 0.50
                ),
            }
        )
    return transitions, per_seed, pd.DataFrame(compact_rows)


def _rule_mask(frame: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "none":
        return pd.Series(True, index=frame.index)
    if rule == "sa_nonworsening":
        return frame["sa_nonworsening"]
    if rule == "np_nondecreasing":
        return frame["np_nondecreasing"]
    if rule == "sa_and_np":
        return frame["sa_and_np"]
    raise KeyError(rule)


def quality_adjusted_constrained_design(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for method in METHODS:
        for property_name in PROPERTIES:
            property_rows = candidates[
                candidates["method"].eq(method)
                & candidates["property"].eq(property_name)
            ]
            for seed_id in range(50):
                seed_rows = property_rows[property_rows["seed_id"].eq(seed_id)]
                if len(seed_rows) != 100:
                    raise AssertionError(
                        f"{method}/{property_name}/seed {seed_id}: {len(seed_rows)}"
                    )
                base = seed_rows[
                    seed_rows["is_rdkit_valid"] & seed_rows["changed_from_seed"]
                ].copy()
                for objective in ("maximize", "minimize"):
                    base["objective_improvement"] = (
                        base["property_delta"]
                        if objective == "maximize"
                        else -base["property_delta"]
                    )
                    for cutoff in SIMILARITY_CUTOFFS:
                        similarity_eligible = base[
                            base["seed_similarity_tanimoto"].ge(cutoff)
                            & base["objective_improvement"].gt(0)
                        ]
                        for rule in QUALITY_RULES:
                            eligible = similarity_eligible[
                                _rule_mask(similarity_eligible, rule)
                            ].sort_values(
                                [
                                    "objective_improvement",
                                    "seed_similarity_tanimoto",
                                    "latent_displacement",
                                    "canonical_smiles",
                                ],
                                ascending=[False, False, True, True],
                                kind="mergesort",
                            )
                            selected = eligible.iloc[0] if len(eligible) else None
                            rows.append(
                                {
                                    "method": method,
                                    "property": property_name,
                                    "seed_id": seed_id,
                                    "objective": objective,
                                    "seed_similarity_cutoff": cutoff,
                                    "quality_rule": rule,
                                    "candidate_budget": int(len(seed_rows)),
                                    "eligible_candidates": int(len(eligible)),
                                    "success": selected is not None,
                                    "best_objective_improvement": (
                                        selected["objective_improvement"]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_alpha": (
                                        selected["alpha"]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_seed_similarity": (
                                        selected["seed_similarity_tanimoto"]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_scaffold_retained": (
                                        selected["scaffold_retained"]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_sa_score": (
                                        selected["candidate_sa_score"]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_delta_sa_vs_seed": (
                                        selected["delta_sa_vs_seed"]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_np_likeness": (
                                        selected["candidate_np_likeness"]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_delta_np_vs_seed": (
                                        selected["delta_np_vs_seed"]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_np_fragment_confidence": (
                                        selected[
                                            "candidate_np_fragment_confidence"
                                        ]
                                        if selected is not None
                                        else np.nan
                                    ),
                                    "selected_canonical_smiles": (
                                        selected["canonical_smiles"]
                                        if selected is not None
                                        else None
                                    ),
                                }
                            )
    per_seed = pd.DataFrame(rows)
    compact = (
        per_seed.groupby(
            [
                "method",
                "property",
                "objective",
                "seed_similarity_cutoff",
                "quality_rule",
            ],
            as_index=False,
        )
        .agg(
            seeds=("seed_id", "size"),
            successful_seeds=("success", "sum"),
            success_fraction=("success", "mean"),
            median_best_improvement=(
                "best_objective_improvement",
                "median",
            ),
            median_selected_similarity=(
                "selected_seed_similarity",
                "median",
            ),
            median_selected_delta_sa=(
                "selected_delta_sa_vs_seed",
                "median",
            ),
            median_selected_delta_np=(
                "selected_delta_np_vs_seed",
                "median",
            ),
            median_selected_np_confidence=(
                "selected_np_fragment_confidence",
                "median",
            ),
            selected_scaffold_retention=(
                "selected_scaffold_retained",
                "mean",
            ),
        )
        .sort_values(
            [
                "objective",
                "quality_rule",
                "property",
                "seed_similarity_cutoff",
                "method",
            ]
        )
        .reset_index(drop=True)
    )
    return per_seed, compact


def create_figures(
    per_alpha: pd.DataFrame,
    constrained: pd.DataFrame,
    adjacent_per_seed: pd.DataFrame,
    output_dir: Path = OUTPUTS,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    colors = {METHOD_OLS: "#2457A7", METHOD_CHEMSPACE: "#D95F02"}

    for value, ylabel, filename in (
        ("median_delta_sa_vs_seed", "median SA change vs seed", "sa_change_vs_alpha.png"),
        ("median_delta_np_vs_seed", "median NP-likeness change vs seed", "np_change_vs_alpha.png"),
    ):
        figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
        for axis, property_name in zip(axes.flat, PROPERTIES):
            for method in METHODS:
                group = per_alpha[
                    per_alpha["property"].eq(property_name)
                    & per_alpha["method"].eq(method)
                ].sort_values("alpha")
                axis.plot(
                    group["alpha"],
                    group[value],
                    color=colors[method],
                    label=method,
                )
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_title(property_name)
            axis.set_xlabel(r"$\alpha$")
            axis.set_ylabel(ylabel)
        axes.flat[0].legend(frameon=False, fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=250, bbox_inches="tight")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=adjacent_per_seed,
        x="structure_changed_fraction_evaluable",
        y="property",
        hue="method",
        order=list(PROPERTIES),
        hue_order=list(METHODS),
        palette=colors,
        whis=(5, 95),
        showfliers=False,
        ax=axis,
    )
    axis.set_xlabel("per-seed fraction of adjacent steps changing exact structure")
    axis.set_ylabel("")
    axis.set_yticks(range(len(PROPERTIES)), labels=list(PROPERTIES))
    axis.set_xlim(0, 1)
    axis.legend(frameon=False, fontsize=8, title="", loc="lower right")
    figure.tight_layout()
    figure.savefig(
        output_dir / "adjacent_decode_continuity.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(figure)

    for rule, filename, title_suffix in (
        (
            "sa_nonworsening",
            "sa_preserving_constrained_success.png",
            "SA nonworsening",
        ),
        (
            "np_nondecreasing",
            "np_preserving_constrained_success.png",
            "NP-likeness nondecreasing",
        ),
    ):
        plot = constrained[
            constrained["objective"].eq("maximize")
            & constrained["quality_rule"].isin(["none", rule])
        ]
        figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        for axis, property_name in zip(axes.flat, PROPERTIES):
            for method in METHODS:
                for quality_rule, linestyle in (("none", "--"), (rule, "-")):
                    group = plot[
                        plot["property"].eq(property_name)
                        & plot["method"].eq(method)
                        & plot["quality_rule"].eq(quality_rule)
                    ].sort_values("seed_similarity_cutoff")
                    label = (
                        f"{method}: {title_suffix}"
                        if quality_rule == rule
                        else f"{method}: property only"
                    )
                    axis.plot(
                        group["seed_similarity_cutoff"],
                        group["success_fraction"],
                        color=colors[method],
                        linestyle=linestyle,
                        marker="o",
                        label=label,
                    )
            axis.set_title(property_name)
            axis.set_xlabel("seed-similarity cutoff")
            axis.set_ylabel("success fraction")
            axis.set_ylim(0, 1.02)
        axes.flat[0].legend(frameon=False, fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=250, bbox_inches="tight")
        plt.close(figure)


def verify_outputs(output_dir: Path = OUTPUTS) -> pd.DataFrame:
    checks = []

    def record(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    provenance = pd.read_csv(output_dir / "scorer_provenance.csv")
    record(
        "official scorer assets",
        len(provenance) == 4 and provenance["sha256"].str.len().eq(64).all(),
        provenance[["artifact", "rdkit_version"]].to_dict("records").__str__(),
    )
    candidate = pd.read_csv(output_dir / "candidate_quality_scores.csv")
    valid = candidate["is_rdkit_valid"].astype(bool)
    record("60,000 candidate rows", len(candidate) == 60_000, str(len(candidate)))
    record(
        "valid scores complete",
        candidate.loc[
            valid,
            [
                "candidate_sa_score",
                "candidate_np_likeness",
                "candidate_np_fragment_confidence",
            ],
        ].notna().all().all(),
        str(int(valid.sum())),
    )
    record(
        "invalid scores absent",
        candidate.loc[
            ~valid,
            [
                "candidate_sa_score",
                "candidate_np_likeness",
                "candidate_np_fragment_confidence",
            ],
        ].isna().all().all(),
        str(int((~valid).sum())),
    )
    record(
        "SA range",
        candidate.loc[valid, "candidate_sa_score"].between(1, 10).all(),
        str(
            (
                candidate.loc[valid, "candidate_sa_score"].min(),
                candidate.loc[valid, "candidate_sa_score"].max(),
            )
        ),
    )
    record(
        "NP confidence range",
        candidate.loc[valid, "candidate_np_fragment_confidence"]
        .between(0, 1)
        .all(),
        str(
            (
                candidate.loc[valid, "candidate_np_fragment_confidence"].min(),
                candidate.loc[valid, "candidate_np_fragment_confidence"].max(),
            )
        ),
    )
    canonical_consistency = (
        candidate.loc[valid]
        .groupby("candidate_scoring_canonical_smiles")[
            [
                "candidate_sa_score",
                "candidate_np_likeness",
                "candidate_np_fragment_confidence",
            ]
        ]
        .nunique(dropna=False)
        .le(1)
        .all()
        .all()
    )
    record("canonical score consistency", canonical_consistency, "exact")
    mapping = pd.read_csv(output_dir / "structure_score_mapping.csv")
    unique_scores = pd.read_csv(output_dir / "unique_structure_scores.csv")
    record(
        "scorer-version structure identity",
        mapping["scoring_canonical_smiles"].nunique() == len(unique_scores),
        (
            f"{len(mapping)} source spellings -> "
            f"{len(unique_scores)} scorer-version structures; "
            f"{int((~mapping['source_matches_scoring_canonical']).sum())} "
            "spellings recanonicalized"
        ),
    )

    compact = pd.read_csv(output_dir / "quality_summary.csv")
    record(
        "three weighting views",
        set(compact["weighting"]) == {
            "candidate_occurrence",
            "unique_structure",
            "seed_weighted",
        },
        str(sorted(compact["weighting"].unique())),
    )
    constrained = pd.read_csv(output_dir / "quality_adjusted_constrained_summary.csv")
    record(
        "quality rules explicit",
        set(constrained["quality_rule"]) == set(QUALITY_RULES),
        str(sorted(constrained["quality_rule"].unique())),
    )
    original = pd.read_csv(BASE_OUTPUTS / "constrained_design_compact_summary.csv")
    reproduced = constrained[constrained["quality_rule"].eq("none")]
    keys = ["method", "property", "objective", "seed_similarity_cutoff"]
    comparison = original[keys + ["successful_seeds"]].merge(
        reproduced[keys + ["successful_seeds"]],
        on=keys,
        suffixes=("_original", "_quality"),
        validate="one_to_one",
    )
    record(
        "property-only constrained success reproduced",
        comparison["successful_seeds_original"].eq(
            comparison["successful_seeds_quality"]
        ).all(),
        str(len(comparison)),
    )
    exact_pairs = pd.read_csv(output_dir / "matched_latent_quality_pairs.csv")
    record(
        "exact latent pair count",
        len(exact_pairs) == 30_000,
        str(len(exact_pairs)),
    )
    similarity_pairs = pd.read_csv(
        output_dir / "matched_seed_similarity_quality_pairs.csv"
    )
    source_similarity_pairs = pd.read_csv(
        BASE_OUTPUTS / "matched_seed_similarity_pairs.csv"
    )
    record(
        "matched-similarity pair preservation",
        len(similarity_pairs) == len(source_similarity_pairs),
        str(len(similarity_pairs)),
    )
    transitions = pd.read_csv(output_dir / "adjacent_transition_quality.csv")
    record(
        "adjacent transition count",
        len(transitions) == 59_400,
        str(len(transitions)),
    )
    return pd.DataFrame(checks)


def run_analysis(output_dir: Path = OUTPUTS) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = scorer_provenance()
    raw_candidates = load_candidates()
    structure_scores = score_unique_structures(raw_candidates)
    candidates = attach_scores(raw_candidates, structure_scores)
    compact, per_seed = quality_summary_tables(candidates)
    per_alpha = signed_alpha_summary(candidates)
    endpoint_summary = per_alpha[per_alpha["alpha"].abs().eq(150)].copy()
    matched_pairs, matched_summary = matched_property_change_tables(candidates)
    exact_pairs, exact_summary = exact_matched_latent_quality_tables(candidates)
    similarity_pairs, similarity_summary = (
        matched_seed_similarity_quality_tables(candidates)
    )
    transitions, transition_per_seed, transition_summary = (
        adjacent_transition_tables(candidates)
    )
    constrained_per_seed, constrained_summary = quality_adjusted_constrained_design(
        candidates
    )

    _atomic_csv(provenance, output_dir / "scorer_provenance.csv")
    _atomic_csv(structure_scores, output_dir / "structure_score_mapping.csv")
    _atomic_csv(
        structure_scores.drop_duplicates("scoring_canonical_smiles"),
        output_dir / "unique_structure_scores.csv",
    )
    _atomic_csv(candidates, output_dir / "candidate_quality_scores.csv")
    _atomic_csv(compact, output_dir / "quality_summary.csv")
    _atomic_csv(per_seed, output_dir / "quality_per_seed.csv")
    _atomic_csv(per_alpha, output_dir / "quality_per_signed_alpha.csv")
    _atomic_csv(
        endpoint_summary, output_dir / "endpoint_quality_summary.csv"
    )
    _atomic_csv(
        matched_pairs, output_dir / "matched_property_change_quality_pairs.csv"
    )
    _atomic_csv(
        matched_summary, output_dir / "matched_property_change_quality_summary.csv"
    )
    _atomic_csv(exact_pairs, output_dir / "matched_latent_quality_pairs.csv")
    _atomic_csv(exact_summary, output_dir / "matched_latent_quality_summary.csv")
    _atomic_csv(
        similarity_pairs,
        output_dir / "matched_seed_similarity_quality_pairs.csv",
    )
    _atomic_csv(
        similarity_summary,
        output_dir / "matched_seed_similarity_quality_summary.csv",
    )
    _atomic_csv(
        transitions, output_dir / "adjacent_transition_quality.csv"
    )
    _atomic_csv(
        transition_per_seed, output_dir / "adjacent_transition_per_seed.csv"
    )
    _atomic_csv(
        transition_summary, output_dir / "adjacent_transition_summary.csv"
    )
    _atomic_csv(
        constrained_per_seed,
        output_dir / "quality_adjusted_constrained_per_seed.csv",
    )
    _atomic_csv(
        constrained_summary,
        output_dir / "quality_adjusted_constrained_summary.csv",
    )
    create_figures(
        per_alpha,
        constrained_summary,
        transition_per_seed,
        output_dir=output_dir,
    )
    verification = verify_outputs(output_dir)
    _atomic_csv(verification, output_dir / "verification_checks.csv")
    return {
        "provenance": provenance,
        "structure_scores": structure_scores,
        "candidates": candidates,
        "summary": compact,
        "per_seed": per_seed,
        "per_alpha": per_alpha,
        "endpoint_summary": endpoint_summary,
        "matched_summary": matched_summary,
        "exact_matched_summary": exact_summary,
        "matched_similarity_summary": similarity_summary,
        "transition_summary": transition_summary,
        "constrained_summary": constrained_summary,
        "verification": verification,
    }
