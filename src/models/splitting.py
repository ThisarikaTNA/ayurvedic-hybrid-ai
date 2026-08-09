"""Deterministic, disease-group-disjoint splitting for Phase 3.

Scikit-learn has group-aware splitters and multilabel-aware metrics, but no
native grouped multilabel stratifier. This module therefore searches many
seeded group assignments and chooses the best approximate balance. It never
splits a normalized disease group across partitions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from data.preprocessing import FORBIDDEN_MODEL_COLUMNS, MODEL_FEATURE_COLUMNS


TARGET_COLUMNS: tuple[str, ...] = ("dosha_vata", "dosha_pitta", "dosha_kapha")
PARTITION_NAMES: tuple[str, ...] = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when split creation or verification fails."""


@dataclass(frozen=True)
class SplitConfig:
    """Reproducible split settings."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42
    candidate_count: int = 5000

    def validate(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions):
            raise SplitError("All split fractions must be positive.")
        if not np.isclose(sum(fractions), 1.0):
            raise SplitError("Split fractions must sum to 1.0.")
        if self.candidate_count < 1:
            raise SplitError("candidate_count must be positive.")


class TrainableTransformer(Protocol):
    """Minimal interface used by the training-only preprocessing guard."""

    def fit(self, values: pd.DataFrame) -> Any: ...
    def transform(self, values: pd.DataFrame) -> Any: ...


def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest without changing the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_split_input(dataframe: pd.DataFrame) -> None:
    """Validate identifiers, groups, targets, and the locked feature policy."""

    required = {
        "knowledge_profile_id", "normalized_disease", *TARGET_COLUMNS,
        *MODEL_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise SplitError(f"Cleaned dataset is missing required columns: {missing}")
    if dataframe["knowledge_profile_id"].duplicated().any():
        raise SplitError("knowledge_profile_id must be unique.")
    if dataframe["normalized_disease"].isna().any() or (
        dataframe["normalized_disease"].astype(str).str.strip() == ""
    ).any():
        raise SplitError("Every knowledge profile must have a normalized disease group.")
    for column in TARGET_COLUMNS:
        values = set(pd.to_numeric(dataframe[column], errors="raise").astype(int).unique())
        if not values <= {0, 1}:
            raise SplitError(f"Target {column} must be binary; observed {sorted(values)}")
    validate_model_features(MODEL_FEATURE_COLUMNS)


def validate_model_features(feature_columns: Iterable[str]) -> None:
    """Ensure no target, disease-name, or post-diagnosis field is selected."""

    selected = tuple(feature_columns)
    forbidden = sorted(set(selected) & FORBIDDEN_MODEL_COLUMNS)
    not_allowlisted = sorted(set(selected) - set(MODEL_FEATURE_COLUMNS))
    if forbidden or not_allowlisted:
        raise SplitError(
            f"Unsafe feature selection. Forbidden={forbidden}; "
            f"not_allowlisted={not_allowlisted}"
        )


def _combination_series(dataframe: pd.DataFrame) -> pd.Series:
    return dataframe.loc[:, TARGET_COLUMNS].astype(int).astype(str).agg("".join, axis=1)


def _split_summary(dataframe: pd.DataFrame, assignments: pd.Series) -> dict[str, Any]:
    overall_prevalence = dataframe.loc[:, TARGET_COLUMNS].astype(int).mean()
    result: dict[str, Any] = {}
    combinations = _combination_series(dataframe)
    for partition in PARTITION_NAMES:
        mask = assignments == partition
        subset = dataframe.loc[mask]
        counts = subset.loc[:, TARGET_COLUMNS].astype(int).sum()
        prevalence = subset.loc[:, TARGET_COLUMNS].astype(int).mean()
        result[partition] = {
            "profile_count": int(mask.sum()),
            "profile_fraction": round(float(mask.mean()), 6),
            "disease_group_count": int(subset["normalized_disease"].nunique()),
            "positive_label_counts": {column: int(counts[column]) for column in TARGET_COLUMNS},
            "label_prevalence": {
                column: round(float(prevalence[column]), 6) for column in TARGET_COLUMNS
            },
            "absolute_prevalence_difference_from_overall": {
                column: round(float(abs(prevalence[column] - overall_prevalence[column])), 6)
                for column in TARGET_COLUMNS
            },
            "combination_counts": {
                str(combo): int(count)
                for combo, count in combinations.loc[mask].value_counts().sort_index().items()
            },
        }
    return result


def _score_assignment(
    dataframe: pd.DataFrame,
    assignments: pd.Series,
    config: SplitConfig,
) -> float:
    """Penalize missing labels, size drift, and prevalence drift."""

    desired = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    overall_label = dataframe.loc[:, TARGET_COLUMNS].astype(float).mean()
    all_combinations = sorted(_combination_series(dataframe).unique())
    overall_combo = _combination_series(dataframe).value_counts(normalize=True)
    score = 0.0
    for partition in PARTITION_NAMES:
        mask = assignments == partition
        subset = dataframe.loc[mask]
        if subset.empty:
            return float("inf")
        counts = subset.loc[:, TARGET_COLUMNS].astype(int).sum()
        missing_labels = int((counts == 0).sum())
        score += 1000.0 * missing_labels
        score += 12.0 * abs(float(mask.mean()) - desired[partition])
        score += float((subset.loc[:, TARGET_COLUMNS].astype(float).mean() - overall_label).abs().sum())
        combo_prevalence = _combination_series(subset).value_counts(normalize=True)
        score += 0.35 * sum(
            abs(float(combo_prevalence.get(combo, 0.0) - overall_combo.get(combo, 0.0)))
            for combo in all_combinations
        )
    return score


def create_grouped_multilabel_split(
    dataframe: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Search seeded group assignments and return the best verified partition."""

    config.validate()
    validate_split_input(dataframe)
    groups = np.array(sorted(dataframe["normalized_disease"].astype(str).unique()))
    if len(groups) < 3:
        raise SplitError("At least three disease groups are required.")

    desired_group_counts = np.array(
        [config.train_fraction, config.validation_fraction, config.test_fraction]
    ) * len(groups)
    train_group_count = int(round(desired_group_counts[0]))
    validation_group_count = int(round(desired_group_counts[1]))
    train_group_count = min(max(train_group_count, 1), len(groups) - 2)
    validation_group_count = min(
        max(validation_group_count, 1), len(groups) - train_group_count - 1
    )

    # Precompute numeric arrays so thousands of candidates can be scored
    # without repeatedly constructing pandas group-bys.
    group_to_position = {group: position for position, group in enumerate(groups)}
    row_group_positions = (
        dataframe["normalized_disease"].astype(str).map(group_to_position).to_numpy(dtype=int)
    )
    label_matrix = dataframe.loc[:, TARGET_COLUMNS].astype(int).to_numpy(dtype=float)
    overall_label = label_matrix.mean(axis=0)
    combination_strings = _combination_series(dataframe)
    combination_names = sorted(combination_strings.unique())
    combination_to_position = {
        combination: position for position, combination in enumerate(combination_names)
    }
    combination_positions = combination_strings.map(combination_to_position).to_numpy(dtype=int)
    overall_combination = np.bincount(
        combination_positions, minlength=len(combination_names)
    ) / len(dataframe)
    desired_fractions = np.array(
        [config.train_fraction, config.validation_fraction, config.test_fraction]
    )

    rng = np.random.default_rng(config.random_state)
    best_score = float("inf")
    best_assignments: pd.Series | None = None
    best_candidate = -1
    for candidate_index in range(config.candidate_count):
        shuffled = rng.permutation(groups)
        group_partitions = np.full(len(groups), 2, dtype=np.int8)
        group_partitions[
            [group_to_position[group] for group in shuffled[:train_group_count]]
        ] = 0
        group_partitions[
            [
                group_to_position[group]
                for group in shuffled[
                    train_group_count : train_group_count + validation_group_count
                ]
            ]
        ] = 1
        row_partitions = group_partitions[row_group_positions]
        score = 0.0
        for partition_code in range(3):
            mask = row_partitions == partition_code
            row_count = int(mask.sum())
            if row_count == 0:
                score = float("inf")
                break
            label_counts = label_matrix[mask].sum(axis=0)
            score += 1000.0 * int((label_counts == 0).sum())
            score += 12.0 * abs(row_count / len(dataframe) - desired_fractions[partition_code])
            score += float(np.abs(label_matrix[mask].mean(axis=0) - overall_label).sum())
            candidate_combination = np.bincount(
                combination_positions[mask], minlength=len(combination_names)
            ) / row_count
            score += 0.35 * float(
                np.abs(candidate_combination - overall_combination).sum()
            )
        if score < best_score:
            best_score = score
            best_assignments = pd.Series(
                np.array(PARTITION_NAMES, dtype=object)[row_partitions],
                index=dataframe.index,
            )
            best_candidate = candidate_index

    if best_assignments is None:
        raise SplitError("No valid group assignment was found.")

    assignment_table = dataframe.loc[
        :, ["knowledge_profile_id", "normalized_disease", *TARGET_COLUMNS]
    ].copy()
    assignment_table["split"] = best_assignments.to_numpy()
    assignment_table = assignment_table.loc[
        :, ["knowledge_profile_id", "normalized_disease", "split", *TARGET_COLUMNS]
    ]
    verification = verify_assignments(dataframe, assignment_table)
    search = {
        "method": "seeded random search over normalized-disease group assignments",
        "candidate_count": config.candidate_count,
        "selected_candidate_zero_based": best_candidate,
        "selected_score": round(float(best_score), 8),
        "configuration": asdict(config),
        "verification": verification,
        "summary": _split_summary(dataframe, best_assignments),
    }
    return assignment_table, search


def verify_assignments(
    dataframe: pd.DataFrame, assignment_table: pd.DataFrame
) -> dict[str, Any]:
    """Verify coverage, group disjointness, target coverage, and feature safety."""

    if set(assignment_table["knowledge_profile_id"]) != set(dataframe["knowledge_profile_id"]):
        raise SplitError("Split assignments do not cover exactly the cleaned dataset.")
    if assignment_table["knowledge_profile_id"].duplicated().any():
        raise SplitError("A knowledge profile is assigned more than once.")
    if set(assignment_table["split"]) != set(PARTITION_NAMES):
        raise SplitError("All train, validation, and test partitions must be present.")

    group_sets = {
        partition: set(
            assignment_table.loc[
                assignment_table["split"] == partition, "normalized_disease"
            ].astype(str)
        )
        for partition in PARTITION_NAMES
    }
    overlaps = {
        "train_validation": sorted(group_sets["train"] & group_sets["validation"]),
        "train_test": sorted(group_sets["train"] & group_sets["test"]),
        "validation_test": sorted(group_sets["validation"] & group_sets["test"]),
    }
    if any(overlaps.values()):
        raise SplitError(f"Disease groups overlap between partitions: {overlaps}")

    label_coverage: dict[str, dict[str, bool]] = {}
    for partition in PARTITION_NAMES:
        subset = assignment_table[assignment_table["split"] == partition]
        label_coverage[partition] = {
            target: bool(pd.to_numeric(subset[target]).astype(int).sum() > 0)
            for target in TARGET_COLUMNS
        }
    if not all(all(values.values()) for values in label_coverage.values()):
        raise SplitError(f"A partition is missing a Dosha label: {label_coverage}")

    validate_model_features(MODEL_FEATURE_COLUMNS)
    return {
        "all_profiles_assigned_once": True,
        "disease_group_overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "all_three_labels_present": label_coverage,
        "feature_policy_passed": True,
        "preprocessing_fit_partition": "train only (enforced by fit_preprocessor_train_only)",
    }


def fit_preprocessor_train_only(
    preprocessor: TrainableTransformer,
    dataframe: pd.DataFrame,
    assignment_table: pd.DataFrame,
    feature_columns: Iterable[str] = MODEL_FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Fit once on training rows, then transform validation and final test rows.

    This helper is the Phase 3 programmatic contract used by Phase 4. The
    final-test rows are transformed but never used by ``fit``.
    """

    features = tuple(feature_columns)
    validate_model_features(features)
    merged = dataframe.merge(
        assignment_table.loc[:, ["knowledge_profile_id", "split"]],
        on="knowledge_profile_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise SplitError("Missing split assignment during preprocessing.")
    train_values = merged.loc[merged["split"] == "train", list(features)]
    preprocessor.fit(train_values)
    return {
        partition: preprocessor.transform(
            merged.loc[merged["split"] == partition, list(features)]
        )
        for partition in PARTITION_NAMES
    }


def build_manifest(
    cleaned_path: Path,
    assignment_table: pd.DataFrame,
    search_report: dict[str, Any],
) -> dict[str, Any]:
    """Create a locked, source-bound manifest payload."""

    return {
        "phase": 3,
        "cleaned_dataset_path": str(Path(cleaned_path).resolve()),
        "cleaned_dataset_sha256": file_sha256(cleaned_path),
        "random_state": search_report["configuration"]["random_state"],
        "method": search_report["method"],
        "final_test_policy": (
            "The final test partition is locked now and must not be used for "
            "feature selection, threshold selection, or hyperparameter tuning."
        ),
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "assignments": assignment_table.to_dict(orient="records"),
        "search_report": search_report,
    }


def write_or_validate_locked_manifest(path: Path, payload: dict[str, Any]) -> bool:
    """Write a manifest once, or validate that an existing lock is identical."""

    path = Path(path)
    serialized = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SplitError(
                "Existing locked split differs from the requested split. Preserve the "
                "lock; use a new explicitly named experiment instead of overwriting it."
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return True
