"""Locked, one-time Phase 10 evaluation of the frozen multilabel ML component.

This module contains reusable synthetic-testable helpers and the guarded real-data
execution. It never fits, refits, tunes, calibrates, resamples, or selects features.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "final_test_evaluation.v1.json"
LABELS: tuple[str, ...] = ("Vata", "Pitta", "Kapha")
TARGETS: tuple[str, ...] = ("dosha_vata", "dosha_pitta", "dosha_kapha")
PROHIBITED_POSITIVE_LANGUAGE: tuple[str, ...] = (
    "clinical accuracy",
    "correct ayurvedic diagnosis",
    "true dosha identification",
    "treatment effectiveness",
    "medically safe",
    "expert validated",
    "clinical confidence",
)
DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|milligrams?|g|grams?|mcg|µg|ml|"
    r"millilit(?:er|re)s?|tablets?|capsules?|drops?)\b",
    re.IGNORECASE,
)


class FinalTestEvaluationError(RuntimeError):
    """Raised on any locked-protocol, integrity, or one-time-guard failure."""


@dataclass(frozen=True)
class SplitPlan:
    """Programmatically validated split membership without displaying its content."""

    test_ids: tuple[str, ...]
    expected_labels: Mapping[str, tuple[int, int, int]]
    expected_test_row_count: int


class OneBatchPredictor:
    """Permit exactly one probability call for one complete feature batch."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.call_count = 0

    def predict_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        if self.call_count != 0:
            raise FinalTestEvaluationError("More than one batch prediction call is forbidden.")
        self.call_count += 1
        values = np.asarray(self.pipeline.predict_proba(features), dtype=float)
        expected = (len(features), len(LABELS))
        if values.shape != expected or not np.isfinite(values).all():
            raise FinalTestEvaluationError(
                f"Expected finite probability matrix {expected}; observed {values.shape}."
            )
        return values


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise FinalTestEvaluationError("Unsupported Phase 10 configuration version.")
    if tuple(config.get("label_order", [])) != LABELS:
        raise FinalTestEvaluationError("Frozen label order differs from Vata, Pitta, Kapha.")
    if tuple(config.get("feature_fields", [])) != ("symptoms",):
        raise FinalTestEvaluationError("Frozen feature schema differs from symptoms only.")
    if tuple(config.get("label_fields", [])) != TARGETS:
        raise FinalTestEvaluationError("Frozen target schema differs from approval.")
    frozen = config["frozen_model"]
    if frozen.get("threshold_comparison_operator") != "greater_than_or_equal":
        raise FinalTestEvaluationError("Frozen threshold operator differs from >=.")
    thresholds = tuple(float(frozen["thresholds"][label]) for label in LABELS)
    if thresholds != (0.45, 0.45, 0.45):
        raise FinalTestEvaluationError("Frozen thresholds differ from 0.45.")
    if config["metrics"].get("zero_division") != 0:
        raise FinalTestEvaluationError("Metric zero_division policy differs from 0.")
    return config


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "sqlite": sqlite3.sqlite_version,
    }


def verify_runtime_versions(config: Mapping[str, Any]) -> dict[str, Any]:
    observed = runtime_versions()
    required = config["required_runtime_versions"]
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in required.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise FinalTestEvaluationError(f"Runtime version mismatch: {mismatches}")
    return {"required": dict(required), "observed": observed, "passed": True}


def verify_artifact_hashes(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, artifact in config["artifacts"].items():
        path = project_root / artifact["path"]
        if not path.is_file():
            raise FinalTestEvaluationError(f"Required artifact missing: {artifact['path']}")
        observed = file_sha256(path)
        expected = artifact["sha256"]
        if observed != expected:
            raise FinalTestEvaluationError(
                f"Hash mismatch for {artifact['path']}: expected {expected}, observed {observed}."
            )
        results[name] = {
            "path": artifact["path"], "expected_sha256": expected,
            "observed_sha256": observed, "passed": True,
        }
    return results


def verify_production_database(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    artifact = config["artifacts"]["production_database"]
    path = project_root / artifact["path"]
    before = file_sha256(path)
    if before != artifact["sha256"]:
        raise FinalTestEvaluationError("Production database hash differs from the Phase 9 lock.")
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in config["production_tables_required_empty"]
        }
        violations = list(connection.execute("PRAGMA foreign_key_check"))
    finally:
        connection.close()
    if any(counts.values()) or violations:
        raise FinalTestEvaluationError(
            f"Production database integrity failure: counts={counts}, foreign_keys={len(violations)}."
        )
    after = file_sha256(path)
    if after != before:
        raise FinalTestEvaluationError("Read-only database verification changed the database bytes.")
    return {
        "path": artifact["path"], "sha256_before": before, "sha256_after": after,
        "byte_for_byte_unchanged": True, "production_table_row_counts": counts,
        "foreign_key_violation_count": 0,
    }


def load_and_validate_frozen_bundle(
    project_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = config["artifacts"]["frozen_model_bundle"]
    path = project_root / artifact["path"]
    if file_sha256(path) != artifact["sha256"]:
        raise FinalTestEvaluationError("Frozen model bundle hash mismatch before loading.")
    bundle = joblib.load(path)
    required_keys = {
        "pipeline", "thresholds", "feature_columns", "candidate",
        "score_semantics", "final_test_evaluated",
    }
    if not isinstance(bundle, dict) or set(bundle) != required_keys:
        raise FinalTestEvaluationError("Frozen model bundle structure differs from approval.")
    frozen = config["frozen_model"]
    if bundle["candidate"] != frozen["candidate"]:
        raise FinalTestEvaluationError("Frozen candidate identifier mismatch.")
    if bundle["feature_columns"] != config["feature_fields"]:
        raise FinalTestEvaluationError("Frozen bundle feature fields mismatch.")
    if bundle["final_test_evaluated"] is not False:
        raise FinalTestEvaluationError("Frozen bundle final-test flag is not false.")
    expected_thresholds = np.asarray(
        [frozen["thresholds"][label] for label in LABELS], dtype=float
    )
    observed_thresholds = np.asarray(bundle["thresholds"], dtype=float)
    if not np.array_equal(expected_thresholds, observed_thresholds):
        raise FinalTestEvaluationError("Frozen bundle thresholds mismatch.")

    pipeline = bundle["pipeline"]
    if not isinstance(pipeline, Pipeline) or [name for name, _ in pipeline.steps] != [
        "preprocessor", "classifier"
    ]:
        raise FinalTestEvaluationError("Unexpected frozen sklearn pipeline structure.")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise FinalTestEvaluationError("Frozen preprocessor is not ColumnTransformer.")
    if not isinstance(classifier, OneVsRestClassifier):
        raise FinalTestEvaluationError("Frozen classifier is not OneVsRestClassifier.")
    if not isinstance(classifier.estimator, LogisticRegression):
        raise FinalTestEvaluationError("Frozen base estimator is not LogisticRegression.")
    parameters = classifier.estimator.get_params()
    expected_parameters = {
        "C": float(frozen["C"]),
        "class_weight": frozen["class_weight"],
        "random_state": int(frozen["random_state"]),
        "solver": "liblinear",
        "max_iter": 2000,
    }
    for key, expected in expected_parameters.items():
        if parameters.get(key) != expected:
            raise FinalTestEvaluationError(
                f"Frozen Logistic Regression parameter {key} differs: {parameters.get(key)!r}."
            )
    check_is_fitted(preprocessor)
    check_is_fitted(classifier)
    if len(classifier.estimators_) != 3:
        raise FinalTestEvaluationError("Frozen OneVsRestClassifier does not contain three estimators.")
    transformer_columns = [
        list(columns) for name, _, columns in preprocessor.transformers_
        if name == "symptoms_tfidf"
    ]
    if transformer_columns != [["symptoms"]] or len(preprocessor.transformers_) != 1:
        raise FinalTestEvaluationError("Frozen preprocessing is not symptoms-only TF-IDF.")
    validation = {
        "bundle_sha256": artifact["sha256"],
        "pipeline_type": "sklearn.pipeline.Pipeline",
        "preprocessor_type": "sklearn.compose.ColumnTransformer",
        "classifier_type": "sklearn.multiclass.OneVsRestClassifier",
        "base_estimator_type": "sklearn.linear_model.LogisticRegression",
        "parameters": expected_parameters,
        "feature_fields": ["symptoms"],
        "label_order": list(LABELS),
        "thresholds": dict(frozen["thresholds"]),
        "already_fitted": True,
        "fit_calls_permitted": 0,
        "passed": True,
    }
    return bundle, validation


def validate_locked_split_metadata(
    project_root: Path, config: Mapping[str, Any]
) -> tuple[SplitPlan, dict[str, Any]]:
    """Read locked metadata programmatically without printing row-level content."""

    manifest_path = project_root / config["artifacts"]["split_manifest"]["path"]
    assignment_path = project_root / config["artifacts"]["split_assignments"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cleaned_dataset_sha256") != config["artifacts"]["cleaned_dataset"]["sha256"]:
        raise FinalTestEvaluationError("Split manifest is bound to a different cleaned dataset hash.")
    if tuple(manifest.get("target_columns", [])) != TARGETS:
        raise FinalTestEvaluationError("Split-manifest target order differs from the frozen order.")
    rows = manifest.get("assignments")
    if not isinstance(rows, list) or not rows:
        raise FinalTestEvaluationError("Locked split manifest has no assignments.")
    required = {
        config["identifier_field"], config["group_field"], config["split_field"], *TARGETS
    }
    if any(set(row) != required for row in rows if isinstance(row, dict)) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise FinalTestEvaluationError("Split assignment fields are missing or unexpected.")
    identifiers = [str(row[config["identifier_field"]]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise FinalTestEvaluationError("Duplicate split identifiers detected.")
    partitions = {"train", "validation", "test"}
    if {str(row[config["split_field"]]) for row in rows} != partitions:
        raise FinalTestEvaluationError("Locked split does not contain exactly three partitions.")
    group_partitions: dict[str, set[str]] = {}
    for row in rows:
        group = str(row[config["group_field"]])
        group_partitions.setdefault(group, set()).add(str(row[config["split_field"]]))
        for target in TARGETS:
            if str(row[target]) not in {"0", "1"}:
                raise FinalTestEvaluationError("Split manifest contains a non-binary target.")
    if any(len(values) != 1 for values in group_partitions.values()):
        raise FinalTestEvaluationError("A normalized disease group overlaps partitions.")

    with assignment_path.open(encoding="utf-8", newline="") as source:
        csv_rows = list(csv.DictReader(source))
        fieldnames = set(source.name and (csv_rows[0].keys() if csv_rows else []))
    if fieldnames != required:
        raise FinalTestEvaluationError("Split-assignment CSV fields differ from the manifest.")
    normalized_manifest = sorted(
        tuple(str(row[field]) for field in sorted(required)) for row in rows
    )
    normalized_csv = sorted(
        tuple(str(row[field]) for field in sorted(required)) for row in csv_rows
    )
    if normalized_csv != normalized_manifest:
        raise FinalTestEvaluationError("Split-assignment CSV differs from the locked manifest.")

    test_rows = [row for row in rows if row[config["split_field"]] == config["test_split_value"]]
    test_ids = tuple(str(row[config["identifier_field"]]) for row in test_rows)
    if not test_ids:
        raise FinalTestEvaluationError("Locked final-test partition is empty.")
    expected_labels = {
        str(row[config["identifier_field"]]): tuple(int(row[target]) for target in TARGETS)
        for row in test_rows
    }
    plan = SplitPlan(
        test_ids=test_ids,
        expected_labels=expected_labels,
        expected_test_row_count=len(test_ids),
    )
    return plan, {
        "split_manifest_and_csv_identical": True,
        "identifiers_unique": True,
        "all_partitions_present": True,
        "normalized_disease_group_overlap_count": 0,
        "targets_binary": True,
        "test_row_count_source": "locked_manifest_computed_without_display",
        "passed": True,
    }


def consume_one_time_guard(
    project_root: Path,
    config: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
) -> Path:
    guard = config["one_time_guard"]
    paths = {
        name: project_root / guard[name]
        for name in ("started_marker", "completion_marker", "abort_marker")
    }
    final_outputs = [
        project_root / relative for relative in config["expected_output_files"]
        if relative not in {
            guard["started_marker"], guard["completion_marker"],
            "outputs/phase10_final_evaluation/amended_pre_execution_manifest.json",
            "outputs/phase10_final_evaluation/amended_integrity_checks.json",
        }
    ]
    existing = [str(path) for path in [*paths.values(), *final_outputs] if path.exists()]
    if existing:
        raise FinalTestEvaluationError(
            f"One-time execution or result artifact already exists; overwrite refused: {existing}"
        )
    marker = paths["started_marker"]
    payload = {
        "phase": 10,
        "status": "one_time_guard_consumed_before_cleaned_data_read",
        "consumed_at_utc": utc_now(),
        "authorization": "PHASE10-AMENDMENT-001",
        "split_manifest_metadata_exposures": 1,
        "cleaned_final_test_data_read_executions_at_marker_creation": 0,
        "batch_prediction_calls_at_marker_creation": 0,
        "locked_code_and_configuration_hashes": dict(lock_hashes),
        "rerun_without_new_amendment_permitted": False,
    }
    write_json_exclusive(marker, payload)
    return marker


def stream_locked_test_rows_once(
    cleaned_path: Path,
    plan: SplitPlan,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Open the cleaned CSV once, retaining only the locked ID, feature and targets."""

    identifier = config["identifier_field"]
    selected_columns = [identifier, *config["feature_fields"], *TARGETS]
    test_ids = set(plan.test_ids)
    selected: list[dict[str, Any]] = []
    observed_test_ids: set[str] = set()
    with cleaned_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(set(selected_columns) - fieldnames)
        if missing:
            raise FinalTestEvaluationError(f"Cleaned data is missing required columns: {missing}")
        for row in reader:
            profile_id = str(row[identifier])
            if profile_id not in test_ids:
                continue
            if profile_id in observed_test_ids:
                raise FinalTestEvaluationError("Duplicate final-test identifier in cleaned data.")
            observed_test_ids.add(profile_id)
            record = {column: row[column] for column in selected_columns}
            for target in TARGETS:
                if str(record[target]) not in {"0", "1"}:
                    raise FinalTestEvaluationError("Final-test target is not binary.")
                record[target] = int(record[target])
            expected = plan.expected_labels[profile_id]
            actual = tuple(int(record[target]) for target in TARGETS)
            if actual != expected:
                raise FinalTestEvaluationError("Cleaned final-test label differs from split lock.")
            selected.append(record)
    if observed_test_ids != test_ids:
        raise FinalTestEvaluationError("One or more locked final-test identifiers are missing.")
    if len(selected) != plan.expected_test_row_count:
        raise FinalTestEvaluationError("Final-test row count differs from the locked manifest.")
    frame = pd.DataFrame(selected, columns=selected_columns)
    if list(frame.columns) != selected_columns:
        raise FinalTestEvaluationError("Selected final-test columns are missing or unexpected.")
    return frame


def apply_frozen_thresholds(
    probabilities: np.ndarray, thresholds: Sequence[float]
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    limits = np.asarray(tuple(thresholds), dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or limits.shape != (3,):
        raise FinalTestEvaluationError("Probability or threshold shape is not n x 3 / length 3.")
    return (values >= limits.reshape(1, -1)).astype(int)


def confusion_counts(
    truth: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, int]]:
    matrices = multilabel_confusion_matrix(truth, predicted)
    return {
        label: {
            "tn": int(matrices[index, 0, 0]),
            "fp": int(matrices[index, 0, 1]),
            "fn": int(matrices[index, 1, 0]),
            "tp": int(matrices[index, 1, 1]),
        }
        for index, label in enumerate(LABELS)
    }


def bootstrap_intervals(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    resamples: int,
    seed: int,
    percentiles: Sequence[float],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    macro_values = np.empty(resamples, dtype=float)
    micro_values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.integers(0, len(truth), size=len(truth))
        macro_values[index] = f1_score(
            truth[sample], predicted[sample], average="macro", zero_division=0
        )
        micro_values[index] = f1_score(
            truth[sample], predicted[sample], average="micro", zero_division=0
        )
    lower, upper = (float(percentiles[0]), float(percentiles[1]))
    return {
        "method": "record_level_resampling_with_replacement",
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence_level": (upper - lower) / 100.0,
        "macro_f1": {
            "lower_percentile": lower,
            "lower": float(np.percentile(macro_values, lower)),
            "upper_percentile": upper,
            "upper": float(np.percentile(macro_values, upper)),
        },
        "micro_f1": {
            "lower_percentile": lower,
            "lower": float(np.percentile(micro_values, lower)),
            "upper_percentile": upper,
            "upper": float(np.percentile(micro_values, upper)),
        },
        "estimator_refitted": False,
        "warning": (
            "Intervals may be unstable for this small grouped final-test partition and "
            "do not represent clinical uncertainty."
        ),
    }


def calculate_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise FinalTestEvaluationError("Truth and predictions must share n x 3 shape.")
    if not set(np.unique(truth)) <= {0, 1} or not set(np.unique(predicted)) <= {0, 1}:
        raise FinalTestEvaluationError("Truth and predictions must be binary.")
    zero_division = int(config["metrics"]["zero_division"])
    precision, recall, per_f1, support = precision_recall_fscore_support(
        truth, predicted, average=None, zero_division=zero_division
    )
    confusion = confusion_counts(truth, predicted)
    combinations = Counter(
        "+".join(label for label, decision in zip(LABELS, row, strict=True) if decision)
        or "ABSTAIN"
        for row in predicted
    )
    abstention_count = int((predicted.sum(axis=1) == 0).sum())
    macro = float(f1_score(truth, predicted, average="macro", zero_division=zero_division))
    micro = float(f1_score(truth, predicted, average="micro", zero_division=zero_division))
    validation_macro = float(config["validation_comparison"]["frozen_validation_macro_f1"])
    bootstrap = config["bootstrap"]
    return {
        "scope": "held_out_dataset_label_agreement_not_clinical_accuracy",
        "primary_metric": {"name": "macro_f1", "value": macro},
        "secondary_metrics": {
            "micro_f1": micro,
            "weighted_f1": float(
                f1_score(truth, predicted, average="weighted", zero_division=zero_division)
            ),
            "samples_f1": float(
                f1_score(truth, predicted, average="samples", zero_division=zero_division)
            ),
            "jaccard_samples": float(
                jaccard_score(truth, predicted, average="samples", zero_division=zero_division)
            ),
            "exact_match_subset_accuracy": float(accuracy_score(truth, predicted)),
            "hamming_loss": float(hamming_loss(truth, predicted)),
        },
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
                **confusion[label],
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_counts": confusion,
        "label_prevalence": {
            label: {
                "positive_count": int(truth[:, index].sum()),
                "profile_count": int(len(truth)),
                "prevalence": float(truth[:, index].mean()),
            }
            for index, label in enumerate(LABELS)
        },
        "cardinality": {
            "true_mean_labels_per_profile": float(truth.sum(axis=1).mean()),
            "predicted_mean_labels_per_profile": float(predicted.sum(axis=1).mean()),
        },
        "abstention": {
            "definition": "zero labels meeting their frozen thresholds",
            "count": abstention_count,
            "rate": float(abstention_count / len(truth)),
        },
        "predicted_label_combinations": [
            {
                "combination": combination,
                "count": int(count),
                "fraction": float(count / len(truth)),
            }
            for combination, count in sorted(combinations.items())
        ],
        "bootstrap_intervals": bootstrap_intervals(
            truth,
            predicted,
            resamples=int(bootstrap["resamples"]),
            seed=int(bootstrap["seed"]),
            percentiles=bootstrap["percentiles"],
        ),
        "validation_comparison": {
            "frozen_validation_macro_f1": validation_macro,
            "final_test_macro_f1": macro,
            "test_minus_validation": float(macro - validation_macro),
            "interpretation": "observed_generalization_gap_without_significance_claim",
            "pre_registered_success_threshold": None,
        },
        "zero_division": zero_division,
    }


def build_prediction_audit(
    profile_ids: Sequence[str],
    truth: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float],
    predicted: np.ndarray,
    expected_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    limits = np.asarray(tuple(thresholds), dtype=float)
    for index, profile_id in enumerate(profile_ids):
        selected = [
            label for label, value in zip(LABELS, predicted[index], strict=True) if value
        ]
        rows.append(
            {
                "knowledge_profile_id": profile_id,
                "true_vata": int(truth[index, 0]),
                "true_pitta": int(truth[index, 1]),
                "true_kapha": int(truth[index, 2]),
                "probability_vata": float(probabilities[index, 0]),
                "probability_pitta": float(probabilities[index, 1]),
                "probability_kapha": float(probabilities[index, 2]),
                "threshold_vata": float(limits[0]),
                "threshold_pitta": float(limits[1]),
                "threshold_kapha": float(limits[2]),
                "prediction_vata": int(predicted[index, 0]),
                "prediction_pitta": int(predicted[index, 1]),
                "prediction_kapha": int(predicted[index, 2]),
                "predicted_label_set": "+".join(selected) if selected else "ABSTAIN",
                "abstention": bool(not selected),
                "exact_match": bool(np.array_equal(truth[index], predicted[index])),
            }
        )
    frame = pd.DataFrame(rows, columns=list(expected_columns))
    if list(frame.columns) != list(expected_columns):
        raise FinalTestEvaluationError("Prediction audit column schema differs from lock.")
    forbidden = {"symptoms", "normalized_disease", "disease"} & set(frame.columns)
    if forbidden:
        raise FinalTestEvaluationError(f"Prediction audit contains forbidden content: {forbidden}")
    return frame


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as target:
            json.dump(payload, target, indent=2, ensure_ascii=False)
            target.write("\n")
    except FileExistsError as error:
        raise FinalTestEvaluationError(f"Output overwrite refused: {path}") from error


def write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as target:
            frame.to_csv(target, index=False, lineterminator="\n")
    except FileExistsError as error:
        raise FinalTestEvaluationError(f"Output overwrite refused: {path}") from error


def scan_prohibited_output_text(paths: Sequence[Path]) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {"exact_dosage": [], "prohibited_positive_language": []}
    for path in paths:
        text = path.read_text(encoding="utf-8").casefold()
        findings["exact_dosage"].extend(
            f"{path.name}:{match.group(0)}" for match in DOSAGE_PATTERN.finditer(text)
        )
        findings["prohibited_positive_language"].extend(
            f"{path.name}:{phrase}" for phrase in PROHIBITED_POSITIVE_LANGUAGE if phrase in text
        )
    return findings


def execute_one_time_evaluation(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume the guard, read once, predict once, and finalize immutable artifacts."""

    plan, split_checks = validate_locked_split_metadata(project_root, config)
    bundle, model_checks = load_and_validate_frozen_bundle(project_root, config)
    guard_path = consume_one_time_guard(project_root, config, lock_hashes)
    output_dir = project_root / "outputs" / "phase10_final_evaluation"
    abort_path = project_root / config["one_time_guard"]["abort_marker"]
    counters = {
        "split_manifest_metadata_exposures": 1,
        "aborted_pre_execution_attempts": 2,
        "cleaned_final_test_data_read_executions": 0,
        "final_test_rows_loaded": 0,
        "final_test_symptom_fields_loaded": 0,
        "batch_prediction_calls": 0,
        "metric_calculation_executions": 0,
        "error_analysis_executions": 0,
        "successful_evaluation_runs": 0,
        "aborted_evaluation_runs": 0,
    }
    try:
        counters["cleaned_final_test_data_read_executions"] = 1
        final_frame = stream_locked_test_rows_once(
            project_root / config["artifacts"]["cleaned_dataset"]["path"],
            plan,
            config,
        )
        counters["final_test_rows_loaded"] = int(len(final_frame))
        counters["final_test_symptom_fields_loaded"] = int(len(final_frame))
        features = final_frame.loc[:, config["feature_fields"]]
        truth = final_frame.loc[:, config["label_fields"]].to_numpy(dtype=int)
        predictor = OneBatchPredictor(bundle["pipeline"])
        probabilities = predictor.predict_probabilities(features)
        counters["batch_prediction_calls"] = predictor.call_count
        thresholds = [config["frozen_model"]["thresholds"][label] for label in LABELS]
        predicted = apply_frozen_thresholds(probabilities, thresholds)
        counters["metric_calculation_executions"] = 1
        metrics = calculate_metrics(truth, predicted, config)
        prediction_audit = build_prediction_audit(
            final_frame[config["identifier_field"]].astype(str).tolist(),
            truth,
            probabilities,
            thresholds,
            predicted,
            config["prediction_audit_columns"],
        )

        metrics_path = output_dir / "final_test_metrics.json"
        predictions_path = output_dir / "final_test_predictions.csv"
        confusion_path = output_dir / "final_test_confusion_counts.json"
        write_json_exclusive(metrics_path, metrics)
        write_csv_exclusive(predictions_path, prediction_audit)
        write_json_exclusive(
            confusion_path,
            {
                "label_order": list(LABELS),
                "profile_count": int(len(truth)),
                "confusion_counts": metrics["confusion_counts"],
                "scope": "dataset_label_agreement_counts_not_clinical_errors",
            },
        )
        output_hashes = {
            "final_test_metrics.json": file_sha256(metrics_path),
            "final_test_predictions.csv": file_sha256(predictions_path),
            "final_test_confusion_counts.json": file_sha256(confusion_path),
        }
        findings = scan_prohibited_output_text(
            [metrics_path, predictions_path, confusion_path]
        )
        if any(findings.values()):
            raise FinalTestEvaluationError(f"Prohibited output finding: {findings}")
        counters["successful_evaluation_runs"] = 1
        database = verify_production_database(project_root, config)
        completion_path = project_root / config["one_time_guard"]["completion_marker"]
        completion_payload = {
            "phase": 10,
            "status": "one_time_final_test_evaluation_complete",
            "completed_at_utc": utc_now(),
            "authorization": ["PHASE10-AMENDMENT-001", "PHASE10-AMENDMENT-002"],
            "one_time_guard_consumed": True,
            "rerun_permitted": False,
            "access_counters": counters,
            "artifact_hashes": output_hashes,
            "locked_code_and_configuration_hashes": dict(lock_hashes),
            "model_or_threshold_changes_after_test": 0,
        }
        write_json_exclusive(completion_path, completion_payload)
        completion_hash = file_sha256(completion_path)

        manifest_path = output_dir / "final_test_evaluation_manifest.json"
        incident_manifest = config["artifacts"]["incident_pre_unsealing_manifest"]
        incident_integrity = config["artifacts"]["incident_integrity_checks"]
        amendment = config["artifacts"]["protocol_amendment_001"]
        manifest = {
            "phase": 10,
            "status": "one_time_final_test_evaluation_complete",
            "completed_at_utc": completion_payload["completed_at_utc"],
            "protocol_incident": {
                "perfect_pre_unsealing_blindness": False,
                "split_manifest_metadata_exposures": 1,
                "aborted_pre_execution_attempts": 2,
                "false_lock_file_collision": True,
                "continuation_authority": [
                    "PHASE10-AMENDMENT-001", "PHASE10-AMENDMENT-002"
                ],
                "cleaned_rows_or_predictions_at_exposure": 0,
                "possible_effect_dismissed": False,
            },
            "protocol_hashes": {
                "original_pre_unsealing_manifest": incident_manifest["sha256"],
                "original_integrity_checks": incident_integrity["sha256"],
                "protocol_amendment_001": amendment["sha256"],
                "protocol_amendment_002": file_sha256(
                    output_dir / "protocol_amendment_002.json"
                ),
                "amended_pre_execution_manifest": file_sha256(
                    output_dir / "amended_pre_execution_manifest.json"
                ),
                "amended_integrity_checks": file_sha256(
                    output_dir / "amended_integrity_checks.json"
                ),
                "amended_pre_execution_manifest_v2": file_sha256(
                    output_dir / "amended_pre_execution_manifest_v2.json"
                ),
                "amended_integrity_checks_v2": file_sha256(
                    output_dir / "amended_integrity_checks_v2.json"
                ),
                "second_pre_execution_abort": file_sha256(
                    output_dir / "AMENDED_PRE_EXECUTION_ABORT_002.json"
                ),
            },
            "locked_code_and_configuration": dict(lock_hashes),
            "frozen_artifact_hashes": {
                name: result["observed_sha256"]
                for name, result in preflight["artifact_hash_checks"].items()
            },
            "runtime_versions": preflight["runtime_versions"]["observed"],
            "access_counters": counters,
            "test_partition": {
                "profile_count": int(len(truth)),
                "per_label_decision_count": int(truth.size),
                "label_order": list(LABELS),
                "label_prevalence": metrics["label_prevalence"],
            },
            "metrics": {
                "primary": metrics["primary_metric"],
                "secondary": metrics["secondary_metrics"],
                "per_label": metrics["per_label"],
                "bootstrap_intervals": metrics["bootstrap_intervals"],
                "abstention": metrics["abstention"],
                "validation_comparison": metrics["validation_comparison"],
            },
            "artifact_hashes": {
                **output_hashes,
                "FINAL_TEST_EVALUATION_STARTED.json": file_sha256(guard_path),
                "FINAL_TEST_EVALUATION_COMPLETE.json": completion_hash,
            },
            "production_database": database,
            "exact_dosage_findings": findings["exact_dosage"],
            "prohibited_language_findings": findings["prohibited_positive_language"],
            "integrity_warnings_or_deviations": [
                "One acknowledged pre-execution split-manifest metadata exposure; perfect pre-unsealing blindness cannot be claimed.",
                "Two pre-execution attempts aborted before cleaned-data access; the second was caused by a false collision with required Amendment 001 lock files.",
                "Continuation used Amendments 001 and 002 without changing the frozen model or evaluation definitions."
            ],
            "integrity_status": "passed_with_acknowledged_metadata_exposure",
            "post_test_changes": {
                "model_changes": 0,
                "preprocessing_changes": 0,
                "threshold_changes": 0,
                "split_changes": 0,
                "metric_definition_changes": 0,
                "retrieval_rule_or_hybrid_changes": 0,
            },
            "evidence_boundary": {
                "frozen_ml_model": "held-out agreement with dataset-assigned Dosha labels",
                "rules_retrieval_safety_gate": "synthetic implementation and traceability evidence only",
                "full_hybrid_pipeline": "functional educational prototype; no combined accuracy score",
            },
        }
        write_json_exclusive(manifest_path, manifest)
        return manifest
    except Exception as error:
        counters["aborted_evaluation_runs"] = 1
        if not abort_path.exists():
            write_json_exclusive(
                abort_path,
                {
                    "phase": 10,
                    "status": "one_time_execution_aborted_after_guard_consumption",
                    "aborted_at_utc": utc_now(),
                    "reason": str(error),
                    "access_counters": counters,
                    "rerun_permitted_without_new_amendment": False,
                },
            )
        raise
