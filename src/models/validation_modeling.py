"""Leakage-safe Phase 4 model comparison on the validation partition only.

All fitted objects are scikit-learn pipelines. Text imputation, TF-IDF,
categorical imputation, and one-hot encoding are learned from training rows.
The final-test partition is outside this module's evaluation interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from data.preprocessing import FORBIDDEN_MODEL_COLUMNS, MODEL_FEATURE_COLUMNS
from models.splitting import TARGET_COLUMNS


RANDOM_STATE = 42
LABEL_NAMES: tuple[str, ...] = ("Vata", "Pitta", "Kapha")
TEXT_FEATURE = "symptoms"
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    column for column in MODEL_FEATURE_COLUMNS if column != TEXT_FEATURE
)
FEATURE_EXPERIMENTS: Mapping[str, tuple[str, ...]] = {
    "symptoms_only": (TEXT_FEATURE,),
    "symptoms_plus_categorical": (TEXT_FEATURE, *CATEGORICAL_FEATURES),
}
MODEL_NAMES: tuple[str, ...] = (
    "dummy_prior", "logistic_regression_ovr", "random_forest_ovr"
)
THRESHOLD_GRID: tuple[float, ...] = tuple(
    round(float(value), 2) for value in np.arange(0.20, 0.801, 0.05)
)


class ModelingError(ValueError):
    """Raised when Phase 4 safety or output validation fails."""


class FittablePipeline(Protocol):
    """Minimal interface for the train-only fitting guard and its tests."""

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> Any: ...
    def predict_proba(self, x: pd.DataFrame) -> Any: ...


def normalize_missing_matrix(values: Any) -> np.ndarray:
    """Convert blanks and pandas missing values to NumPy NaN inside a pipeline."""

    frame = pd.DataFrame(values, copy=True)
    return frame.map(
        lambda value: np.nan
        if pd.isna(value) or not str(value).strip()
        else str(value).strip()
    ).to_numpy(dtype=object)


def flatten_text_column(values: Any) -> np.ndarray:
    """Flatten one imputed text column for ``TfidfVectorizer``."""

    return np.asarray(values, dtype=object).reshape(-1).astype(str)


def validate_experiment_features(feature_columns: Iterable[str]) -> tuple[str, ...]:
    """Reject fields outside the approved Phase 2 allowlist."""

    selected = tuple(feature_columns)
    forbidden = sorted(set(selected) & FORBIDDEN_MODEL_COLUMNS)
    unapproved = sorted(set(selected) - set(MODEL_FEATURE_COLUMNS))
    if forbidden or unapproved or TEXT_FEATURE not in selected:
        raise ModelingError(
            f"Unsafe feature experiment. Forbidden={forbidden}; "
            f"unapproved={unapproved}; symptoms_present={TEXT_FEATURE in selected}"
        )
    return selected


def build_preprocessor(
    experiment_name: str,
    *,
    tfidf_min_df: int = 2,
) -> ColumnTransformer:
    """Create missing-safe TF-IDF and optional categorical one-hot transforms."""

    if experiment_name not in FEATURE_EXPERIMENTS:
        raise ModelingError(f"Unknown feature experiment: {experiment_name}")
    features = validate_experiment_features(FEATURE_EXPERIMENTS[experiment_name])
    text_pipeline = Pipeline(
        steps=[
            (
                "normalize_missing",
                FunctionTransformer(normalize_missing_matrix, validate=False),
            ),
            ("impute_missing", SimpleImputer(strategy="constant", fill_value="")),
            (
                "flatten_text",
                FunctionTransformer(flatten_text_column, validate=False),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=tfidf_min_df,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = [
        ("symptoms_tfidf", text_pipeline, [TEXT_FEATURE])
    ]
    if experiment_name == "symptoms_plus_categorical":
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "normalize_missing",
                    FunctionTransformer(normalize_missing_matrix, validate=False),
                ),
                (
                    "impute_missing",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformers.append(
            ("categorical_onehot", categorical_pipeline, list(features[1:]))
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def hyperparameter_candidates(model_name: str) -> list[dict[str, Any]]:
    """Return the intentionally small, documented Phase 4 search space."""

    if model_name == "dummy_prior":
        return [{"strategy": "prior"}]
    if model_name == "logistic_regression_ovr":
        return [
            {"C": c_value, "class_weight": class_weight}
            for c_value in (0.5, 1.0)
            for class_weight in (None, "balanced")
        ]
    if model_name == "random_forest_ovr":
        return [
            {
                "n_estimators": 200,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": "balanced_subsample",
            }
            for max_depth in (None, 12)
            for min_samples_leaf in (1, 3)
        ]
    raise ModelingError(f"Unknown model: {model_name}")


def build_pipeline(
    experiment_name: str,
    model_name: str,
    parameters: Mapping[str, Any],
    *,
    random_state: int = RANDOM_STATE,
    tfidf_min_df: int = 2,
) -> Pipeline:
    """Build one fully leakage-safe model pipeline."""

    preprocessor = build_preprocessor(experiment_name, tfidf_min_df=tfidf_min_df)
    if model_name == "dummy_prior":
        classifier = OneVsRestClassifier(
            DummyClassifier(strategy=str(parameters["strategy"]), random_state=random_state)
        )
    elif model_name == "logistic_regression_ovr":
        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=float(parameters["C"]),
                class_weight=parameters["class_weight"],
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            )
        )
    elif model_name == "random_forest_ovr":
        classifier = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=int(parameters["n_estimators"]),
                max_depth=parameters["max_depth"],
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                class_weight=parameters["class_weight"],
                random_state=random_state,
                n_jobs=1,
            )
        )
    else:
        raise ModelingError(f"Unknown model: {model_name}")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def positive_probabilities(pipeline: FittablePipeline, features: pd.DataFrame) -> np.ndarray:
    """Return one uncalibrated positive-class score/probability per label."""

    probabilities = pipeline.predict_proba(features)
    if isinstance(probabilities, list):
        columns = [np.asarray(values)[:, -1] for values in probabilities]
        result = np.column_stack(columns)
    else:
        result = np.asarray(probabilities, dtype=float)
    expected = (len(features), len(TARGET_COLUMNS))
    if result.shape != expected:
        raise ModelingError(
            f"Expected score shape {expected}; observed {result.shape}"
        )
    return result


def fit_pipeline_training_only(
    pipeline: FittablePipeline,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_validation: pd.DataFrame,
) -> np.ndarray:
    """Fit on training values exactly once and score validation without fitting it."""

    pipeline.fit(x_train, y_train)
    return positive_probabilities(pipeline, x_validation)


def apply_thresholds(
    probabilities: np.ndarray, thresholds: Iterable[float]
) -> np.ndarray:
    """Convert uncalibrated scores/probabilities into three binary outputs."""

    values = np.asarray(probabilities, dtype=float)
    threshold_array = np.asarray(tuple(thresholds), dtype=float)
    if values.ndim != 2 or values.shape[1] != len(TARGET_COLUMNS):
        raise ModelingError(f"Unexpected probability shape: {values.shape}")
    if threshold_array.shape != (len(TARGET_COLUMNS),):
        raise ModelingError(f"Unexpected threshold shape: {threshold_array.shape}")
    return (values >= threshold_array.reshape(1, -1)).astype(int)


def evaluate_predictions(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
) -> dict[str, Any]:
    """Calculate all requested multilabel validation metrics."""

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ModelingError(
            f"Truth and predictions must both be n x 3; got {truth.shape}, {predicted.shape}"
        )
    precision, recall, per_label_f1, support = precision_recall_fscore_support(
        truth, predicted, average=None, zero_division=0
    )
    confusion = multilabel_confusion_matrix(truth, predicted)
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(truth, predicted, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(truth, predicted)),
        "exact_match_subset_accuracy": float(accuracy_score(truth, predicted)),
        "jaccard_samples": float(
            jaccard_score(truth, predicted, average="samples", zero_division=0)
        ),
        "actual_label_prevalence": {
            label: float(truth[:, index].mean())
            for index, label in enumerate(LABEL_NAMES)
        },
        "predicted_label_prevalence": {
            label: float(predicted[:, index].mean())
            for index, label in enumerate(LABEL_NAMES)
        },
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_label_f1[index]),
                "positive_support": int(support[index]),
                "negative_support": int(len(truth) - support[index]),
            }
            for index, label in enumerate(LABEL_NAMES)
        },
        "multilabel_confusion_matrices": {
            label: {
                "true_negative": int(confusion[index, 0, 0]),
                "false_positive": int(confusion[index, 0, 1]),
                "false_negative": int(confusion[index, 1, 0]),
                "true_positive": int(confusion[index, 1, 1]),
            }
            for index, label in enumerate(LABEL_NAMES)
        },
    }


def tune_per_label_thresholds(
    y_true: pd.DataFrame | np.ndarray,
    probabilities: np.ndarray,
    threshold_grid: Iterable[float] = THRESHOLD_GRID,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Choose each label threshold by validation F1, preferring proximity to 0.5."""

    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    grid = tuple(float(value) for value in threshold_grid)
    selected: list[float] = []
    rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(LABEL_NAMES):
        scored: list[tuple[float, float]] = []
        for threshold in grid:
            predictions = (scores[:, label_index] >= threshold).astype(int)
            label_f1 = float(
                f1_score(truth[:, label_index], predictions, zero_division=0)
            )
            scored.append((threshold, label_f1))
        best_threshold, best_f1 = max(
            scored, key=lambda item: (item[1], -abs(item[0] - 0.5), -item[0])
        )
        selected.append(best_threshold)
        for threshold, label_f1 in scored:
            rows.append(
                {
                    "label": label,
                    "threshold": threshold,
                    "validation_f1": label_f1,
                    "selected": threshold == best_threshold,
                    "selected_validation_f1": best_f1,
                }
            )
    return np.asarray(selected, dtype=float), rows


def aggregate_metric_row(
    candidate_name: str,
    experiment_name: str,
    model_name: str,
    threshold_mode: str,
    thresholds: Iterable[float],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten aggregate metrics for the validation comparison CSV."""

    threshold_values = tuple(float(value) for value in thresholds)
    return {
        "candidate": candidate_name,
        "feature_experiment": experiment_name,
        "model": model_name,
        "threshold_mode": threshold_mode,
        "threshold_vata": threshold_values[0],
        "threshold_pitta": threshold_values[1],
        "threshold_kapha": threshold_values[2],
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "samples_f1": metrics["samples_f1"],
        "hamming_loss": metrics["hamming_loss"],
        "exact_match_subset_accuracy": metrics["exact_match_subset_accuracy"],
        "jaccard_samples": metrics["jaccard_samples"],
    }


def save_confusion_figure(
    metrics: Mapping[str, Any], title: str, output_path: Path
) -> None:
    """Render three validation multilabel confusion matrices."""

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for index, label in enumerate(LABEL_NAMES):
        values = metrics["multilabel_confusion_matrices"][label]
        matrix = np.array(
            [
                [values["true_negative"], values["false_positive"]],
                [values["false_negative"], values["true_positive"]],
            ]
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=axes[index],
        )
        axes[index].set_title(label)
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_comparison_figure(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot validation Macro-F1 for default and tuned thresholds."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 6))
    sns.barplot(
        data=comparison,
        x="candidate",
        y="macro_f1",
        hue="threshold_mode",
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("")
    plt.ylabel("Validation Macro-F1")
    plt.title("Phase 4 validation comparison — final test sealed")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_prevalence_figure(
    metrics: Mapping[str, Any], title: str, output_path: Path
) -> None:
    """Plot actual versus predicted validation label prevalence."""

    records = []
    for label in LABEL_NAMES:
        records.extend(
            [
                {
                    "label": label,
                    "series": "Actual",
                    "prevalence": metrics["actual_label_prevalence"][label],
                },
                {
                    "label": label,
                    "series": "Predicted",
                    "prevalence": metrics["predicted_label_prevalence"][label],
                },
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=pd.DataFrame(records), x="label", y="prevalence", hue="series")
    plt.ylim(0, 1)
    plt.ylabel("Validation prevalence")
    plt.xlabel("Dataset-assigned label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write readable, deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
