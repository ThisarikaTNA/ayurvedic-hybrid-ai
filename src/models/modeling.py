"""Leakage-safe Phase 4 multilabel models and evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from models.splitting import TARGET_COLUMNS, validate_model_features


RANDOM_STATE = 42
THRESHOLD = 0.5
LABEL_NAMES: tuple[str, ...] = ("Vata", "Pitta", "Kapha")
TEXT_FEATURE = "symptoms"
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "symptom_severity",
    "age_group",
    "gender",
    "sleep_patterns",
    "stress_levels",
    "physical_activity_levels",
    "seasonal_variation",
)
PHASE4_FEATURES: tuple[str, ...] = (TEXT_FEATURE, *CATEGORICAL_FEATURES)


def flatten_text_column(values: Any) -> np.ndarray:
    """Flatten a one-column table into clean strings for TF-IDF.

    The function is top-level so fitted pipelines remain serializable by
    joblib and portable to later coursework phases.
    """

    flattened = np.asarray(values, dtype=object).reshape(-1)
    return pd.Series(flattened).fillna("").astype(str).to_numpy()


def build_feature_preprocessor() -> ColumnTransformer:
    """Build train-fitted TF-IDF and one-hot preprocessing."""

    validate_model_features(PHASE4_FEATURES)
    text_pipeline = Pipeline(
        steps=[
            ("flatten", FunctionTransformer(flatten_text_column, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("symptoms_tfidf", text_pipeline, [TEXT_FEATURE]),
            (
                "categorical_onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )


def build_model_pipelines(
    *, random_state: int = RANDOM_STATE, random_forest_estimators: int = 300
) -> dict[str, Pipeline]:
    """Return the three fixed pipelines used in the Phase 4 comparison."""

    return {
        "dummy_prior": Pipeline(
            steps=[
                ("preprocessor", build_feature_preprocessor()),
                (
                    "classifier",
                    OneVsRestClassifier(
                        DummyClassifier(strategy="prior", random_state=random_state)
                    ),
                ),
            ]
        ),
        "logistic_regression_ovr": Pipeline(
            steps=[
                ("preprocessor", build_feature_preprocessor()),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=random_state,
                        )
                    ),
                ),
            ]
        ),
        "random_forest_multioutput": Pipeline(
            steps=[
                ("preprocessor", build_feature_preprocessor()),
                (
                    "classifier",
                    MultiOutputClassifier(
                        RandomForestClassifier(
                            n_estimators=random_forest_estimators,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            # Single-process execution is reproducible and avoids
                            # restricted Windows worker-pipe permissions.
                            n_jobs=1,
                        )
                    ),
                ),
            ]
        ),
    }


def positive_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return one positive-class probability column per Dosha label."""

    probabilities = pipeline.predict_proba(features)
    if isinstance(probabilities, list):
        classifier = pipeline.named_steps["classifier"]
        columns: list[np.ndarray] = []
        for estimator, values in zip(classifier.estimators_, probabilities, strict=True):
            classes = np.asarray(estimator.classes_)
            positive = np.flatnonzero(classes == 1)
            if len(positive) == 0:
                columns.append(np.zeros(len(features), dtype=float))
            else:
                columns.append(np.asarray(values)[:, int(positive[0])])
        result = np.column_stack(columns)
    else:
        result = np.asarray(probabilities, dtype=float)
    if result.shape != (len(features), len(TARGET_COLUMNS)):
        raise ValueError(
            f"Expected probability shape {(len(features), len(TARGET_COLUMNS))}, "
            f"observed {result.shape}"
        )
    return result


def evaluate_predictions(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
) -> dict[str, Any]:
    """Calculate the requested multilabel metrics and confusion matrices."""

    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if truth.shape != predicted.shape or truth.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(f"Unexpected truth/prediction shapes: {truth.shape}, {predicted.shape}")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, average=None, zero_division=0
    )
    confusion = multilabel_confusion_matrix(truth, predicted)
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(truth, predicted)),
        "exact_match_subset_accuracy": float(accuracy_score(truth, predicted)),
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
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


def prevalence_report(
    y_train: pd.DataFrame, y_validation: pd.DataFrame
) -> dict[str, Any]:
    """Report label prevalence and the implied majority-label baseline."""

    train_prevalence = y_train.astype(int).mean()
    validation_prevalence = y_validation.astype(int).mean()
    return {
        "train": {
            label: float(train_prevalence[column])
            for label, column in zip(LABEL_NAMES, TARGET_COLUMNS, strict=True)
        },
        "validation": {
            label: float(validation_prevalence[column])
            for label, column in zip(LABEL_NAMES, TARGET_COLUMNS, strict=True)
        },
        "majority_prediction_from_training": {
            label: int(train_prevalence[column] >= THRESHOLD)
            for label, column in zip(LABEL_NAMES, TARGET_COLUMNS, strict=True)
        },
    }


def save_metric_comparison_figure(
    metrics_by_model: Mapping[str, Mapping[str, Any]], output_path: Path
) -> None:
    """Save a grouped bar chart of aggregate validation metrics."""

    records = []
    for model, metrics in metrics_by_model.items():
        for metric in ("macro_f1", "micro_f1", "exact_match_subset_accuracy"):
            records.append({"model": model, "metric": metric, "value": metrics[metric]})
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    sns.barplot(data=frame, x="metric", y="value", hue="model")
    plt.ylim(0, 1)
    plt.title("Phase 4 validation metrics (group-disjoint validation set)")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=12, ha="right")
    plt.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_per_label_f1_figure(
    metrics_by_model: Mapping[str, Mapping[str, Any]], output_path: Path
) -> None:
    """Save per-Dosha validation F1 comparisons."""

    records = [
        {"model": model, "label": label, "f1": metrics["per_label"][label]["f1"]}
        for model, metrics in metrics_by_model.items()
        for label in LABEL_NAMES
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(records), x="label", y="f1", hue="model")
    plt.ylim(0, 1)
    plt.title("Per-label F1 on validation knowledge profiles")
    plt.xlabel("Dataset-assigned Dosha label")
    plt.ylabel("F1")
    plt.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_confusion_matrices_figure(
    metrics_by_model: Mapping[str, Mapping[str, Any]], output_path: Path
) -> None:
    """Save one binary confusion matrix per model and Dosha label."""

    models = list(metrics_by_model)
    figure, axes = plt.subplots(len(models), len(LABEL_NAMES), figsize=(12, 10))
    for row, model in enumerate(models):
        for column, label in enumerate(LABEL_NAMES):
            values = metrics_by_model[model]["multilabel_confusion_matrices"][label]
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
                ax=axes[row, column],
            )
            axes[row, column].set_title(f"{model}\n{label}")
    figure.suptitle("Phase 4 multilabel confusion matrices — validation only", y=1.01)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic, readable JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
