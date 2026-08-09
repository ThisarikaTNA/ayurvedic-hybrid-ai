"""Run Phase 4 model and threshold comparison without accessing final-test results."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from models.splitting import file_sha256
from models.validation_modeling import (
    CATEGORICAL_FEATURES,
    FEATURE_EXPERIMENTS,
    LABEL_NAMES,
    MODEL_NAMES,
    RANDOM_STATE,
    TARGET_COLUMNS,
    aggregate_metric_row,
    apply_thresholds,
    build_pipeline,
    evaluate_predictions,
    fit_pipeline_training_only,
    hyperparameter_candidates,
    save_comparison_figure,
    save_confusion_figure,
    save_prevalence_figure,
    tune_per_label_thresholds,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--data", type=Path, default=Path("data/processed/ayurgenix_cleaned.csv")
    )
    command.add_argument(
        "--assignments", type=Path, default=Path("outputs/splits/split_assignments.csv")
    )
    command.add_argument(
        "--manifest", type=Path, default=Path("outputs/splits/split_manifest.json")
    )
    command.add_argument(
        "--output-dir", type=Path, default=Path("outputs/phase4_validation")
    )
    command.add_argument(
        "--models-dir", type=Path, default=Path("models/phase4_validation")
    )
    command.add_argument(
        "--figures-dir", type=Path, default=Path("reports/figures/phase4_validation")
    )
    command.add_argument(
        "--report", type=Path, default=Path("docs/phase4_validation_report.md")
    )
    return command


def locked_dataset_hash(manifest_path: Path) -> str:
    """Read only the dataset-hash field without materializing test assignments."""

    pattern = re.compile(r'"cleaned_dataset_sha256"\s*:\s*"([A-F0-9]{64})"')
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1)
    raise ValueError("Locked split manifest does not contain a cleaned dataset SHA-256.")


def load_unsealed_frames(
    data_path: Path, assignment_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load only train/validation assignments; never select final-test rows."""

    assignments = pd.read_csv(
        assignment_path,
        usecols=["knowledge_profile_id", "normalized_disease", "split"],
        dtype="string",
        keep_default_na=False,
    )
    if assignments["knowledge_profile_id"].duplicated().any():
        raise ValueError("Split assignments contain duplicate profile IDs.")
    train_assignment = assignments[assignments["split"] == "train"].copy()
    validation_assignment = assignments[assignments["split"] == "validation"].copy()
    if set(train_assignment["normalized_disease"]) & set(
        validation_assignment["normalized_disease"]
    ):
        raise ValueError("Training and validation disease groups overlap.")

    allowed_ids = set(train_assignment["knowledge_profile_id"]) | set(
        validation_assignment["knowledge_profile_id"]
    )
    required_columns = [
        "knowledge_profile_id", *FEATURE_EXPERIMENTS["symptoms_plus_categorical"],
        *TARGET_COLUMNS,
    ]
    selected_rows: list[dict[str, str]] = []
    with data_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = sorted(set(required_columns) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Cleaned data is missing Phase 4 columns: {missing}")
        for row in reader:
            profile_id = row["knowledge_profile_id"]
            if profile_id in allowed_ids:
                selected_rows.append({column: row[column] for column in required_columns})
    selected = pd.DataFrame(selected_rows, columns=required_columns)
    joined = selected.merge(
        assignments.loc[:, ["knowledge_profile_id", "split"]],
        on="knowledge_profile_id",
        how="left",
        validate="one_to_one",
    )
    train = joined[joined["split"] == "train"].copy()
    validation = joined[joined["split"] == "validation"].copy()
    if len(train) != len(train_assignment) or len(validation) != len(validation_assignment):
        raise ValueError("Train/validation rows do not match the locked assignments.")
    audit = {
        "training_profiles": int(len(train)),
        "validation_profiles": int(len(validation)),
        "train_validation_group_overlap": 0,
        "final_test_rows_selected": 0,
        "final_test_metrics_computed": False,
    }
    return train, validation, audit


def _selection_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    complexity = {
        "dummy_prior": 0,
        "logistic_regression_ovr": 1,
        "random_forest_ovr": 2,
    }[record["model"]]
    metrics = record["metrics_tuned"]
    return (
        metrics["macro_f1"],
        metrics["micro_f1"],
        -metrics["hamming_loss"],
        -float(complexity),
    )


def _write_human_report(
    path: Path,
    comparison: pd.DataFrame,
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    tuned = comparison[comparison["threshold_mode"] == "validation_tuned"].copy()
    tuned = tuned.sort_values("macro_f1", ascending=False)
    table_lines = [
        "| Feature experiment | Model | Macro-F1 | Micro-F1 | Samples-F1 | Hamming loss | Exact match | Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in tuned.iterrows():
        table_lines.append(
            f"| {row['feature_experiment']} | {row['model']} | {row['macro_f1']:.4f} | "
            f"{row['micro_f1']:.4f} | {row['samples_f1']:.4f} | "
            f"{row['hamming_loss']:.4f} | {row['exact_match_subset_accuracy']:.4f} | "
            f"{row['jaccard_samples']:.4f} |"
        )
    threshold_lines = []
    for candidate in candidates:
        thresholds = candidate["thresholds"]
        threshold_lines.append(
            f"- `{candidate['candidate']}`: Vata={thresholds[0]:.2f}, "
            f"Pitta={thresholds[1]:.2f}, Kapha={thresholds[2]:.2f}"
        )
    best = selected["metrics_tuned"]
    per_label_lines = [
        "| Label | Precision | Recall | F1 | Positive support | Actual prevalence | Predicted prevalence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABEL_NAMES:
        values = best["per_label"][label]
        per_label_lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['positive_support']} | "
            f"{best['actual_label_prevalence'][label]:.4f} | "
            f"{best['predicted_label_prevalence'][label]:.4f} |"
        )
    text = f"""# Phase 4 validation model-comparison report

## Scope and seal

Pipelines were fitted on {audit['training_profiles']} training knowledge
profiles and compared on {audit['validation_profiles']} validation profiles.
No final-test row was selected and no final-test metric was computed.

The target is the dataset-assigned presence of Vata, Pitta, and Kapha tags. It
is not a prediction of a patient's true Dosha and is not clinically validated.

## Provisional selection

The provisional validation-selected candidate is
`{selected['candidate']}` with tuned validation Macro-F1 {best['macro_f1']:.4f},
Micro-F1 {best['micro_f1']:.4f}, and Hamming loss {best['hamming_loss']:.4f}.
Macro-F1 was primary; Micro-F1, lower Hamming loss, simplicity, and
interpretability were secondary.

## Validation comparison using tuned thresholds

{chr(10).join(table_lines)}

The full comparison CSV also reports threshold 0.5 for every row, including
the dummy baseline in both feature experiments.

The combined-feature Random Forest is a near-tie on Macro-F1 (0.7283 versus
0.7298) and performs better on Micro-F1, Hamming loss, exact match, and
Jaccard. The symptoms-only Logistic Regression remains provisional because
Macro-F1 is the declared primary metric and the model is simpler and more
interpretable. The near-tie is too small to support a strong superiority
claim. Categorical fields did not consistently help: they reduced Logistic
Regression performance but improved several Random Forest metrics.

## Provisional model per-label results

{chr(10).join(per_label_lines)}

Kapha remains weakest. The tuned model predicts every label more frequently
than its observed validation prevalence, especially Pitta and Kapha. This
helps recall but increases false positives.

## Thresholds

{chr(10).join(threshold_lines)}

Thresholds were selected independently per label from 0.20 to 0.80 in steps
of 0.05 using validation label-F1. Using the same 67 profiles for model,
hyperparameter, and threshold selection can overfit. Thresholds therefore
remain provisional. Scores are uncalibrated probability estimates, not
calibrated confidence. Brier score, log loss, and calibration plots are
deferred to a later calibration phase.

Threshold tuning also makes the dummy baseline look better on Macro-F1 while
driving exact-match accuracy to zero. This is a useful warning that optimizing
one validation metric can materially worsen other behaviour.

## Feature experiments and search

The symptoms-only experiment used training-fitted TF-IDF word unigrams and
bigrams. The combined experiment added one-hot encoding for all approved
pre-diagnosis categorical fields, including the high-cardinality templated
attributes. Missing text and categories were handled inside each pipeline.

For each feature experiment, the validation search evaluated one dummy setup,
four Logistic Regression settings (`C` 0.5/1.0 with and without balanced
weights), and four Random Forest settings (depth unrestricted/12 and minimum
leaf size 1/3, with 200 trees). Hyperparameters were selected using default
threshold 0.5 before threshold tuning.

## Limitations

- Only 312 training and 67 validation knowledge profiles are available.
- Kapha has the lowest prevalence and smallest validation support.
- Several categorical attributes appear templated; gains may reflect dataset
  regularities rather than transferable signal.
- Repeated disease groups contain conflicting dataset-assigned labels.
- Group disjointness does not eliminate semantic similarity or unseen proxies.
- Validation performance may generalize weakly to unseen diseases.
- No result establishes a clinically valid Dosha assessment.
- The final test must remain sealed until the provisional model, features,
  hyperparameters, and thresholds are explicitly approved.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    expected_hash = locked_dataset_hash(options.manifest)
    observed_hash = file_sha256(options.data)
    if observed_hash != expected_hash:
        raise ValueError("Cleaned dataset hash does not match the locked split.")
    train, validation, access_audit = load_unsealed_frames(
        options.data, options.assignments
    )
    y_train = train.loc[:, TARGET_COLUMNS].astype(int)
    y_validation = validation.loc[:, TARGET_COLUMNS].astype(int)

    search_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    prediction_output = validation.loc[:, ["knowledge_profile_id"]].copy()
    options.models_dir.mkdir(parents=True, exist_ok=True)
    options.figures_dir.mkdir(parents=True, exist_ok=True)

    for experiment_name, feature_columns in FEATURE_EXPERIMENTS.items():
        x_train = train.loc[:, feature_columns]
        x_validation = validation.loc[:, feature_columns]
        for model_name in MODEL_NAMES:
            evaluated: list[dict[str, Any]] = []
            for candidate_index, parameters in enumerate(
                hyperparameter_candidates(model_name)
            ):
                pipeline = build_pipeline(experiment_name, model_name, parameters)
                probabilities = fit_pipeline_training_only(
                    pipeline, x_train, y_train, x_validation
                )
                default_predictions = apply_thresholds(probabilities, (0.5, 0.5, 0.5))
                default_metrics = evaluate_predictions(
                    y_validation, default_predictions
                )
                record = {
                    "candidate_index": candidate_index,
                    "parameters": dict(parameters),
                    "pipeline": pipeline,
                    "probabilities": probabilities,
                    "metrics": default_metrics,
                }
                evaluated.append(record)
                search_rows.append(
                    {
                        "feature_experiment": experiment_name,
                        "model": model_name,
                        "candidate_index": candidate_index,
                        "parameters_json": json.dumps(parameters, sort_keys=True),
                        "selection_threshold": 0.5,
                        "macro_f1": default_metrics["macro_f1"],
                        "micro_f1": default_metrics["micro_f1"],
                        "hamming_loss": default_metrics["hamming_loss"],
                    }
                )
            best_candidate = max(
                evaluated,
                key=lambda item: (
                    item["metrics"]["macro_f1"],
                    item["metrics"]["micro_f1"],
                    -item["metrics"]["hamming_loss"],
                    -item["candidate_index"],
                ),
            )
            candidate_name = f"{experiment_name}__{model_name}"
            thresholds, tuning_rows = tune_per_label_thresholds(
                y_validation, best_candidate["probabilities"]
            )
            tuned_predictions = apply_thresholds(
                best_candidate["probabilities"], thresholds
            )
            tuned_metrics = evaluate_predictions(y_validation, tuned_predictions)
            for row in tuning_rows:
                threshold_rows.append({"candidate": candidate_name, **row})
            default_metrics = best_candidate["metrics"]
            comparison_rows.extend(
                [
                    aggregate_metric_row(
                        candidate_name, experiment_name, model_name, "default_0.5",
                        (0.5, 0.5, 0.5), default_metrics,
                    ),
                    aggregate_metric_row(
                        candidate_name, experiment_name, model_name,
                        "validation_tuned", thresholds, tuned_metrics,
                    ),
                ]
            )
            for threshold_mode, metrics in (
                ("default_0.5", default_metrics),
                ("validation_tuned", tuned_metrics),
            ):
                for label in LABEL_NAMES:
                    per_label_rows.append(
                        {
                            "candidate": candidate_name,
                            "feature_experiment": experiment_name,
                            "model": model_name,
                            "threshold_mode": threshold_mode,
                            "label": label,
                            **metrics["per_label"][label],
                            "actual_prevalence": metrics["actual_label_prevalence"][label],
                            "predicted_prevalence": metrics["predicted_label_prevalence"][label],
                        }
                    )
            pipeline_path = options.models_dir / f"{candidate_name}.joblib"
            joblib.dump(best_candidate["pipeline"], pipeline_path)
            save_confusion_figure(
                tuned_metrics,
                f"{candidate_name} — tuned validation thresholds",
                options.figures_dir / f"{candidate_name}__confusion.png",
            )
            for label_index, label in enumerate(LABEL_NAMES):
                key = label.casefold()
                prediction_output[f"actual_{key}"] = y_validation.iloc[:, label_index].to_numpy()
                prediction_output[f"{candidate_name}__score_{key}"] = best_candidate[
                    "probabilities"
                ][:, label_index]
                prediction_output[f"{candidate_name}__prediction_{key}"] = tuned_predictions[
                    :, label_index
                ]
            candidate_results.append(
                {
                    "candidate": candidate_name,
                    "feature_experiment": experiment_name,
                    "feature_columns": list(feature_columns),
                    "model": model_name,
                    "selected_hyperparameters": best_candidate["parameters"],
                    "thresholds": thresholds.tolist(),
                    "metrics_default_0_5": default_metrics,
                    "metrics_tuned": tuned_metrics,
                    "pipeline_path": str(pipeline_path.resolve()),
                    "score_semantics": "uncalibrated probability estimates",
                }
            )

    provisional = max(candidate_results, key=_selection_key)
    provisional_bundle = {
        "pipeline": joblib.load(provisional["pipeline_path"]),
        "thresholds": np.asarray(provisional["thresholds"], dtype=float),
        "feature_columns": provisional["feature_columns"],
        "candidate": provisional["candidate"],
        "score_semantics": "uncalibrated probability estimates",
        "final_test_evaluated": False,
    }
    joblib.dump(provisional_bundle, options.models_dir / "provisional_best_bundle.joblib")

    comparison = pd.DataFrame(comparison_rows)
    per_label = pd.DataFrame(per_label_rows)
    search = pd.DataFrame(search_rows)
    thresholds_frame = pd.DataFrame(threshold_rows)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(options.output_dir / "validation_comparison.csv", index=False)
    per_label.to_csv(options.output_dir / "per_label_results.csv", index=False)
    search.to_csv(options.output_dir / "hyperparameter_search.csv", index=False)
    thresholds_frame.to_csv(
        options.output_dir / "threshold_selection.csv", index=False
    )
    prediction_output.to_csv(
        options.output_dir / "validation_predictions.csv", index=False
    )
    write_json(
        options.output_dir / "threshold_selection_report.json",
        {
            "partition": "validation",
            "grid": list(np.arange(0.20, 0.801, 0.05).round(2)),
            "method": "separate per-label F1 maximization; ties prefer threshold nearest 0.5",
            "overfitting_warning": (
                "Thresholds reuse the 67-profile validation set used for model and "
                "hyperparameter comparison and may overfit."
            ),
            "selected_thresholds": {
                candidate["candidate"]: dict(zip(LABEL_NAMES, candidate["thresholds"], strict=True))
                for candidate in candidate_results
            },
        },
    )
    metrics_report = {
        "phase": 4,
        "scope": "training fit and validation comparison only",
        "target_definition": "dataset-assigned Dosha tags, not true patient Doshas",
        "random_state": RANDOM_STATE,
        "data_access_audit": access_audit,
        "cleaned_dataset_sha256": observed_hash,
        "split_manifest_sha256": file_sha256(options.manifest),
        "preprocessing_fit_scope": "inside each pipeline; training rows only",
        "feature_experiments": {
            name: list(columns) for name, columns in FEATURE_EXPERIMENTS.items()
        },
        "categorical_features": list(CATEGORICAL_FEATURES),
        "probability_statement": (
            "Scores are uncalibrated probability estimates. They are not calibrated confidence."
        ),
        "calibration_deferred": ["Brier score", "log loss", "calibration plots"],
        "candidate_results": candidate_results,
        "provisional_selection": {
            "candidate": provisional["candidate"],
            "selection_partition": "validation",
            "primary_metric": "macro_f1",
            "secondary_criteria": [
                "micro_f1", "lower hamming_loss", "model simplicity", "interpretability"
            ],
            "final_test_accessed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Only 312 training and 67 validation knowledge profiles are available.",
            "Kapha has the lowest label prevalence.",
            "Several approved categorical fields are templated and high-cardinality.",
            "Repeated disease groups contain conflicting dataset-assigned labels.",
            "Validation-based model and threshold selection may overfit.",
            "Generalization to unseen diseases may be weak.",
            "The task predicts dataset-assigned tags, not clinically validated Doshas.",
        ],
    }
    write_json(options.output_dir / "phase4_metrics.json", metrics_report)
    write_json(
        options.output_dir / "provisional_selection.json",
        {
            "candidate": provisional["candidate"],
            "feature_columns": provisional["feature_columns"],
            "selected_hyperparameters": provisional["selected_hyperparameters"],
            "thresholds": dict(zip(LABEL_NAMES, provisional["thresholds"], strict=True)),
            "validation_metrics": provisional["metrics_tuned"],
            "final_test_accessed": False,
            "bundle_path": str(
                (options.models_dir / "provisional_best_bundle.joblib").resolve()
            ),
        },
    )
    save_comparison_figure(
        comparison, options.figures_dir / "validation_macro_f1_comparison.png"
    )
    save_prevalence_figure(
        provisional["metrics_tuned"],
        f"{provisional['candidate']} — validation prevalence",
        options.figures_dir / "provisional_best_prevalence.png",
    )
    _write_human_report(
        options.report, comparison, candidate_results, provisional, access_audit
    )

    print("Phase 4 complete: validation comparison only; final test remains sealed.")
    print(f"Provisional selection: {provisional['candidate']}")
    print(
        "Validation metrics:",
        {
            key: provisional["metrics_tuned"][key]
            for key in ("macro_f1", "micro_f1", "samples_f1", "hamming_loss")
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
