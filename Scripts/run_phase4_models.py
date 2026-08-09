"""Train fixed Phase 4 models and evaluate on validation only."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn
import sklearn

from models.modeling import (
    CATEGORICAL_FEATURES,
    LABEL_NAMES,
    PHASE4_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMNS,
    TEXT_FEATURE,
    THRESHOLD,
    build_model_pipelines,
    evaluate_predictions,
    positive_probabilities,
    prevalence_report,
    save_confusion_matrices_figure,
    save_metric_comparison_figure,
    save_per_label_f1_figure,
    write_json,
)
from models.splitting import file_sha256, verify_assignments


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
    command.add_argument("--output-dir", type=Path, default=Path("outputs/modeling"))
    command.add_argument("--models-dir", type=Path, default=Path("models"))
    command.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    return command


def main(arguments: list[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    dataframe = pd.read_csv(
        options.data, dtype="string", keep_default_na=False, encoding="utf-8"
    )
    assignments = pd.read_csv(
        options.assignments, dtype="string", keep_default_na=False, encoding="utf-8"
    )
    manifest = json.loads(options.manifest.read_text(encoding="utf-8"))
    observed_hash = file_sha256(options.data)
    if observed_hash != manifest["cleaned_dataset_sha256"]:
        raise ValueError("Cleaned dataset SHA-256 does not match the locked split manifest.")
    verify_assignments(dataframe, assignments)

    joined = dataframe.merge(
        assignments.loc[:, ["knowledge_profile_id", "split"]],
        on="knowledge_profile_id",
        how="inner",
        validate="one_to_one",
    )
    train = joined[joined["split"] == "train"].copy()
    validation = joined[joined["split"] == "validation"].copy()
    # The final-test rows are deliberately not selected or evaluated here.
    x_train = train.loc[:, PHASE4_FEATURES]
    y_train = train.loc[:, TARGET_COLUMNS].astype(int)
    x_validation = validation.loc[:, PHASE4_FEATURES]
    y_validation = validation.loc[:, TARGET_COLUMNS].astype(int)

    pipelines = build_model_pipelines(random_state=RANDOM_STATE)
    metrics_by_model: dict[str, dict] = {}
    prediction_output = validation.loc[:, ["knowledge_profile_id"]].copy()
    options.models_dir.mkdir(parents=True, exist_ok=True)
    for model_name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        probabilities = positive_probabilities(pipeline, x_validation)
        predictions = (probabilities >= THRESHOLD).astype(int)
        metrics_by_model[model_name] = evaluate_predictions(y_validation, predictions)
        metrics_by_model[model_name]["threshold"] = THRESHOLD
        metrics_by_model[model_name]["training_profile_count"] = len(train)
        metrics_by_model[model_name]["validation_profile_count"] = len(validation)
        joblib.dump(pipeline, options.models_dir / f"{model_name}.joblib")
        for index, label in enumerate(LABEL_NAMES):
            key = label.casefold()
            prediction_output[f"actual_{key}"] = y_validation.iloc[:, index].to_numpy()
            prediction_output[f"{model_name}_probability_{key}"] = probabilities[:, index]
            prediction_output[f"{model_name}_prediction_{key}"] = predictions[:, index]

    ranked = sorted(
        metrics_by_model,
        key=lambda name: (
            metrics_by_model[name]["macro_f1"], metrics_by_model[name]["micro_f1"]
        ),
        reverse=True,
    )
    selected_model_name = ranked[0]
    joblib.dump(
        pipelines[selected_model_name],
        options.models_dir / "selected_phase4_pipeline.joblib",
    )
    report = {
        "phase": 4,
        "scope": "training partition fit; group-disjoint validation evaluation only",
        "target_definition": "dataset-assigned Dosha tags, not a patient's true Dosha",
        "data": {
            "cleaned_dataset_sha256": observed_hash,
            "split_manifest": str(options.manifest.resolve()),
            "train_profiles": len(train),
            "validation_profiles": len(validation),
            "final_test_accessed_for_evaluation": False,
        },
        "configuration": {
            "random_state": RANDOM_STATE,
            "classification_threshold": THRESHOLD,
            "hyperparameter_tuning": False,
            "random_forest_execution": "single process (n_jobs=1)",
            "text_feature": TEXT_FEATURE,
            "text_processing": "TF-IDF word unigrams and bigrams; min_df=2; sublinear_tf=True",
            "categorical_features": list(CATEGORICAL_FEATURES),
            "categorical_processing": "one-hot encoding with unknown-category tolerance",
            "excluded_from_initial_phase4_features": [
                "risk_factors", "environmental_factors", "dietary_habits",
                "occupation_and_lifestyle",
            ],
            "preprocessing_fit_scope": "inside each pipeline, training rows only",
        },
        "label_prevalence_baseline": prevalence_report(y_train, y_validation),
        "validation_ranking_by_macro_then_micro_f1": ranked,
        "selected_validation_model": selected_model_name,
        "models": metrics_by_model,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Validation contains only 67 knowledge profiles.",
            "No hyperparameter or probability-threshold tuning was performed.",
            "Internal validation performance is not evidence of clinical validity.",
            "The final test remains sealed for later full-system evaluation.",
        ],
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(options.output_dir / "phase4_metrics.json", report)
    write_json(
        options.output_dir / "selected_model.json",
        {
            "model": selected_model_name,
            "selection_partition": "validation",
            "selection_metric": "macro_f1, with micro_f1 as tie-break",
            "final_test_accessed_for_selection": False,
            "pipeline_path": str(
                (options.models_dir / "selected_phase4_pipeline.joblib").resolve()
            ),
        },
    )
    prediction_output.to_csv(
        options.output_dir / "validation_predictions.csv", index=False, encoding="utf-8"
    )

    comparison_rows = [
        {
            "model": name,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "hamming_loss": metrics["hamming_loss"],
            "exact_match_subset_accuracy": metrics["exact_match_subset_accuracy"],
        }
        for name, metrics in metrics_by_model.items()
    ]
    pd.DataFrame(comparison_rows).to_csv(
        options.output_dir / "model_comparison.csv", index=False, encoding="utf-8"
    )
    per_label_rows = [
        {"model": model, "label": label, **metrics["per_label"][label]}
        for model, metrics in metrics_by_model.items()
        for label in LABEL_NAMES
    ]
    pd.DataFrame(per_label_rows).to_csv(
        options.output_dir / "per_label_metrics.csv", index=False, encoding="utf-8"
    )

    save_metric_comparison_figure(
        metrics_by_model, options.figures_dir / "phase4_metric_comparison.png"
    )
    save_per_label_f1_figure(
        metrics_by_model, options.figures_dir / "phase4_per_label_f1.png"
    )
    save_confusion_matrices_figure(
        metrics_by_model, options.figures_dir / "phase4_confusion_matrices.png"
    )

    print("Phase 4 validation results (final test remains sealed):")
    for row in comparison_rows:
        print(row)
    print(f"Validation ranking: {ranked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
