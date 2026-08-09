"""Freeze and validate the accepted Phase 4 model-selection decision."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn


class SelectionManifestError(ValueError):
    """Raised when a frozen model-selection decision would change."""


def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_selection_manifest(
    split_manifest: Path,
    preprocessing_output: Path,
    model_bundle: Path,
) -> dict[str, Any]:
    """Build the exact accepted, hash-bound model-selection record."""

    return {
        "status": "frozen_after_user_acceptance",
        "phase": 4,
        "selected_model": "symptoms-only one-vs-rest Logistic Regression",
        "implementation_candidate": "symptoms_only__logistic_regression_ovr",
        "hyperparameters": {"C": 0.5, "class_weight": "balanced"},
        "thresholds": {"Vata": 0.45, "Pitta": 0.45, "Kapha": 0.45},
        "primary_selection_metric": "validation Macro-F1",
        "features": ["symptoms"],
        "random_seed": 42,
        "software_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts": {
            "split_manifest": {
                "path": str(Path(split_manifest).resolve()),
                "sha256": file_sha256(split_manifest),
            },
            "preprocessing_output": {
                "path": str(Path(preprocessing_output).resolve()),
                "sha256": file_sha256(preprocessing_output),
            },
            "saved_model_bundle": {
                "path": str(Path(model_bundle).resolve()),
                "sha256": file_sha256(model_bundle),
            },
        },
        "rationale": (
            "Highest validation Macro-F1 among the compared candidates, with greater "
            "simplicity and interpretability."
        ),
        "limitation": (
            "The combined-feature Random Forest was a near-tie on validation Macro-F1 "
            "and performed better on several secondary metrics. Logistic Regression "
            "must not be described as decisively superior."
        ),
        "probability_statement": (
            "Outputs are uncalibrated probability estimates, not calibrated confidence."
        ),
        "final_test_status": "sealed; not accessed or evaluated for this decision",
    }


def write_or_validate_frozen_manifest(path: Path, payload: dict[str, Any]) -> bool:
    """Write once, or verify exact identity without replacing the frozen decision."""

    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SelectionManifestError(
                "The existing frozen model-selection manifest differs from the "
                "accepted decision or artifact hashes."
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True
