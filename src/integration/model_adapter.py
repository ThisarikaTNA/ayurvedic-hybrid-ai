"""Verified, prediction-only adapter for the frozen Phase 4 model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd

from knowledge_base.database import file_sha256


LABELS = ("Vata", "Pitta", "Kapha")


class ModelIntegrityError(RuntimeError):
    """Raised when the frozen model artifact or metadata differs from approval."""


class FrozenModelAdapter:
    """Load and invoke the approved model without fitting or threshold changes."""

    def __init__(
        self,
        project_root: Path,
        config: Mapping[str, Any],
        *,
        loader: Callable[[Path], Any] = joblib.load,
    ) -> None:
        self.project_root = project_root
        self.config = dict(config)
        self.loader = loader
        self.invocation_count = 0

    def verify(self) -> dict[str, Any]:
        path = self.project_root / self.config["bundle_path"]
        actual = file_sha256(path)
        expected = self.config["bundle_sha256"]
        if actual != expected:
            raise ModelIntegrityError(
                f"Frozen model hash mismatch: expected {expected}, observed {actual}."
            )
        return {"verified": True, "path": str(path.resolve()), "sha256": actual}

    def predict(self, symptom_text: str) -> dict[str, Any]:
        verification = self.verify()
        try:
            bundle = self.loader(self.project_root / self.config["bundle_path"])
        except Exception as error:
            raise ModelIntegrityError(f"Frozen model bundle could not be loaded: {error}") from error
        required = {
            "pipeline", "thresholds", "feature_columns", "candidate",
            "score_semantics", "final_test_evaluated",
        }
        if not isinstance(bundle, dict) or set(bundle) != required:
            raise ModelIntegrityError("Frozen model bundle has an unexpected structure.")
        if bundle["candidate"] != self.config["candidate"]:
            raise ModelIntegrityError("Frozen model candidate identifier differs from approval.")
        if bundle["feature_columns"] != self.config["feature_columns"]:
            raise ModelIntegrityError("Frozen model feature allowlist differs from approval.")
        if bundle["final_test_evaluated"] is not False:
            raise ModelIntegrityError("Frozen bundle does not preserve the final-test seal flag.")
        expected_thresholds = np.asarray(
            [self.config["thresholds"][label] for label in LABELS], dtype=float
        )
        observed_thresholds = np.asarray(bundle["thresholds"], dtype=float)
        if observed_thresholds.shape != (3,) or not np.array_equal(
            observed_thresholds, expected_thresholds
        ):
            raise ModelIntegrityError("Frozen thresholds differ from approval.")

        frame = pd.DataFrame({"symptoms": [symptom_text]})
        try:
            scores = np.asarray(bundle["pipeline"].predict_proba(frame), dtype=float)
        except Exception as error:
            raise ModelIntegrityError(f"Frozen model prediction failed: {error}") from error
        if scores.shape != (1, 3) or not np.isfinite(scores).all():
            raise ModelIntegrityError(f"Unexpected prediction shape or values: {scores.shape}.")
        self.invocation_count += 1
        decisions = scores[0] >= expected_thresholds
        labels = [label for label, selected in zip(LABELS, decisions, strict=True) if selected]
        label_outputs = {
            label: {
                "raw_model_probability": round(float(score), 10),
                "frozen_threshold": float(threshold),
                "threshold_decision": bool(selected),
            }
            for label, score, threshold, selected in zip(
                LABELS, scores[0], expected_thresholds, decisions, strict=True
            )
        }
        return {
            "status": "abstained" if not labels else "success",
            "provenance": "model_generated",
            "model_predicted_dosha_labels": labels,
            "abstention": not labels,
            "abstention_reason": (
                "No label met its frozen threshold; no highest-scoring label was forced."
                if not labels else None
            ),
            "label_outputs": label_outputs,
            "score_semantics": "uncalibrated probability estimates",
            "not_confidence_statement": (
                "These are model outputs for dataset-assigned tags, not confidence in a "
                "true or clinically validated Dosha."
            ),
            "model_version": self.config["model_version"],
            "preprocessing_version": self.config["preprocessing_version"],
            "candidate": self.config["candidate"],
            "feature_columns": list(self.config["feature_columns"]),
            "input_validation_outcome": "valid_non_empty_symptom_text",
            "artifact_verification": verification,
        }
