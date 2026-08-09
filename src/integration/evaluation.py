"""Synthetic Phase 9 orchestration-scenario evaluation utilities."""

from __future__ import annotations

import copy
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from integration.hybrid_pipeline import HybridPipeline, PROJECT_ROOT, _open_read_only
from integration.retrieval_adapter import RetrievalAdapterError
from knowledge_base.rule_engine import RuleEngine
from integration.safety_gate import SafetyGate


REQUIRED_SECTIONS = {
    "condition_context", "safety_gate_result", "rule_trace", "model_prediction",
    "reference_checked_information", "dataset_derived_profiles",
    "dataset_derived_recommendations", "retrieval_trace",
    "agreements_and_disagreements", "suppressed_items", "limitations",
    "prototype_disclaimer",
}
DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|milligrams?|g|grams?|mcg|µg|ml|"
    r"millilit(?:er|re)s?|tablets?|capsules?|drops?)\b",
    re.IGNORECASE,
)
PROHIBITED_ASSERTIONS = (
    "you are medically safe", "clinically safe", "you have been diagnosed",
    "this is clinically validated", "guarantees benefit", "will cure",
    "clinical confidence",
)


class AlwaysVerified:
    def verify(self) -> dict[str, Any]:
        return {"status": "verified", "mode": "controlled_synthetic_component_test"}


class IntegrityFailure:
    def verify(self) -> dict[str, Any]:
        raise RuntimeError("Synthetic component-integrity mismatch detected.")


class RaisingGate:
    def evaluate(self, condition: str, facts: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Synthetic rule-engine boundary failure.")


class RaisingRetriever:
    def retrieve(self, **kwargs: Any) -> dict[str, Any]:
        raise RetrievalAdapterError("Synthetic FTS5 boundary failure.")


class ControlledModel:
    """Test double for orchestration paths; it never changes the production bundle."""

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.invocation_count = 0

    def predict(self, symptom_text: str) -> dict[str, Any]:
        self.invocation_count += 1
        selected = set(self.labels)
        values = {"Vata": 0.8, "Pitta": 0.8, "Kapha": 0.8}
        return {
            "status": "abstained" if not selected else "success",
            "provenance": "model_generated",
            "model_predicted_dosha_labels": self.labels,
            "abstention": not selected,
            "abstention_reason": (
                "Controlled technical abstention; no highest-scoring label was forced."
                if not selected else None
            ),
            "label_outputs": {
                label: {
                    "raw_model_probability": 0.2 if label not in selected else values[label],
                    "frozen_threshold": 0.45,
                    "threshold_decision": label in selected,
                }
                for label in ("Vata", "Pitta", "Kapha")
            },
            "score_semantics": "controlled uncalibrated probability estimates",
            "not_confidence_statement": "Technical test double; outputs are not confidence claims.",
            "model_version": "controlled-test-double",
            "preprocessing_version": "not_invoked_in_controlled_scenario",
            "candidate": "controlled_orchestration_test_double",
            "feature_columns": ["symptoms"],
            "input_validation_outcome": "valid_non_empty_symptom_text",
            "artifact_verification": {"mode": "controlled_synthetic_scenario"},
        }


def memory_copy(project_root: Path = PROJECT_ROOT) -> sqlite3.Connection:
    source = _open_read_only(project_root / "data/knowledge_base/ayurvedic_knowledge.db")
    target = sqlite3.connect(":memory:")
    target.row_factory = sqlite3.Row
    target.execute("PRAGMA foreign_keys=ON")
    source.backup(target)
    source.close()
    return target


def add_ambiguous_alias(connection: sqlite3.Connection) -> None:
    condition_ids = [
        int(row[0])
        for row in connection.execute(
            "SELECT condition_id FROM conditions ORDER BY condition_id LIMIT 2"
        )
    ]
    for condition_id, profile in zip(
        condition_ids, ("synthetic-a", "synthetic-b"), strict=True
    ):
        connection.execute(
            """
            INSERT INTO condition_aliases(
                condition_id, alias_text, normalized_alias, alias_type,
                source_profile_id, provenance_status, record_version
            ) VALUES (?, 'Ambiguous Demonstration', 'ambiguous demonstration',
                      'common_name', ?, 'draft', 'synthetic-test')
            """,
            (condition_id, profile),
        )
    connection.commit()


def add_synthetic_rule_conflict(connection: sqlite3.Connection) -> None:
    source = connection.execute(
        "SELECT * FROM knowledge_rules WHERE rule_id='R-ACNE-SELFCARE-001'"
    ).fetchone()
    if source is None:
        raise RuntimeError("Source technical rule is unavailable.")
    record = dict(source)
    record["rule_id"] = "R-ACNE-SELFCARE-TEST"
    record["rule_key"] = "R-ACNE-SELFCARE-TEST"
    action = json.loads(record["action_json"])
    action["value"] = "synthetic_incompatible_test_value"
    record["action_json"] = json.dumps(action, sort_keys=True)
    columns = list(record)
    connection.execute(
        f"INSERT INTO knowledge_rules({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        [record[column] for column in columns],
    )
    evidence = dict(
        connection.execute(
            "SELECT * FROM rule_evidence WHERE rule_id='R-ACNE-SELFCARE-001'"
        ).fetchone()
    )
    evidence.pop("rule_evidence_id")
    evidence["rule_id"] = record["rule_id"]
    columns = list(evidence)
    connection.execute(
        f"INSERT INTO rule_evidence({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        [evidence[column] for column in columns],
    )
    validation = dict(
        connection.execute(
            "SELECT * FROM rule_validation WHERE rule_id='R-ACNE-SELFCARE-001'"
        ).fetchone()
    )
    validation.pop("rule_validation_id")
    validation["rule_id"] = record["rule_id"]
    validation["validator"] = "synthetic_phase9_conflict_test"
    columns = list(validation)
    connection.execute(
        f"INSERT INTO rule_validation({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        [validation[column] for column in columns],
    )
    connection.commit()


def _controlled_pipeline(scenario: Mapping[str, Any]) -> tuple[HybridPipeline, sqlite3.Connection | None]:
    mode = scenario.get("mode", "production_components")
    connection: sqlite3.Connection | None = None
    kwargs: dict[str, Any] = {}
    if mode == "integrity_failure":
        kwargs["integrity_verifier"] = IntegrityFailure()
    elif mode == "rule_engine_failure":
        kwargs["safety_gate"] = RaisingGate()
    elif mode == "fts_failure":
        kwargs["retrieval_adapter"] = RaisingRetriever()
    elif mode == "ambiguous_temp_database":
        connection = memory_copy()
        add_ambiguous_alias(connection)
        kwargs["connection"] = connection
    elif mode == "rule_conflict_temp_database":
        connection = memory_copy()
        add_synthetic_rule_conflict(connection)
        kwargs["connection"] = connection
        kwargs["safety_gate"] = SafetyGate(
            RuleEngine(connection),
            {
                "professional_referral_rules": 5, "general_information_rules": 5,
                "contraindication_rules": 0, "exclusion_rules": 0,
                "expert_reviewed_rules": 0,
            },
        )
    if "controlled_model_labels" in scenario:
        kwargs["model_adapter"] = ControlledModel(scenario["controlled_model_labels"])
    return HybridPipeline(**kwargs), connection


def _findings(result: Mapping[str, Any]) -> list[str]:
    text = json.dumps(result, ensure_ascii=False).casefold()
    findings = [phrase for phrase in PROHIBITED_ASSERTIONS if phrase in text]
    findings.extend(match.group(0) for match in DOSAGE_PATTERN.finditer(text))
    return sorted(set(findings))


def _provenance_correct(result: Mapping[str, Any]) -> bool:
    model = result.get("model_prediction", {})
    return (
        (not model or model.get("provenance") == "model_generated")
        and all(
            item.get("provenance") == "reference_checked"
            for item in result.get("reference_checked_information", [])
        )
        and all(
            item.get("provenance") == "dataset_derived"
            for item in result.get("dataset_derived_profiles", [])
        )
        and all(
            item.get("provenance") == "dataset_derived"
            for item in result.get("dataset_derived_recommendations", [])
        )
        and all(
            item.get("provenance") == item.get("evidence_status")
            for item in result.get("rule_trace", [])
        )
    )


def evaluate_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    all_outputs: list[dict[str, Any]] = []
    for scenario in scenarios:
        pipeline, connection = _controlled_pipeline(scenario)
        try:
            result = pipeline.run(scenario["input"])
            repeat_result = pipeline.run(scenario["input"]) if scenario.get("repeat") else None
        finally:
            if connection is not None:
                connection.close()
        expected_state = scenario["expected_state"]
        checks = {
            "state_correct": result["orchestration_state"] == expected_state,
            "trace_complete": REQUIRED_SECTIONS.issubset(result),
            "provenance_labels_correct": _provenance_correct(result),
            "prohibited_output_findings": _findings(result),
            "deterministic_repeat": (
                result == repeat_result if repeat_result is not None else None
            ),
        }
        for assertion in scenario.get("assertions", []):
            if assertion == "no_model_invocation":
                checks[assertion] = "frozen_ml_model" not in result["invocation_trace"]
            elif assertion == "no_retrieval_invocation":
                checks[assertion] = "condition_scoped_retrieval" not in result["invocation_trace"]
            elif assertion == "referral_prominent":
                checks[assertion] = bool(
                    result["safety_gate_result"].get("prominent_referral_information")
                )
            elif assertion == "recommendations_suppressed":
                checks[assertion] = bool(result["suppressed_items"])
            elif assertion == "lexical_only":
                checks[assertion] = (
                    result["retrieval_trace"].get("adapter_trace", {}).get("tag_source") == "none"
                )
            elif assertion == "insomnia_disagreement":
                checks[assertion] = any(
                    item.get("type") == "retained_dataset_disagreement"
                    for item in result["agreements_and_disagreements"]
                )
            elif assertion == "dosha_disagreement":
                checks[assertion] = any(
                    item.get("relationship") in {"disagreement", "partial_disagreement"}
                    for item in result["agreements_and_disagreements"]
                )
            elif assertion == "dosha_agreement":
                checks[assertion] = any(
                    item.get("relationship") == "set_agreement"
                    for item in result["agreements_and_disagreements"]
                )
            elif assertion == "no_matching_dosha_profile":
                checks[assertion] = any(
                    item.get("type") == "no_matching_dosha_profile"
                    for item in result["agreements_and_disagreements"]
                )
            elif assertion == "empty_profiles":
                checks[assertion] = not result["dataset_derived_profiles"]
            elif assertion == "integrity_detected":
                checks[assertion] = result["component_integrity"].get("status") == "failed"
        boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
        passed = all(boolean_checks) and not checks["prohibited_output_findings"]
        record = {
            "scenario_id": scenario["scenario_id"],
            "description": scenario["description"],
            "mode": scenario.get("mode", "production_components"),
            "expected_state": expected_state,
            "actual_state": result["orchestration_state"],
            "checks": checks,
            "passed": passed,
        }
        records.append(record)
        traces.append({"scenario_id": scenario["scenario_id"], "result": result})
        all_outputs.append(result)
        if repeat_result is not None:
            all_outputs.append(repeat_result)

    states = Counter(record["actual_state"] for record in records)
    invocation_counts = Counter(
        step for output in all_outputs for step in output["invocation_trace"]
    )
    disagreement_count = sum(
        1
        for output in all_outputs
        for item in output["agreements_and_disagreements"]
        if item.get("relationship") in {"disagreement", "partial_disagreement"}
    )
    agreement_count = sum(
        1
        for output in all_outputs
        for item in output["agreements_and_disagreements"]
        if item.get("relationship") == "set_agreement"
    )
    passed = sum(record["passed"] for record in records)
    metrics = {
        "scenario_count": len(records),
        "passed_scenarios": passed,
        "scenario_pass_rate": passed / len(records),
        "safety_state_correctness": sum(
            record["checks"]["state_correct"] for record in records
        ) / len(records),
        "trace_completeness": sum(
            record["checks"]["trace_complete"] for record in records
        ) / len(records),
        "provenance_label_correctness": sum(
            record["checks"]["provenance_labels_correct"] for record in records
        ) / len(records),
        "referral_suppression_correctness": next(
            record["checks"].get("recommendations_suppressed", False)
            for record in records if record["scenario_id"] == "P9-REFERRAL-ACTIVATION"
        ),
        "missing_input_clarification_correctness": next(
            record["checks"]["state_correct"]
            for record in records if record["scenario_id"] == "P9-MISSING-SAFETY"
        ),
        "component_integrity_detection": next(
            record["checks"].get("integrity_detected", False)
            for record in records
            if record["scenario_id"] == "P9-COMPONENT-INTEGRITY-FAILURE"
        ),
        "zero_prohibited_output_finding_rate": sum(
            not record["checks"]["prohibited_output_findings"] for record in records
        ) / len(records),
        "deterministic_repeatability_rate": (
            sum(record["checks"]["deterministic_repeat"] is True for record in records)
            / max(1, sum(record["checks"]["deterministic_repeat"] is not None for record in records))
        ),
    }
    return {
        "scope": "synthetic technical orchestration evaluation; not clinical accuracy",
        "scenario_results": records,
        "metrics": metrics,
        "safety_gate_outcomes": dict(sorted(states.items())),
        "component_invocation_counts": dict(sorted(invocation_counts.items())),
        "model_invocations": {
            "total_adapter_attempts": int(invocation_counts["frozen_ml_model"]),
            "verified_frozen_bundle": sum(
                (2 if scenario.get("repeat") else 1)
                for scenario in scenarios
                if "controlled_model_labels" not in scenario
                and scenario.get("mode", "production_components") not in {
                    "integrity_failure", "rule_engine_failure",
                    "rule_conflict_temp_database", "ambiguous_temp_database",
                }
                and scenario["expected_state"] not in {
                    "blocked_invalid_input", "clarification_required",
                    "escalation_information", "blocked_rule_conflict",
                }
            ),
            "controlled_orchestration_test_double": sum(
                1 for scenario in scenarios if "controlled_model_labels" in scenario
            ),
        },
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "sample_traces": traces,
    }
