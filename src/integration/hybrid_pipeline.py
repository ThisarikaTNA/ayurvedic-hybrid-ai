"""Deterministic Phase 9 orchestration of frozen model, rules, and retrieval."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from integration.input_schema import (
    HybridInputError,
    validate_base_input,
    validate_condition_facts,
)
from integration.model_adapter import FrozenModelAdapter, ModelIntegrityError
from integration.result_composer import compose, empty_result
from integration.retrieval_adapter import RetrievalAdapter, RetrievalAdapterError
from integration.safety_gate import SafetyGate
from knowledge_base.database import file_sha256, foreign_key_violations
from knowledge_base.rule_engine import RuleEngine
from retrieval.profile_retriever import ProfileRetriever
from retrieval.query_schema import load_retrieval_config
from retrieval.repository import RetrievalRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_pipeline.v1.json"


class ComponentIntegrityError(RuntimeError):
    """Raised when an approved component differs from its frozen manifest."""


def load_hybrid_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise ComponentIntegrityError("Unsupported hybrid configuration version.")
    expected_order = [
        "validate_input", "resolve_condition", "safety_gate",
        "frozen_ml_model", "condition_scoped_retrieval", "compose_result",
    ]
    if config.get("component_order") != expected_order:
        raise ComponentIntegrityError("Hybrid component order differs from approval.")
    return config


def _stable_id(payload: Mapping[str, Any], config_version: str) -> str:
    encoded = json.dumps(
        {"config_version": config_version, "request": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"HYB-{hashlib.sha256(encoded).hexdigest()[:16].upper()}"


class ComponentIntegrityVerifier:
    """Verify all approved files and database invariants before invocation."""

    def __init__(self, project_root: Path, config: Mapping[str, Any]) -> None:
        self.project_root = project_root
        self.config = config

    def _file(self, relative: str, expected: str) -> dict[str, Any]:
        path = self.project_root / relative
        if not path.is_file():
            raise ComponentIntegrityError(f"Required component file is missing: {relative}")
        actual = file_sha256(path)
        if actual != expected:
            raise ComponentIntegrityError(
                f"Component hash mismatch for {relative}: expected {expected}, observed {actual}."
            )
        return {"path": relative, "expected_sha256": expected, "actual_sha256": actual, "verified": True}

    def verify(self) -> dict[str, Any]:
        model = self.config["frozen_model"]
        knowledge = self.config["knowledge_base"]
        retrieval = self.config["retrieval"]
        checks = {
            "model_manifest": self._file(model["manifest_path"], model["manifest_sha256"]),
            "model_bundle": self._file(model["bundle_path"], model["bundle_sha256"]),
            "phase7_manifest": self._file(
                knowledge["phase7_manifest_path"], knowledge["phase7_manifest_sha256"]
            ),
            "knowledge_database": self._file(
                knowledge["database_path"], knowledge["database_sha256"]
            ),
            "rule_catalog": self._file(
                knowledge["rule_catalog_path"], knowledge["rule_catalog_sha256"]
            ),
            "phase8_manifest": self._file(
                retrieval["phase8_manifest_path"], retrieval["phase8_manifest_sha256"]
            ),
            "retrieval_config": self._file(
                retrieval["config_path"], retrieval["config_sha256"]
            ),
            "retrieval_code": [
                self._file(path, expected)
                for path, expected in sorted(retrieval["code_hashes"].items())
            ],
        }
        phase4 = json.loads(
            (self.project_root / model["manifest_path"]).read_text(encoding="utf-8")
        )
        if (
            phase4.get("implementation_candidate") != model["candidate"]
            or phase4.get("features") != model["feature_columns"]
            or phase4.get("hyperparameters")
            != {"C": model["C"], "class_weight": model["class_weight"]}
            or phase4.get("thresholds") != model["thresholds"]
            or phase4.get("final_test_status")
            != "sealed; not accessed or evaluated for this decision"
        ):
            raise ComponentIntegrityError("Phase 4 manifest metadata differs from the approved decision.")
        phase7 = json.loads(
            (self.project_root / knowledge["phase7_manifest_path"]).read_text(encoding="utf-8")
        )
        if phase7.get("rule_counts", {}).get("rule_type") != {
            "general_recommendation": 5,
            "professional_referral": 5,
        }:
            raise ComponentIntegrityError("Phase 7 rule inventory differs from approval.")
        if phase7.get("expert_reviewed_rules") != 0:
            raise ComponentIntegrityError("Unexpected expert-reviewed Phase 7 rule record.")
        phase8 = json.loads(
            (self.project_root / retrieval["phase8_manifest_path"]).read_text(encoding="utf-8")
        )
        if phase8.get("database", {}).get("sha256_after") != knowledge["database_sha256"]:
            raise ComponentIntegrityError("Phase 8 and Phase 7 database hashes are inconsistent.")
        if phase8.get("configuration_file", {}).get("sha256") != retrieval["config_sha256"]:
            raise ComponentIntegrityError("Phase 8 retrieval configuration hash is inconsistent.")

        connection = _open_read_only(self.project_root / knowledge["database_path"])
        try:
            violations = foreign_key_violations(connection)
            current_version = connection.execute(
                "SELECT schema_version FROM knowledge_base_versions WHERE is_current=1"
            ).fetchone()
            rule_counts = dict(
                connection.execute(
                    "SELECT rule_type, COUNT(*) FROM knowledge_rules "
                    "WHERE lifecycle_status='active' AND is_stale=0 "
                    "GROUP BY rule_type"
                ).fetchall()
            )
            empty_counts = {
                table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                for table in self.config["persistence"]["production_tables_required_empty"]
            }
        finally:
            connection.close()
        if violations:
            raise ComponentIntegrityError("Knowledge database has foreign-key violations.")
        if current_version is None or current_version[0] != knowledge["schema_version"]:
            raise ComponentIntegrityError("Knowledge-base schema version differs from approval.")
        if rule_counts != {"general_recommendation": 5, "professional_referral": 5}:
            raise ComponentIntegrityError("Active non-stale production rule counts differ from approval.")
        if any(empty_counts.values()):
            raise ComponentIntegrityError(
                f"Production persistence tables are not empty: {empty_counts}."
            )
        checks["database_invariants"] = {
            "foreign_key_violation_count": 0,
            "schema_version": current_version[0],
            "active_non_stale_rule_counts": rule_counts,
            "production_table_counts": empty_counts,
            "verified": True,
        }
        return {"status": "verified", "checks": checks}


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA query_only = ON")
    return connection


class HybridPipeline:
    """Orchestrate approved components in one fixed, auditable order."""

    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        config: Mapping[str, Any] | None = None,
        integrity_verifier: Any | None = None,
        connection: sqlite3.Connection | None = None,
        safety_gate: Any | None = None,
        model_adapter: Any | None = None,
        retrieval_adapter: Any | None = None,
    ) -> None:
        self.project_root = project_root
        self.config = dict(config or load_hybrid_config())
        self.integrity_verifier = integrity_verifier or ComponentIntegrityVerifier(
            project_root, self.config
        )
        self._provided_connection = connection
        self._provided_gate = safety_gate
        self._provided_model = model_adapter
        self._provided_retrieval = retrieval_adapter

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        invocation: list[str] = ["validate_input"]
        try:
            validated = validate_base_input(request, self.config)
        except HybridInputError as error:
            run_id = _stable_id(
                request if isinstance(request, Mapping) else {"invalid_type": type(request).__name__},
                self.config["config_version"],
            )
            result = empty_result(run_id=run_id, pipeline_version=self.config["config_version"])
            result["component_versions"] = self._versions()
            result["component_integrity"] = {
                "status": "not_run", "reason": "input_validation_failed"
            }
            result["orchestration_state"] = "blocked_invalid_input"
            result["safety_gate_result"] = {
                "state": "blocked_invalid_input", "reason": str(error),
                "personalization_permitted": False,
            }
            result["invocation_trace"] = invocation + ["compose_result"]
            result["limitations"] = ["Input validation stopped all component invocation."]
            return result

        run_id = validated.request_id or _stable_id(
            validated.as_dict(), self.config["config_version"]
        )
        result = empty_result(run_id=run_id, pipeline_version=self.config["config_version"])
        result["component_versions"] = self._versions()
        result["invocation_trace"] = invocation
        try:
            integrity = self.integrity_verifier.verify()
        except Exception as error:
            result["orchestration_state"] = "blocked_component_failure"
            result["component_integrity"] = {
                "status": "failed", "component": "preflight", "reason": str(error)
            }
            result["safety_gate_result"] = {
                "state": "blocked_component_failure",
                "personalization_permitted": False,
                "reason": "Approved component integrity verification failed.",
            }
            result["invocation_trace"].append("compose_result")
            result["limitations"] = ["No model, rule or retrieval component was invoked."]
            return result
        result["component_integrity"] = integrity

        owns_connection = self._provided_connection is None
        connection = self._provided_connection or _open_read_only(
            self.project_root / self.config["knowledge_base"]["database_path"]
        )
        try:
            invocation.append("resolve_condition")
            resolution = RetrievalRepository(connection).resolve_condition(validated.condition)
            result["condition_context"] = {
                "supplied_condition": validated.condition,
                "resolution": resolution,
                "condition_selection_source": "caller_selected_not_inferred",
            }
            if resolution["status"] != "resolved":
                result["orchestration_state"] = "clarification_required"
                result["safety_gate_result"] = {
                    "state": "clarification_required",
                    "personalization_permitted": False,
                    "reason": (
                        "Unknown condition or alias; no fallback was used."
                        if resolution["status"] == "no_match"
                        else "Condition input is ambiguous and requires clarification."
                    ),
                }
                invocation.append("compose_result")
                result["limitations"] = ["The condition was not inferred from symptom text."]
                return result

            canonical = str(resolution["canonical_name"])
            result["condition_context"]["canonical_name"] = canonical
            try:
                facts = validate_condition_facts(canonical, validated.safety_facts)
            except HybridInputError as error:
                result["orchestration_state"] = "blocked_invalid_input"
                result["safety_gate_result"] = {
                    "state": "blocked_invalid_input", "reason": str(error),
                    "personalization_permitted": False,
                }
                invocation.append("compose_result")
                return result

            gate = self._provided_gate or SafetyGate(
                RuleEngine(connection), self.config["safety_inventory"]
            )
            invocation.append("safety_gate")
            try:
                safety = gate.evaluate(canonical, facts)
            except Exception as error:
                safety = {
                    "state": "blocked_component_failure",
                    "personalization_permitted": False,
                    "reason": f"Safety-gate component failed: {error}",
                    "failure_code": "safety_gate_failure",
                    "missing_structured_fields": [],
                    "prominent_referral_information": [],
                    "rule_engine_result": {},
                    "limited_inventory_disclosure": self.config["safety_inventory"],
                }
            if not safety.get("personalization_permitted"):
                invocation.append("compose_result")
                result["invocation_trace"] = invocation
                result["component_versions"] = self._versions()
                result["limitations"] = self._limitations()
                return compose(result=result, safety=safety)

            model_adapter = self._provided_model or FrozenModelAdapter(
                self.project_root, self.config["frozen_model"]
            )
            invocation.append("frozen_ml_model")
            try:
                model = model_adapter.predict(validated.symptom_text)
            except (ModelIntegrityError, Exception) as error:
                failure = {
                    **safety,
                    "state": "blocked_component_failure",
                    "personalization_permitted": False,
                    "reason": f"Frozen ML component failed: {error}",
                    "failure_code": "frozen_model_failure",
                }
                invocation.append("compose_result")
                result["invocation_trace"] = invocation
                result["component_versions"] = self._versions()
                result["limitations"] = self._limitations() + [
                    "No substitute model or personalized retrieval was used."
                ]
                return compose(result=result, safety=failure)

            retriever = self._provided_retrieval
            if retriever is None:
                retrieval_config = load_retrieval_config(
                    self.project_root / self.config["retrieval"]["config_path"]
                )
                retriever = RetrievalAdapter(
                    ProfileRetriever(connection, config=retrieval_config)
                )
            invocation.append("condition_scoped_retrieval")
            try:
                retrieval = retriever.retrieve(
                    condition=canonical,
                    symptom_text=validated.symptom_text,
                    predicted_labels=model["model_predicted_dosha_labels"],
                    categories=validated.requested_information_categories,
                    top_k=validated.top_k,
                )
            except (RetrievalAdapterError, Exception) as error:
                failure = {
                    **safety,
                    "state": "blocked_component_failure",
                    "personalization_permitted": False,
                    "reason": f"Retrieval component failed: {error}",
                    "failure_code": "retrieval_failure",
                }
                invocation.append("compose_result")
                result["invocation_trace"] = invocation
                result["component_versions"] = self._versions()
                result["limitations"] = self._limitations() + [
                    "No retrieval-derived profile or recommendation was displayed."
                ]
                return compose(result=result, safety=failure, model=model)

            invocation.append("compose_result")
            result["invocation_trace"] = invocation
            result["component_versions"] = self._versions()
            result["limitations"] = self._limitations()
            return compose(result=result, safety=safety, model=model, retrieval=retrieval)
        finally:
            if owns_connection:
                connection.close()

    def _versions(self) -> dict[str, Any]:
        return {
            "hybrid_configuration": self.config["config_version"],
            "model": self.config["frozen_model"]["model_version"],
            "preprocessing": self.config["frozen_model"]["preprocessing_version"],
            "knowledge_base_schema": self.config["knowledge_base"]["schema_version"],
            "retrieval_configuration": "1.0.0",
            "configuration_hashes": {
                "hybrid_configuration": file_sha256(DEFAULT_CONFIG_PATH),
                "model_bundle": self.config["frozen_model"]["bundle_sha256"],
                "knowledge_database": self.config["knowledge_base"]["database_sha256"],
                "rule_catalog": self.config["knowledge_base"]["rule_catalog_sha256"],
                "retrieval_config": self.config["retrieval"]["config_sha256"],
            },
        }

    @staticmethod
    def _limitations() -> list[str]:
        return [
            "Only five approved conditions and five referral-information rules are represented.",
            "There are zero contraindication, exclusion and expert-reviewed rules.",
            "Passing the implemented gate does not establish comprehensive medical safety.",
            "Model outputs target dataset-assigned tags and may generalize weakly to unseen disease profiles.",
            "Reference checking is claim-specific and is not expert review or clinical validation.",
            "Dataset-derived profiles and recommendations may be templated or medically unverified.",
        ]
