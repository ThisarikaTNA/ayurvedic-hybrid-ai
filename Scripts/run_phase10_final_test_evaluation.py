"""Prepare and execute the amended, one-time Phase 10 final-test protocol."""

from __future__ import annotations

import argparse
import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from evaluation.final_test_evaluator import (
    DEFAULT_CONFIG_PATH,
    FinalTestEvaluationError,
    PROJECT_ROOT,
    execute_one_time_evaluation,
    file_sha256,
    load_and_validate_frozen_bundle,
    load_config,
    utc_now,
    validate_locked_split_metadata,
    verify_artifact_hashes,
    verify_production_database,
    verify_runtime_versions,
    write_json_exclusive,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10_final_evaluation"
AMENDMENT_002_LOCK = OUTPUT_DIR / "amended_pre_execution_manifest_v2.json"
AMENDMENT_002_CHECKS = OUTPUT_DIR / "amended_integrity_checks_v2.json"
PHASE10_JUNIT = OUTPUT_DIR / "phase10_synthetic_tests_v2.xml"
REGRESSION_JUNIT = OUTPUT_DIR / "phase1_9_regression_tests_v2.xml"
PRESERVED_AMENDMENT_001_LOCKS = {
    "outputs/phase10_final_evaluation/amended_pre_execution_manifest.json": (
        "8529E6BB81D7C931DDABA7E0E863F773442C326E32EC5D35FCB4E6B0D5009ED7"
    ),
    "outputs/phase10_final_evaluation/amended_integrity_checks.json": (
        "F77399A06632FCCF5CDDB22183847BB67A985BB82F4F9342EC7397395F2B3980"
    ),
}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-lock", action="store_true")
    mode.add_argument("--execute-once", action="store_true")
    return command


def _junit_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalTestEvaluationError(f"Required synthetic test record is missing: {path}")
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "sha256": file_sha256(path),
    }
    if summary["tests"] < 1 or summary["failures"] or summary["errors"]:
        raise FinalTestEvaluationError(f"Pre-execution tests did not pass: {path.name} {summary}")
    return summary


def _static_no_fit_check(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden: list[dict[str, Any]] = []
    prediction_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"fit", "fit_transform", "fit_resample"}:
            forbidden.append({"method": node.func.attr, "line": node.lineno})
        if node.func.attr == "predict_proba":
            prediction_calls += 1
    if forbidden or prediction_calls != 1:
        raise FinalTestEvaluationError(
            f"Evaluator call audit failed: forbidden={forbidden}, predict_proba={prediction_calls}."
        )
    return {
        "fit_fit_transform_fit_resample_calls": forbidden,
        "predict_proba_call_sites": prediction_calls,
        "feature_selection_or_training_calls": 0,
        "passed": True,
    }


def _result_outputs_absent(
    config: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    required_lock_hashes: dict[str, str] = PRESERVED_AMENDMENT_001_LOCKS,
) -> dict[str, Any]:
    """Reject result overwrites while accepting exactly two immutable protocol locks."""

    guard = config["one_time_guard"]
    canonical_root = project_root.resolve()
    verified_locks: dict[str, dict[str, str | bool]] = {}
    permitted_existing: set[Path] = set()
    for relative, expected_hash in required_lock_hashes.items():
        declared = Path(relative)
        if declared.is_absolute() or ".." in declared.parts:
            raise FinalTestEvaluationError(
                f"Invalid required Amendment 001 lock path: {relative}"
            )
        expected_path = (canonical_root / declared).resolve()
        if not expected_path.is_file():
            raise FinalTestEvaluationError(
                f"Required immutable Amendment 001 lock is missing: {relative}"
            )
        observed_hash = file_sha256(expected_path)
        if observed_hash != expected_hash:
            raise FinalTestEvaluationError(
                f"Required immutable Amendment 001 lock hash mismatch: {relative}"
            )
        permitted_existing.add(expected_path)
        verified_locks[relative] = {
            "canonical_path": str(expected_path),
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "passed": True,
        }

    candidates = set(config["expected_output_files"]) | {
        guard["abort_marker"], guard["started_marker"], guard["completion_marker"]
    }
    existing = []
    for relative in sorted(candidates):
        candidate = (canonical_root / relative).resolve()
        if candidate.exists() and candidate not in permitted_existing:
            existing.append(relative)
    if existing:
        raise FinalTestEvaluationError(f"Existing Phase 10 result/guard artifact: {existing}")
    return {
        "verified_immutable_amendment_001_locks": verified_locks,
        "existing_result_or_guard_artifacts": [],
        "passed": True,
    }


def _locked_files() -> list[str]:
    return [
        "src/evaluation/final_test_evaluator.py",
        "config/final_test_evaluation.v1.json",
        "scripts/run_phase10_final_test_evaluation.py",
        "schemas/phase10_final_test_metrics.schema.json",
        "schemas/phase10_final_evaluation_manifest.schema.json",
        "tests/test_phase10_final_test_evaluator.py",
        "outputs/phase10_final_evaluation/protocol_amendment_001.json",
        "outputs/phase10_final_evaluation/post_exposure_change_log.json",
        "outputs/phase10_final_evaluation/amended_pre_execution_manifest.json",
        "outputs/phase10_final_evaluation/amended_integrity_checks.json",
        "outputs/phase10_final_evaluation/AMENDED_PRE_EXECUTION_ABORT_002.json",
        "outputs/phase10_final_evaluation/protocol_amendment_002.json",
        "outputs/phase10_final_evaluation/post_exposure_change_log_amendment_002.json",
    ]


def _hash_locked_files() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _locked_files():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FinalTestEvaluationError(f"Lock target is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return hashes


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    artifact_hash_checks = verify_artifact_hashes(PROJECT_ROOT, config)
    runtime = verify_runtime_versions(config)
    database = verify_production_database(PROJECT_ROOT, config)
    _, model = load_and_validate_frozen_bundle(PROJECT_ROOT, config)
    _, split = validate_locked_split_metadata(PROJECT_ROOT, config)
    static = _static_no_fit_check(PROJECT_ROOT / "src/evaluation/final_test_evaluator.py")
    outputs = _result_outputs_absent(config)
    return {
        "artifact_hash_checks": artifact_hash_checks,
        "runtime_versions": runtime,
        "production_database": database,
        "frozen_model_and_preprocessing": model,
        "split_metadata_structure": split,
        "static_prediction_only_audit": static,
        "one_time_output_guard": outputs,
        "passed": True,
    }


def prepare_lock() -> int:
    if AMENDMENT_002_LOCK.exists() or AMENDMENT_002_CHECKS.exists():
        raise FinalTestEvaluationError(
            "Amendment 002 pre-execution lock already exists; overwrite refused."
        )
    config = load_config(DEFAULT_CONFIG_PATH)
    preflight = _preflight(config)
    tests = {
        "phase10_synthetic_tests": _junit_summary(PHASE10_JUNIT),
        "phase1_9_regression_tests": _junit_summary(REGRESSION_JUNIT),
    }
    lock_hashes = _hash_locked_files()
    checks_payload = {
        "phase": 10,
        "status": "amended_pre_execution_checks_passed",
        "checked_at_utc": utc_now(),
        "protocol_amendment": "PHASE10-AMENDMENT-002",
        "preserved_amendment_001_lock_hashes": PRESERVED_AMENDMENT_001_LOCKS,
        "perfect_pre_unsealing_blindness": False,
        "preflight": preflight,
        "tests": tests,
        "no_frozen_component_changed_after_exposure": True,
        "no_exposed_metadata_used_for_evaluator_logic": True,
        "additional_split_manifest_content_displayed": False,
        "passed": True,
    }
    write_json_exclusive(AMENDMENT_002_CHECKS, checks_payload)
    lock_payload = {
        "phase": 10,
        "status": "amended_protocol_locked_before_cleaned_data_read",
        "locked_at_utc": utc_now(),
        "protocol_amendment": "PHASE10-AMENDMENT-002",
        "continuation_authority": ["PHASE10-AMENDMENT-001", "PHASE10-AMENDMENT-002"],
        "original_incident_records_preserved": True,
        "amendment_001_locks_preserved": True,
        "perfect_pre_unsealing_blindness": False,
        "locked_code_and_configuration_hashes": lock_hashes,
        "amendment_002_integrity_checks_sha256": file_sha256(AMENDMENT_002_CHECKS),
        "preserved_amendment_001_lock_hashes": PRESERVED_AMENDMENT_001_LOCKS,
        "preserved_abort_records": {
            "outputs/phase10_final_evaluation/pre_unsealing_manifest.json": (
                "712E3EF9D454BB6860DFE06FC317211C82F2FE5DC372B8C4C454467C13ABF204"
            ),
            "outputs/phase10_final_evaluation/AMENDED_PRE_EXECUTION_ABORT_002.json": (
                "95C205AD70DB76B2D696C4FC4F521FD4089DB144E4DE11451F06B6526C831349"
            ),
        },
        "authorized_diff_scope": (
            "Existing-output collision check, its synthetic regression coverage, "
            "versioned Amendment 002 locks, and required final reporting references only."
        ),
        "test_records": tests,
        "access_counters": {
            "split_manifest_metadata_exposures": 1,
            "cleaned_final_test_data_read_executions": 0,
            "final_test_rows_loaded": 0,
            "final_test_symptom_fields_loaded": 0,
            "batch_prediction_calls": 0,
            "metric_calculation_executions": 0,
            "error_analysis_executions": 0,
            "successful_evaluation_runs": 0,
            "aborted_pre_execution_attempts": 2,
            "aborted_evaluation_runs": 0,
        },
        "one_time_guard_consumed": False,
        "completion_marker_exists": False,
        "class_counts_or_test_content_reported_by_lock": False,
        "ready_for_one_time_execution": True,
    }
    write_json_exclusive(AMENDMENT_002_LOCK, lock_payload)
    print(json.dumps({
        "status": lock_payload["status"],
        "access_counters": lock_payload["access_counters"],
        "ready_for_one_time_execution": True,
    }, indent=2))
    return 0


def execute_once() -> int:
    if not AMENDMENT_002_LOCK.is_file() or not AMENDMENT_002_CHECKS.is_file():
        raise FinalTestEvaluationError("Amendment 002 pre-execution lock is missing.")
    config = load_config(DEFAULT_CONFIG_PATH)
    lock = json.loads(AMENDMENT_002_LOCK.read_text(encoding="utf-8"))
    checks = json.loads(AMENDMENT_002_CHECKS.read_text(encoding="utf-8"))
    if not lock.get("ready_for_one_time_execution") or not checks.get("passed"):
        raise FinalTestEvaluationError("Amended pre-execution checks are not passed and locked.")
    current_hashes = _hash_locked_files()
    if current_hashes != lock.get("locked_code_and_configuration_hashes"):
        raise FinalTestEvaluationError("Evaluator/configuration changed after pre-execution lock.")
    if file_sha256(AMENDMENT_002_CHECKS) != lock.get(
        "amendment_002_integrity_checks_sha256"
    ):
        raise FinalTestEvaluationError("Amendment 002 integrity checks changed after lock.")
    preflight = _preflight(config)
    manifest = execute_one_time_evaluation(
        project_root=PROJECT_ROOT,
        config=config,
        lock_hashes=current_hashes,
        preflight=preflight,
    )
    print(json.dumps({
        "status": manifest["status"],
        "access_counters": manifest["access_counters"],
        "test_partition_profile_count": manifest["test_partition"]["profile_count"],
        "primary_metric": manifest["metrics"]["primary"],
        "validation_comparison": manifest["metrics"]["validation_comparison"],
    }, indent=2))
    return 0


def main() -> int:
    options = parser().parse_args()
    return prepare_lock() if options.prepare_lock else execute_once()


if __name__ == "__main__":
    raise SystemExit(main())
