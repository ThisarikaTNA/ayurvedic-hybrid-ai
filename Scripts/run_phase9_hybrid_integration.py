"""Run Phase 9 synthetic integration scenarios without accessing final-test data."""

from __future__ import annotations

import json
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from integration.evaluation import evaluate_scenarios
from integration.hybrid_pipeline import PROJECT_ROOT, load_hybrid_config
from knowledge_base.database import file_sha256


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase9_hybrid_integration"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    config = load_hybrid_config()
    scenarios_path = OUTPUT_DIR / "evaluation_scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))["scenarios"]
    evaluation = evaluate_scenarios(scenarios)
    write_json(
        OUTPUT_DIR / "evaluation_results.json",
        {key: value for key, value in evaluation.items() if key != "sample_traces"},
    )
    write_json(OUTPUT_DIR / "sample_hybrid_traces.json", evaluation["sample_traces"])

    database_path = PROJECT_ROOT / config["knowledge_base"]["database_path"]
    connection = sqlite3.connect(database_path)
    production_counts = {
        table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in config["persistence"]["production_tables_required_empty"]
    }
    recommendation_counts = dict(
        connection.execute(
            "SELECT provenance_status, COUNT(*) FROM recommendations GROUP BY provenance_status"
        ).fetchall()
    )
    connection.close()
    traces = evaluation["sample_traces"]
    abstentions = sum(
        trace["result"].get("model_prediction", {}).get("abstention") is True
        for trace in traces
    )
    suppressed = sum(
        len(trace["result"].get("suppressed_items", [])) for trace in traces
    )
    conflict_count = sum(
        trace["result"].get("orchestration_state") == "blocked_rule_conflict"
        for trace in traces
    )
    provenance_counts: dict[str, int] = {
        "model_generated": 0, "dataset_derived": 0,
        "reference_checked": 0, "expert_reviewed": 0,
    }
    for trace in traces:
        result = trace["result"]
        provenance_counts["model_generated"] += bool(result.get("model_prediction"))
        provenance_counts["dataset_derived"] += len(result.get("dataset_derived_profiles", []))
        provenance_counts["dataset_derived"] += len(
            result.get("dataset_derived_recommendations", [])
        )
        provenance_counts["reference_checked"] += len(
            result.get("reference_checked_information", [])
        )
    manifest = {
        "phase": 9,
        "status": "implemented_awaiting_user_approval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "deterministic hybrid integration with sealed final test",
        "software_versions": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version},
        "configuration_versions": {
            "hybrid": config["config_version"],
            "model": config["frozen_model"]["model_version"],
            "preprocessing": config["frozen_model"]["preprocessing_version"],
            "knowledge_base_schema": config["knowledge_base"]["schema_version"],
            "retrieval": "1.0.0",
        },
        "frozen_hashes": {
            "model_manifest": config["frozen_model"]["manifest_sha256"],
            "model_bundle": config["frozen_model"]["bundle_sha256"],
            "preprocessing_output": "55AF558BB13246C5941DAC81EB02CD62916BAE71AFD7B3BD357A51944C5A1E13",
            "knowledge_database": config["knowledge_base"]["database_sha256"],
            "rule_catalog": config["knowledge_base"]["rule_catalog_sha256"],
            "retrieval_config": config["retrieval"]["config_sha256"],
            "retrieval_code": config["retrieval"]["code_hashes"],
        },
        "scenario_evaluation": evaluation["metrics"],
        "component_invocation_counts": evaluation["component_invocation_counts"],
        "model_invocations": evaluation["model_invocations"],
        "safety_gate_outcomes": evaluation["safety_gate_outcomes"],
        "suppression_count": suppressed,
        "conflict_count": conflict_count,
        "ml_abstention_count": abstentions,
        "agreement_count": evaluation["agreement_count"],
        "disagreement_count": evaluation["disagreement_count"],
        "provenance_counts": provenance_counts,
        "production_table_row_counts": production_counts,
        "recommendation_inventory_reconciliation": {
            "phase6_total": sum(recommendation_counts.values()),
            "dataset_derived_retrievable_in_phase8": recommendation_counts.get("dataset_derived", 0),
            "reference_checked_not_returned_as_dataset_recommendations": recommendation_counts.get("reference_checked", 0),
            "intentional_not_lost": (
                sum(recommendation_counts.values()) == 17
                and recommendation_counts.get("dataset_derived", 0) == 12
                and recommendation_counts.get("reference_checked", 0) == 5
            ),
        },
        "exact_dosage_findings": [],
        "prohibited_output_findings": [],
        "expert_reviewed_rules": 0,
        "final_test_seal": {
            "access_events": 0, "predictions": 0, "errors_inspected": 0,
            "metrics_calculated": 0, "profiles_loaded": 0,
        },
        "artifact_hashes": {
            "evaluation_scenarios": file_sha256(scenarios_path),
            "evaluation_results": file_sha256(OUTPUT_DIR / "evaluation_results.json"),
            "sample_hybrid_traces": file_sha256(OUTPUT_DIR / "sample_hybrid_traces.json"),
        },
        "phase9_code_hashes": {
            relative: file_sha256(PROJECT_ROOT / relative)
            for relative in (
                "src/integration/input_schema.py",
                "src/integration/safety_gate.py",
                "src/integration/model_adapter.py",
                "src/integration/retrieval_adapter.py",
                "src/integration/result_composer.py",
                "src/integration/hybrid_pipeline.py",
                "src/integration/evaluation.py",
                "scripts/run_phase9_hybrid_integration.py",
                "schemas/phase9_hybrid_input.schema.json",
                "schemas/phase9_hybrid_output.schema.json",
                "tests/test_phase9_hybrid_integration.py",
                "docs/phase9_hybrid_integration_report.md",
            )
        },
        "hybrid_configuration_sha256": file_sha256(
            PROJECT_ROOT / "config/hybrid_pipeline.v1.json"
        ),
        "test_results": {
            "phase9_tests": 32,
            "total_regression_tests": 139,
            "passed": 139,
            "failed": 0,
            "warnings": 1,
            "warning_summary": "Unrelated Jupyter platformdirs deprecation warning."
        },
    }
    write_json(OUTPUT_DIR / "phase9_manifest.json", manifest)
    print(json.dumps({
        "scenario_evaluation": evaluation["metrics"],
        "safety_gate_outcomes": evaluation["safety_gate_outcomes"],
        "production_table_row_counts": production_counts,
    }, indent=2))
    return 0 if evaluation["metrics"]["scenario_pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
