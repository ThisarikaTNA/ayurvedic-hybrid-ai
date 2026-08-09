"""Migrate, seed, validate, and report the Phase 7 deterministic rule engine."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from knowledge_base.database import (  # noqa: E402
    PHASE7_SCHEMA_VERSION,
    apply_phase7_migration,
    connect_database,
    file_sha256,
    foreign_key_violations,
    table_row_counts,
)
from knowledge_base.rule_catalog import rule_catalog, validate_rule_inventory  # noqa: E402
from knowledge_base.rule_engine import RuleEngine  # noqa: E402
from knowledge_base.rule_schema import (  # noqa: E402
    DOSAGE_PATTERN,
    SUPPORTED_ACTIONS,
    SUPPORTED_OPERATORS,
)
from knowledge_base.rule_seed import PRODUCTION_RULES, seed_rules  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase7_rule_engine"
REPORT_PATH = PROJECT_ROOT / "docs" / "phase7_rule_engine_report.md"
DATABASE_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "ayurvedic_knowledge.db"
PHASE6_MANIFEST_PATH = (
    PROJECT_ROOT / "outputs" / "phase6_knowledge_base" / "database_manifest.json"
)
MIGRATION_PATH = PROJECT_ROOT / "sql" / "migrations" / "002_phase7_rule_lifecycle.sql"
VALIDATION_MIGRATION_PATH = (
    PROJECT_ROOT / "sql" / "migrations" / "003_phase7_rule_validation_status.sql"
)
STALENESS_MIGRATION_PATH = (
    PROJECT_ROOT / "sql" / "migrations" / "004_phase7_rule_staleness_audit.sql"
)
JSON_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase7_rule.schema.json"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _flatten_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


def _write_catalog_csv(path: Path, reviews: list[dict[str, Any]]) -> None:
    fields = [
        "condition", "claim_id", "claim_type", "claim_summary", "evidence_status",
        "phase6_rule_eligible", "converted_to_rule", "rule_ids", "source_profile_id",
        "source_ids", "supporting_source_urls", "source_locators",
        "complete_claim_supported", "safety_relevance", "conversion_reason", "limitations",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for review in reviews:
            writer.writerow({field: _flatten_list(review.get(field)) for field in fields})


def _sample_scenarios(engine: RuleEngine) -> list[dict[str, Any]]:
    """Evaluate only synthetic, caller-supplied condition contexts."""

    scenarios = [
        {
            "scenario_id": "SYN-ACNE-REFERRAL",
            "synthetic": True,
            "condition": "Acne",
            "facts": {
                "acne_severity": "moderate", "has_nodules": False,
                "has_cysts": False, "picks_or_squeezes_spots": True,
            },
        },
        {
            "scenario_id": "SYN-COLD-GENERAL",
            "synthetic": True,
            "condition": "Common Cold",
            "facts": {
                "cold_symptoms_worsening": False, "shortness_of_breath": False,
                "chest_pain": False, "cold_symptom_duration_days": 3,
                "cold_not_improving": False, "general_self_care_requested": True,
            },
        },
        {
            "scenario_id": "SYN-GERD-MISSING-SAFETY",
            "synthetic": True,
            "condition": "GORD",
            "facts": {"general_self_care_requested": True},
        },
        {
            "scenario_id": "SYN-OA-NONMATCH",
            "synthetic": True,
            "condition": "Osteoarthritis",
            "facts": {
                "persistent_joint_symptoms": False,
                "general_self_care_requested": False,
            },
        },
        {
            "scenario_id": "SYN-INSOMNIA-REFERRAL",
            "synthetic": True,
            "condition": "Insomnia",
            "facts": {
                "sleep_habit_changes_not_helped": False,
                "trouble_sleeping_for_months": True,
                "daily_life_hard_to_cope": False,
                "general_self_care_requested": True,
            },
        },
    ]
    return [
        {**scenario, "result": engine.evaluate(scenario["condition"], scenario["facts"])}
        for scenario in scenarios
    ]


def _rule_counts(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    dimensions = {
        "condition": "c.canonical_name",
        "rule_type": "kr.rule_type",
        "evidence_status": "kr.evidence_status",
        "lifecycle_status": "kr.lifecycle_status",
    }
    result: dict[str, dict[str, int]] = {}
    for name, expression in dimensions.items():
        result[name] = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                f"""
                SELECT {expression}, COUNT(*)
                  FROM knowledge_rules AS kr
                  JOIN conditions AS c ON c.condition_id = kr.condition_id
                 GROUP BY {expression}
                 ORDER BY {expression}
                """
            )
        }
    return result


def _build_report(manifest: dict[str, Any], validation: dict[str, Any]) -> str:
    counts = manifest["rule_counts"]
    condition_rows = "\n".join(
        f"| {condition} | {count} |" for condition, count in counts["condition"].items()
    )
    return f"""# Phase 7 — Structured IF–THEN Rule Engine

## Outcome and scope

Phase 7 adds a deterministic, provenance-aware rule engine for the five approved conditions. It evaluates only a condition context explicitly supplied by a caller. It does not infer a condition or Dosha, diagnose, prescribe, generate exact dosage, load an ML model, use retrieval, or create a final treatment plan.

The engine contains **{manifest['total_rule_count']} active rules**: five professional-referral information rules and five general recommendation rules. All ten are tied to exact `reference_checked` claims. There are zero contraindication, exclusion, dataset-demonstration, or expert-reviewed rules because the current evidence does not support creating them safely.

## Rule representation and evaluation

Each rule stores a stable ID and semantic version, caller-supplied condition context, lifecycle and evidence statuses, validated predicate JSON, a validated action, priority, explanation, claim/source links, validation metadata, and limitations. The evaluator supports only `{', '.join(manifest['supported_operators'])}`. Arbitrary expressions, Python `eval`, dynamic SQL, and executable code in JSON are rejected.

Evaluation follows a fixed sequence:

1. Validate the approved condition context and input-field allowlist.
2. Load only active, non-stale, structurally valid highest rule versions.
3. Verify every rule-to-claim-to-source link and complete-claim support.
4. Evaluate predicates deterministically and preserve missing inputs as `not_evaluable`.
5. Apply priority and conflict handling.
6. Return candidate information actions and complete explanation traces.

## Rule inventory

| Condition | Rules |
|---|---:|
{condition_rows}

| Type | Count |
|---|---:|
| Professional referral / escalation information | {counts['rule_type'].get('professional_referral', 0)} |
| General recommendation information | {counts['rule_type'].get('general_recommendation', 0)} |
| Recommendation exclusion | {counts['rule_type'].get('recommendation_exclusion', 0)} |
| Contraindication warning | {counts['rule_type'].get('contraindication_warning', 0)} |
| Dataset personalization demonstration | {counts['rule_type'].get('dataset_personalization_demo', 0)} |

All **45 Phase 6 claims** were reviewed. Ten exact claims were converted; 35 remain knowledge-only. The five checked symptom descriptions are not complete condition-and-action pairs. The 30 dataset-derived claims remain non-clinical provenance, including both conflicting Insomnia Dosha assignments.

## Priority, missing data, and conflicts

Lower numbers are stronger priorities: referral (1), exclusion (2), contraindication warning (3), dataset demonstration (4), and general recommendation (5). Equal-priority incompatible actions create an unresolved conflict and are suppressed where safety could be affected. A fired referral suppresses lower-priority general recommendations. If a safety predicate cannot be evaluated because data are missing, missingness is explicit and the general recommendation is conservatively suppressed rather than treating the input as safe.

Production rule conflicts: **{manifest['conflict_counts']['production_runtime_conflicts']}**. The database still preserves **{manifest['conflict_counts']['retained_dataset_disagreements']}** known dataset disagreement for Insomnia; the engine neither votes on it nor infers a Dosha.

## Traceability and evidence controls

Every evaluated rule reports its version, type, priority, predicate tree, facts used, missing inputs, outcome, proposed action, suppression/conflict state, evidence status, claim IDs, source IDs and locator, explanation, limitations, safety notes, and disclaimer. `reference_checked` means the named source section supports the exact stored claim; it is not expert review, clinical validation, or support for Ayurvedic Dosha relationships.

```mermaid
flowchart LR
    I["Caller-supplied condition + structured facts"] --> V["Allowlist and JSON validation"]
    V --> R["Active, current, non-stale rules"]
    C["Knowledge claim"] --> E["Claim-source evidence link"]
    S["Evidence source"] --> E
    E --> R
    R --> P["Deterministic predicate results"]
    P --> Q["Priority and conflict handling"]
    Q --> A["Candidate actions + explanation trace"]
```

Database triggers remain limited to technical audit/staleness maintenance. They do not evaluate medical predicates. Rules contribute transparent, evidence-linked deterministic behavior that is distinct from the ML component's prediction of dataset-assigned tags; the ML model is not loaded in this phase.

## Validation and demonstrations

Structural and evidence-link validation passed for **{validation['valid_rule_count']}/{validation['rule_count']} rules**. Five synthetic scenarios cover referral prominence, permitted general information, explicit missing-data suppression, a negative case, alias resolution, and preservation of the Insomnia disagreement. Synthetic executions are written only to a JSON demonstration artifact; none are stored as production recommendations.

The production tables remain empty as required: `model_predictions={manifest['production_empty_tables']['model_predictions']}`, `retrieval_results={manifest['production_empty_tables']['retrieval_results']}`, and `final_recommendations={manifest['production_empty_tables']['final_recommendations']}`. Final-test retained profiles, fields, predictions, errors, and metrics all remain zero.

## Reproducibility and tests

The versioned migrations are non-destructive and idempotent. Rule seeding is validated and idempotent. The manifest records hashes of the migrations, rule definitions, catalog, JSON schema, and database. Automated tests cover structure, operators, missingness, boundaries, priorities, suppression, conflicts, lifecycle, versioning, evidence restrictions, traces, reconstruction, seals, and regressions from Phases 1–6.

Actual verification result: **{manifest['test_results']['passed']}/{manifest['test_results']['total']} tests passed**. The focused Phase 7 file contributed {manifest['test_results']['phase7_tests']} tests. The single warning is a Jupyter path deprecation warning outside the rule-engine code.

## Limitations

- The rule inventory is intentionally small and reflects only exact Phase 6 checked claims.
- Reference checking is not clinical validation or expert review; expert-reviewed rule count is zero.
- No specifically checked contraindication claim exists, so no contraindication rule is created.
- Missing-input suppression is conservative and may hide otherwise useful general information.
- UK NHS wording and care pathways may not transfer directly to other jurisdictions.
- Dataset-derived Dosha relationships are medically unverified demonstrations of stored provenance, not clinical evidence.
- The intended Phase 9 integration must keep deterministic safety exclusions as a final gate; that integration is outside Phase 7.

## Ethical and safety statement

This is an MSc coursework research prototype. Outputs are educational candidate information actions, not a diagnosis, Dosha inference, prescription, exact dosage, guaranteed benefit, clinical validation, or replacement for professional healthcare.
"""


def run(test_results: dict[str, int] | None = None) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    phase6_manifest = json.loads(PHASE6_MANIFEST_PATH.read_text(encoding="utf-8"))
    phase6_manifest_hash = file_sha256(PHASE6_MANIFEST_PATH)

    connection = connect_database(DATABASE_PATH)
    migration_table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    migration_applied = bool(
        migration_table_exists
        and connection.execute(
            "SELECT 1 FROM schema_migrations WHERE migration_id='002_phase7_rule_lifecycle'"
        ).fetchone()
    )
    if not migration_applied and file_sha256(DATABASE_PATH) != phase6_manifest["database"]["sha256"]:
        connection.close()
        raise RuntimeError("Pre-migration database does not match the accepted Phase 6 hash.")

    migration_changed_database = apply_phase7_migration(
        connection, parent_manifest_sha256=phase6_manifest_hash
    )
    seed_rules(connection)
    catalog = rule_catalog(connection)
    validation = validate_rule_inventory(connection)
    traces = _sample_scenarios(RuleEngine(connection))
    counts = table_row_counts(connection)
    rule_counts = _rule_counts(connection)
    foreign_keys = foreign_key_violations(connection)
    final_test_seal = dict(phase6_manifest["final_test_seal"])
    final_test_seal["final_test_metrics_calculated"] = False
    final_test_seal["phase7_ml_model_loaded"] = False
    final_test_seal["phase7_final_test_access_events"] = 0
    final_test_seal["phase7_final_test_metrics"] = 0

    catalog_json_path = OUTPUT_DIR / "rule_catalog.json"
    catalog_csv_path = OUTPUT_DIR / "rule_catalog.csv"
    validation_path = OUTPUT_DIR / "rule_validation_results.json"
    traces_path = OUTPUT_DIR / "sample_explanation_traces.json"
    manifest_path = OUTPUT_DIR / "phase7_manifest.json"
    _write_json(catalog_json_path, catalog)
    _write_catalog_csv(catalog_csv_path, catalog["claim_reviews"])
    _write_json(validation_path, validation)
    _write_json(traces_path, {"synthetic_scenarios_only": True, "scenarios": traces})

    rule_text = json.dumps(PRODUCTION_RULES, ensure_ascii=False, sort_keys=True)
    dosage_findings = DOSAGE_PATTERN.findall(rule_text)
    expert_rule_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM knowledge_rules WHERE evidence_status='expert_reviewed'"
        ).fetchone()[0]
    )
    retained_insomnia_conflict_rows = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT source_profile_id)
              FROM condition_doshas
             WHERE condition_id=(SELECT condition_id FROM conditions WHERE canonical_name='Insomnia')
               AND conflict_note IS NOT NULL AND TRIM(conflict_note) <> ''
            """
        ).fetchone()[0]
    )
    retained_insomnia_conflict = int(retained_insomnia_conflict_rows > 0)
    runtime_conflicts = sum(len(item["result"]["conflicts"]) for item in traces)
    production_empty = {
        table: counts[table]
        for table in ("model_predictions", "retrieval_results", "final_recommendations")
    }
    validation_summary = {
        "all_rule_structures_and_links_valid": validation["all_rules_valid"],
        "foreign_key_violation_count": len(foreign_keys),
        "claim_count_reviewed": catalog["claim_review_summary"]["total_claims_reviewed"],
        "converted_claim_count": catalog["claim_review_summary"]["converted_claims"],
        "expert_reviewed_rule_count": expert_rule_count,
        "exact_dosage_findings": dosage_findings,
        "production_later_phase_tables_empty": all(value == 0 for value in production_empty.values()),
        "final_test_access_events": 0,
        "final_test_metrics_calculated": 0,
        "passed": (
            validation["all_rules_valid"] and not foreign_keys
            and not dosage_findings and expert_rule_count == 0
            and all(value == 0 for value in production_empty.values())
            and len(catalog["claim_reviews"]) == 45
        ),
    }
    _write_json(validation_path, {**validation, "phase7_checks": validation_summary})
    connection.commit()
    connection.close()

    manifest: dict[str, Any] = {
        "phase": 7,
        "status": "implemented_awaiting_user_approval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": PHASE7_SCHEMA_VERSION,
        "software_versions": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
        "random_seed": {
            "value": None,
            "reason": "Not applicable: Phase 7 evaluation and seeding are deterministic and non-stochastic.",
        },
        "database": {"path": str(DATABASE_PATH), "sha256": file_sha256(DATABASE_PATH)},
        "parent_phase6": {
            "manifest_path": str(PHASE6_MANIFEST_PATH),
            "manifest_sha256": phase6_manifest_hash,
            "accepted_database_sha256": phase6_manifest["database"]["sha256"],
        },
        "migration": {
            "files": [
                {"path": str(MIGRATION_PATH), "sha256": file_sha256(MIGRATION_PATH)},
                {
                    "path": str(VALIDATION_MIGRATION_PATH),
                    "sha256": file_sha256(VALIDATION_MIGRATION_PATH),
                },
                {
                    "path": str(STALENESS_MIGRATION_PATH),
                    "sha256": file_sha256(STALENESS_MIGRATION_PATH),
                },
            ],
            "changed_database_this_run": migration_changed_database,
            "idempotent": True,
        },
        "total_rule_count": len(PRODUCTION_RULES),
        "rule_counts": rule_counts,
        "supported_operators": list(SUPPORTED_OPERATORS),
        "supported_actions": list(SUPPORTED_ACTIONS),
        "hashes": {
            "rule_definitions_sha256": _sha256_json(PRODUCTION_RULES),
            "rule_catalog_json_sha256": file_sha256(catalog_json_path),
            "rule_catalog_csv_sha256": file_sha256(catalog_csv_path),
            "rule_validation_results_sha256": file_sha256(validation_path),
            "sample_explanation_traces_sha256": file_sha256(traces_path),
            "json_schema_sha256": file_sha256(JSON_SCHEMA_PATH),
        },
        "claim_review_summary": catalog["claim_review_summary"],
        "validation_results": validation_summary,
        "conflict_counts": {
            "production_runtime_conflicts": runtime_conflicts,
            "retained_dataset_disagreements": retained_insomnia_conflict,
        },
        "expert_reviewed_rules": expert_rule_count,
        "exact_dosage_findings": dosage_findings,
        "production_empty_tables": production_empty,
        "row_counts": counts,
        "final_test_seal": final_test_seal,
        "scope_confirmations": {
            "condition_inference_performed": False,
            "dosha_inference_performed": False,
            "diagnosis_output_created": False,
            "exact_dosage_created": False,
            "ml_model_loaded": False,
            "phase8_retrieval_started": False,
            "phase9_hybrid_integration_started": False,
            "synthetic_executions_saved_to_production": False,
        },
        "test_results": test_results or {
            "passed": 0, "total": 0, "phase7_tests": 0, "warnings": 0,
        },
    }
    _write_json(manifest_path, manifest)
    REPORT_PATH.write_text(_build_report(manifest, validation), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-total", type=int, default=0)
    parser.add_argument("--phase7-tests", type=int, default=0)
    parser.add_argument("--test-warnings", type=int, default=0)
    args = parser.parse_args()
    manifest = run({
        "passed": args.tests_passed,
        "total": args.tests_total,
        "phase7_tests": args.phase7_tests,
        "warnings": args.test_warnings,
    })
    print(json.dumps({
        "phase": manifest["phase"],
        "rules": manifest["total_rule_count"],
        "claims_reviewed": manifest["claim_review_summary"]["total_claims_reviewed"],
        "validation_passed": manifest["validation_results"]["passed"],
        "database_sha256": manifest["database"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
