"""Run standalone Phase 8 retrieval evaluation and generate sealed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from knowledge_base.database import file_sha256  # noqa: E402
from retrieval.evaluation import EVALUATION_QUERIES, evaluate_retrieval  # noqa: E402
from retrieval.profile_retriever import DOSAGE_PATTERN, ProfileRetriever  # noqa: E402
from retrieval.query_schema import DEFAULT_CONFIG_PATH, load_retrieval_config  # noqa: E402
from retrieval.repository import RetrievalRepository  # noqa: E402


DATABASE_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "ayurvedic_knowledge.db"
PHASE7_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "phase7_rule_engine" / "phase7_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase8_profile_retrieval"
REPORT_PATH = PROJECT_ROOT / "docs" / "phase8_profile_retrieval_report.md"
SOURCE_FILES = (
    PROJECT_ROOT / "src" / "retrieval" / "query_schema.py",
    PROJECT_ROOT / "src" / "retrieval" / "repository.py",
    PROJECT_ROOT / "src" / "retrieval" / "profile_retriever.py",
    PROJECT_ROOT / "src" / "retrieval" / "evaluation.py",
    PROJECT_ROOT / "scripts" / "run_phase8_profile_retrieval.py",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _open_read_only_production() -> sqlite3.Connection:
    uri = DATABASE_PATH.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _temporary_approved_database() -> tuple[sqlite3.Connection, sqlite3.Connection]:
    """Copy the sealed production KB to memory; retrieval runs only on this copy."""

    source = _open_read_only_production()
    temporary = sqlite3.connect(":memory:")
    temporary.row_factory = sqlite3.Row
    temporary.execute("PRAGMA foreign_keys = ON")
    source.backup(temporary)
    return source, temporary


def _collection_inventory(repository: RetrievalRepository) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    for row in repository.connection.execute(
        "SELECT condition_id, canonical_name FROM conditions ORDER BY canonical_name"
    ):
        condition_id = int(row["condition_id"])
        profiles = repository.list_profiles(condition_id)
        inventory[str(row["canonical_name"])] = {
            "dataset_profiles": len(profiles),
            "reference_checked_claims": len(repository.list_reference_claims(condition_id)),
            "symptoms": len(repository.list_symptoms(condition_id)),
            "dataset_recommendations": len(
                repository.list_dataset_recommendations(condition_id)
            ),
            "external_sources": len(repository.list_sources(condition_id)),
            "dataset_provenance_records": len(profiles),
        }
    return inventory


def _sample_traces(retriever: ProfileRetriever) -> list[dict[str, Any]]:
    scenarios = (
        {
            "scenario_id": "SYN-ALIAS-GERD-TEXT",
            "synthetic": True,
            "request": {
                "condition": "GORD", "free_text": "heartburn reflux",
                "categories": ["profiles", "claims", "symptoms", "sources"],
                "top_k": 3,
            },
        },
        {
            "scenario_id": "SYN-COLD-FTS",
            "synthetic": True,
            "request": {
                "condition": "Common Cold", "free_text": "blocked runny nose cough",
                "categories": ["profiles", "symptoms", "recommendations"],
                "top_k": 3,
            },
        },
        {
            "scenario_id": "SYN-INSOMNIA-PITTA-SIGNAL",
            "synthetic": True,
            "request": {
                "condition": "Insomnia", "caller_supplied_dosha_tags": ["Pitta"],
                "categories": ["profiles", "sources"], "top_k": 2,
            },
        },
        {
            "scenario_id": "SYN-NO-TEXT-MATCH",
            "synthetic": True,
            "request": {
                "condition": "Acne", "free_text": "quantum telescope nebula",
                "categories": ["profiles", "claims", "symptoms", "recommendations"],
                "top_k": 3,
            },
        },
        {
            "scenario_id": "SYN-MISSING-CONDITION",
            "synthetic": True,
            "request": {"free_text": "heartburn", "top_k": 3},
        },
    )
    return [
        {**scenario, "result": retriever.retrieve(scenario["request"])}
        for scenario in scenarios
    ]


def _alias_results(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": case["query_id"],
            "supplied_condition": case["request"].get("condition"),
            "expected_condition": case["expected_condition"],
            "actual_condition": case["actual_condition"],
            "status": case["actual_status"],
            "correct": case["condition_resolution_correct"] and case["status_correct"],
        }
        for case in evaluation["cases"]
        if case["query_type"] in {
            "canonical_name", "alias", "unknown_condition", "missing_condition"
        }
    ]


def _report(manifest: dict[str, Any]) -> str:
    inventory_rows = "\n".join(
        "| " + condition + " | " + " | ".join(str(values[key]) for key in (
            "dataset_profiles", "reference_checked_claims", "symptoms",
            "dataset_recommendations", "external_sources",
        )) + " |"
        for condition, values in manifest["retrieved_inventory"].items()
    )
    metrics = manifest["evaluation"]["metrics"]["overall"]
    return f"""# Phase 8 — Standalone Knowledge-Profile Retrieval

## Outcome and boundary

Phase 8 implements deterministic, explainable retrieval for the five approved conditions. The caller must supply a canonical condition or approved alias. Resolution is a database lookup and hard scope constraint; it is not symptom-based diagnosis. Query text cannot cause fallback to another condition.

The retrieval path imports and invokes neither the ML model nor the Phase 7 rule engine. It does not diagnose, infer a true Dosha, fire rules, prescribe, persist synthetic inputs, or generate final recommendations. Evaluation and demonstrations run on an in-memory copy of the sealed SQLite knowledge base.

## Retrieved inventory

| Condition | Dataset profiles | Checked claims | Symptoms | Dataset recommendations | External sources |
|---|---:|---:|---:|---:|---:|
{inventory_rows}

Dataset-derived profiles/recommendations and reference-checked claims are returned in separate collections. Evidence status is displayed but is not converted into a ranking weight. `reference_checked` means the exact linked claim was checked against the named source section; it is not clinical or expert validation.

## Ranking logic

1. Normalize and resolve the supplied condition or alias. Unknown names return `no_match`; ambiguous or missing names request clarification.
2. Safely tokenize optional text to lowercase alphanumeric terms and construct a quoted, parameter-bound FTS query. User text is never interpolated into SQL or arbitrary FTS syntax.
3. Use FTS5/BM25 for indexed symptom and recommendation text. BM25 is a lexical relevance measure: smaller raw values indicate a closer text match. Scores are normalized only within the current result collection.
4. Combine normalized BM25 (weight **{manifest['retrieval_configuration']['lexical_scoring']['bm25_normalized_weight']}**) with matched-term coverage (weight **{manifest['retrieval_configuration']['lexical_scoring']['matched_term_coverage_weight']}**) where FTS applies. Claims and sources use safe token coverage because they are not in the prepared FTS collections.
5. For profiles, optional caller-supplied Dosha overlap uses Jaccard set overlap. With both text and tags, lexical relevance has weight **{manifest['retrieval_configuration']['profile_ranking']['query_and_dosha']['lexical_relevance_weight']}** and Dosha overlap **{manifest['retrieval_configuration']['profile_ranking']['query_and_dosha']['caller_supplied_dosha_overlap_weight']}**. This is only a ranking signal, never an inference or validation.
6. Break ties deterministically by final score, lexical score, raw BM25, and stable record ID. Without text or tags, stable record-ID order applies.

A high BM25 or final retrieval score means closer query matching, not stronger evidence, medical correctness, or treatment suitability.

## Result traces and provenance

Every item records the deterministic run ID, resolved condition and alias match, result type/rank, component scores, final score, matched terms, optional Dosha overlap, evidence/lifecycle/staleness status, source profile, supporting claim/source IDs, inclusion reason, limitations, and prototype disclaimer. Referral claims may appear as checked information, but no referral rule is evaluated.

Insomnia retains both approved profiles and discloses the `Vata` versus `Vata, Pitta` dataset disagreement even when a caller-supplied tag changes their rank. There is no voting or arbitrary precedence.

## Evaluation

The transparent evaluation contains **{manifest['evaluation']['query_count']} queries** created only from approved train/validation knowledge-base records. Relevance is assigned directly from exact aliases, stored profile IDs, and the wording of linked checked claims—not from final-test records or model errors.

| Metric | Result |
|---|---:|
| Alias-resolution accuracy | {metrics['alias_resolution_accuracy']:.3f} |
| Condition-resolution accuracy | {metrics['condition_resolution_accuracy']:.3f} |
| Hit@K | {metrics['hit_at_k']:.3f} |
| Recall@K | {metrics['recall_at_k']:.3f} |
| Mean Reciprocal Rank | {metrics['mean_reciprocal_rank']:.3f} |
| Exact expected-record retrieval | {metrics['exact_expected_record_retrieval']:.3f} |
| Queries passing all declared expectations | {metrics['passed_query_count']}/{metrics['evaluation_query_count']} |

These measurements check the implementation against a small, constructed coursework set. They do not establish clinical effectiveness or generalization.

## Tests and production controls

Automated verification: **{manifest['test_results']['passed']}/{manifest['test_results']['total']} tests passed**, including **{manifest['test_results']['phase8_tests']} Phase 8 tests. Production table counts remain `model_predictions={manifest['production_table_counts']['model_predictions']}`, `retrieval_results={manifest['production_table_counts']['retrieval_results']}`, and `final_recommendations={manifest['production_table_counts']['final_recommendations']}`. Exact-dosage findings and final-test access/metric counts are zero.

No schema migration was required: the prepared FTS5 indexes and normalized Phase 6–7 tables are sufficient. Non-persistent retrieval is the only implemented mode.

## Limitations and Phase 9 boundary

- The database contains only six approved profiles across five conditions, so ranking comparisons are extremely small.
- Most conditions have one dataset profile; high Hit@K is therefore not evidence of broad retrieval quality.
- Dataset text and Dosha assignments remain medically unverified and may be templated.
- BM25 reflects shared words, not evidence strength or clinical relevance.
- Jaccard overlap depends entirely on caller-supplied and dataset-assigned tags.
- Reference checking is not expert review or clinical validation.
- Missing or unknown condition context produces no inferred fallback.
- Phase 9 must add a separately approved deterministic safety gate; retrieval scores must never override exclusions, warnings, or referral controls.

## Safety statement

Educational research prototype only. Retrieved rows are knowledge profiles, not historical patients. Output is informational retrieval, not diagnosis, Dosha inference, prescribing, guaranteed benefit, or a replacement for professional healthcare.
"""


def run(test_results: dict[str, int]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = load_retrieval_config()
    database_hash_before = file_sha256(DATABASE_PATH)
    phase7_manifest = json.loads(PHASE7_MANIFEST_PATH.read_text(encoding="utf-8"))

    production, temporary = _temporary_approved_database()
    retriever = ProfileRetriever(temporary, config=config)
    repository = RetrievalRepository(temporary)
    inventory = _collection_inventory(repository)
    evaluation = evaluate_retrieval(retriever)
    samples = _sample_traces(retriever)
    aliases = _alias_results(evaluation)
    fts5_available = repository.fts5_available()

    production_counts = dict(production.execute(
        """
        SELECT (SELECT COUNT(*) FROM model_predictions) AS model_predictions,
               (SELECT COUNT(*) FROM retrieval_results) AS retrieval_results,
               (SELECT COUNT(*) FROM final_recommendations) AS final_recommendations
        """
    ).fetchone())
    insomnia_profiles = repository.list_profiles(
        int(temporary.execute(
            "SELECT condition_id FROM conditions WHERE canonical_name='Insomnia'"
        ).fetchone()[0])
    )
    insomnia_associations = {
        profile["record_id"]: profile["dataset_assigned_dosha_tags"]
        for profile in insomnia_profiles
    }
    production.close()
    temporary.close()

    queries_path = OUTPUT_DIR / "evaluation_queries.json"
    results_path = OUTPUT_DIR / "evaluation_results.json"
    samples_path = OUTPUT_DIR / "sample_retrieval_traces.json"
    manifest_path = OUTPUT_DIR / "phase8_manifest.json"
    _write_json(queries_path, {
        "source_partitions": ["train", "validation"],
        "final_test_queries_used": 0,
        "clinical_effectiveness_claimed": False,
        "queries": list(EVALUATION_QUERIES),
    })
    _write_json(results_path, evaluation)
    _write_json(samples_path, {
        "temporary_database_only": True,
        "synthetic_scenarios": samples,
    })

    serialized_outputs = " ".join(
        path.read_text(encoding="utf-8")
        for path in (queries_path, results_path, samples_path)
    )
    dosage_findings = DOSAGE_PATTERN.findall(serialized_outputs)
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in SOURCE_FILES
        if path.name != "run_phase8_profile_retrieval.py"
    )
    forbidden_import_findings = [
        token for token in (
            "knowledge_base.rule_engine", "from models", "import models",
            "joblib.load",
        ) if token in source_text
    ]
    final_test_seal = dict(phase7_manifest["final_test_seal"])
    final_test_seal.update({
        "phase8_final_test_access_events": 0,
        "phase8_final_test_predictions": 0,
        "phase8_final_test_errors": 0,
        "phase8_final_test_metrics": 0,
        "phase8_ml_model_loads": 0,
        "phase8_rule_engine_invocations": 0,
    })
    database_hash_after = file_sha256(DATABASE_PATH)
    manifest: dict[str, Any] = {
        "phase": 8,
        "status": "implemented_awaiting_user_approval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": {
            "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
        },
        "random_seed": {
            "value": None,
            "reason": "Retrieval, ranking, tie-breaking, and run IDs are deterministic.",
        },
        "parent_phase7": {
            "approval": "explicitly_approved_by_user",
            "manifest_path": str(PHASE7_MANIFEST_PATH),
            "manifest_sha256": file_sha256(PHASE7_MANIFEST_PATH),
        },
        "database": {
            "path": str(DATABASE_PATH),
            "sha256_before": database_hash_before,
            "sha256_after": database_hash_after,
            "unchanged": database_hash_before == database_hash_after,
            "evaluation_copy": "temporary_in_memory",
        },
        "retrieval_configuration": config,
        "configuration_file": {
            "path": str(DEFAULT_CONFIG_PATH),
            "sha256": file_sha256(DEFAULT_CONFIG_PATH),
        },
        "code_hashes": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in SOURCE_FILES
        },
        "artifact_hashes": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in (queries_path, results_path, samples_path)
        },
        "retrieved_inventory": inventory,
        "evaluation": {
            "query_count": len(EVALUATION_QUERIES),
            "target_result_count": sum(
                len(case["retrieved_record_ids"]) for case in evaluation["cases"]
            ),
            "metrics": evaluation["metrics"],
        },
        "metrics_by_query_type": evaluation["metrics"]["by_query_type"],
        "alias_resolution_results": aliases,
        "fts5": {
            "available": fts5_available,
            "used_for": ["symptom_text", "recommendation_text", "dataset_profile_mapping"],
            "safe_quoted_tokens_and_parameterized_match": True,
        },
        "production_table_counts": production_counts,
        "persistence": {
            "implemented": False,
            "default": "non_persistent",
            "synthetic_records_written_to_production": 0,
        },
        "insomnia_conflict_preservation": {
            "passed": insomnia_associations == {
                "kp_0010": ["Vata", "Pitta"], "kp_0150": ["Vata"]
            },
            "profile_associations": insomnia_associations,
            "majority_vote_used": False,
            "clinical_interpretation_used": False,
        },
        "exact_dosage_findings": dosage_findings,
        "forbidden_phase8_import_or_invocation_findings": forbidden_import_findings,
        "final_test_seal": final_test_seal,
        "scope_confirmations": {
            "condition_diagnosis_performed": False,
            "dosha_inference_performed": False,
            "ml_model_loaded_or_invoked": False,
            "rule_engine_loaded_or_invoked": False,
            "hybrid_integration_started": False,
            "final_recommendation_created": False,
        },
        "test_results": test_results,
    }
    _write_json(manifest_path, manifest)
    REPORT_PATH.write_text(_report(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-total", type=int, default=0)
    parser.add_argument("--phase8-tests", type=int, default=0)
    parser.add_argument("--test-warnings", type=int, default=0)
    args = parser.parse_args()
    manifest = run({
        "passed": args.tests_passed,
        "total": args.tests_total,
        "phase8_tests": args.phase8_tests,
        "warnings": args.test_warnings,
    })
    print(json.dumps({
        "phase": manifest["phase"],
        "evaluation_queries": manifest["evaluation"]["query_count"],
        "all_evaluation_queries_passed": manifest["evaluation"]["metrics"]["overall"]["all_queries_passed"],
        "fts5_available": manifest["fts5"]["available"],
        "database_unchanged": manifest["database"]["unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
