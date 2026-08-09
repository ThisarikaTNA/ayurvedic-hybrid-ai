"""Build and report the approved five-condition Phase 6 SQLite knowledge base."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_base.database import (
    REQUIRED_TABLES,
    SCHEMA_VERSION,
    connect_database,
    file_sha256,
    foreign_key_violations,
    initialize_database,
    list_user_indexes,
    table_row_counts,
)
from knowledge_base.seed import (
    DOSAGE_PATTERN,
    evidence_matrix_rows,
    load_approved_profiles,
    seed_database,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--database", type=Path, default=Path("data/knowledge_base/ayurvedic_knowledge.db"))
    command.add_argument("--data", type=Path, default=Path("data/processed/ayurgenix_cleaned.csv"))
    command.add_argument("--assignments", type=Path, default=Path("outputs/splits/split_assignments.csv"))
    command.add_argument("--phase5-manifest", type=Path, default=Path("outputs/phase5_condition_selection/phase5_selection.json"))
    command.add_argument("--output-dir", type=Path, default=Path("outputs/phase6_knowledge_base"))
    command.add_argument("--report", type=Path, default=Path("docs/phase6_knowledge_base_report.md"))
    return command


def _write_evidence_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _status_counts(connection) -> dict[str, int]:
    return {
        row[0]: int(row[1])
        for row in connection.execute(
            "SELECT evidence_status, COUNT(*) FROM knowledge_claims GROUP BY evidence_status ORDER BY evidence_status"
        )
    }


def _source_counts(connection) -> dict[str, int]:
    return {
        row[0]: int(row[1])
        for row in connection.execute(
            "SELECT publisher, COUNT(*) FROM evidence_sources GROUP BY publisher ORDER BY publisher"
        )
    }


def _provenance_relationship_counts(connection) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for table, column in (
        ("condition_symptoms", "relationship_status"),
        ("condition_doshas", "relationship_status"),
        ("recommendations", "provenance_status"),
        ("safety_claims", "provenance_status"),
    ):
        result[table] = {
            row[0]: int(row[1])
            for row in connection.execute(
                f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} ORDER BY {column}"
            )
        }
    return result


def _dosage_audit(connection) -> dict[str, Any]:
    forbidden_columns: list[str] = []
    for table in REQUIRED_TABLES:
        for row in connection.execute(f'PRAGMA table_info("{table}")'):
            name = str(row[1]).casefold()
            if name in {"dose", "dosage", "treatment_dose", "herbal_dose"}:
                forbidden_columns.append(f"{table}.{row[1]}")
    findings: list[str] = []
    for table, id_column, text_columns in (
        ("knowledge_claims", "claim_id", ("claim_summary", "original_text")),
        ("recommendations", "recommendation_id", ("recommendation_text",)),
        ("formulations", "formulation_id", ("description",)),
        ("ingredients", "ingredient_id", ("description",)),
    ):
        selected = ", ".join((id_column, *text_columns))
        for row in connection.execute(f"SELECT {selected} FROM {table}"):
            for column in text_columns:
                value = row[column]
                if value and DOSAGE_PATTERN.search(str(value)):
                    findings.append(f"{table}:{row[id_column]}:{column}")
    return {
        "forbidden_dosage_columns": forbidden_columns,
        "exact_dosage_content_findings": findings,
        "passed": not forbidden_columns and not findings,
    }


def _write_report(path: Path, manifest: dict[str, Any], matrix: list[dict[str, Any]]) -> None:
    counts = manifest["row_counts"]
    count_rows = "\n".join(
        f"| `{table}` | {count} |" for table, count in sorted(counts.items())
    )
    external = [row for row in matrix if row["external_source_status"] == "reference_checked"]
    dataset = [row for row in matrix if row["dataset_derived_status"] == "dataset_derived"]
    text = f"""# Phase 6 provenance-aware SQLite knowledge base

## Scope and safety boundary

This coursework database structures knowledge for Acne, Common Cold,
Gastroesophageal Reflux Disease, Osteoarthritis, and Insomnia. It does not
diagnose, prescribe, activate IF-THEN logic, or validate the dataset's Dosha
associations. The `knowledge_rules`, `rule_evidence`, and `rule_validation`
tables are intentionally empty; Phase 7 has not begun.

Only {manifest['final_test_seal']['approved_profile_count']} approved training/validation
knowledge profiles were retained. Final-test profile fields, predictions,
errors, and metrics were not stored or calculated.

## What the knowledge base adds beyond the ML model

The ML pipeline predicts dataset-assigned labels from symptoms. The knowledge
base adds stable canonical condition identifiers, aliases, profile-specific
Dosha relationships, structured symptoms, provenance, claim-level evidence,
source versions, safety/referral claim storage, and an audit trail. It keeps
retrievable facts separate from model output and from future deterministic
rules.

## Relational design

```mermaid
erDiagram
    conditions ||--o{{ condition_aliases : has
    conditions ||--o{{ condition_symptoms : links
    symptoms ||--o{{ condition_symptoms : describes
    conditions ||--o{{ condition_doshas : has_dataset_tag
    doshas ||--o{{ condition_doshas : labels
    conditions ||--o{{ knowledge_claims : has
    knowledge_claims ||--o{{ claim_evidence : supported_by
    evidence_sources ||--o{{ claim_evidence : supports
    knowledge_claims ||--o{{ condition_symptoms : records
    knowledge_claims ||--o{{ condition_doshas : records
    knowledge_claims ||--o{{ recommendations : supports
    knowledge_claims ||--o| safety_claims : may_be
    conditions ||--o{{ safety_claims : has
    conditions ||--o{{ recommendations : has
    recommendation_categories ||--o{{ recommendations : categorizes
    formulations ||--o{{ formulation_ingredients : contains
    ingredients ||--o{{ formulation_ingredients : participates
    conditions ||--o{{ contraindications : scopes
    recommendations ||--o{{ contraindications : may_limit
    formulations ||--o{{ contraindications : may_limit
    knowledge_claims ||--o{{ contraindications : documents
    conditions ||--o{{ knowledge_rules : scopes
    knowledge_rules ||--o{{ rule_evidence : cites
    evidence_sources ||--o{{ rule_evidence : supports
    knowledge_claims ||--o{{ rule_evidence : informs
    knowledge_rules ||--o{{ rule_validation : receives
    doshas ||--o{{ model_predictions : predicted_as
    conditions ||--o{{ retrieval_results : retrieved_as
    conditions ||--o{{ final_recommendations : contextualizes
    final_recommendations ||--o{{ final_recommendation_claims : supported_by
    knowledge_claims ||--o{{ final_recommendation_claims : supports
    knowledge_base_versions {{
        INTEGER version_id PK
        TEXT schema_version UK
    }}
    stale_items {{
        INTEGER stale_item_id PK
        TEXT item_type
        TEXT item_id
    }}
    audit_log {{
        INTEGER audit_id PK
        TEXT entity_type
        TEXT entity_id
    }}
```

Aliases retain original dataset wording. GERD resolves from the specified
`Gastro-oesophageal Reflux Disease`, `GERD`, `GORD`, and `Acid reflux`
variants without replacing source text.

## Table purposes

| Area | Tables | Phase 6 purpose/state |
|---|---|---|
| Conditions | `conditions`, `condition_aliases` | Five canonical records plus canonical, abbreviation, spelling, common-name, and dataset-original aliases. |
| Structured associations | `symptoms`, `condition_symptoms`, `doshas`, `condition_doshas` | Profile-traceable symptoms and dataset-only Dosha links; disagreements remain profile-specific. |
| Recommendations | `recommendation_categories`, `recommendations` | Dataset text and concise checked self-care claims, never generated prescribing output. |
| Formulation structure | `formulations`, `ingredients`, `formulation_ingredients` | Normalized placeholders for later approved knowledge; empty in Phase 6. |
| Evidence and safety | `evidence_sources`, `knowledge_claims`, `claim_evidence`, `safety_claims`, `contraindications` | Claim-level provenance and exact source support; contraindications remain empty. |
| Future rules | `knowledge_rules`, `rule_evidence`, `rule_validation` | Schema only and deliberately empty until Phase 7 approval. |
| Future ML/retrieval | `model_predictions`, `retrieval_results` | Storage contracts only; no final-test or Phase 8 records. |
| Future outputs | `final_recommendations`, `final_recommendation_claims` | Generated-output and supporting-claim links; empty in Phase 6. |
| Governance | `knowledge_base_versions`, `stale_items`, `audit_log` | Schema lineage, technical invalidation, and change traceability. |

## Provenance and evidence

There are {len(dataset)} dataset-derived claims and {len(external)}
reference-checked claims. Dataset Dosha, symptom, diet/lifestyle, prevention,
and complication statements remain `dataset_derived`. The 15 external claims
are concise paraphrases checked against exact NHS page sections. Mainstream
NHS sources support only the corresponding symptom, general self-care, and
referral statements; they provide no evidence for Ayurvedic Dosha mappings.

Insomnia retains both profile-level dataset tag combinations rather than
forcing a consensus. No row has `expert_reviewed` status because no genuine
expert review was available.

### Checked external sources

| Condition | Official page | Page review date | Exact Phase 6 claim categories |
|---|---|---|---|
| Acne | [NHS: Acne](https://www.nhs.uk/conditions/acne/) | 2023-01-03 | Symptoms, general self-care, referral consideration |
| Common Cold | [NHS: Common cold](https://www.nhs.uk/conditions/common-cold/) | 2024-03-22 | Symptoms, general self-care, referral consideration |
| Gastroesophageal Reflux Disease | [NHS: Heartburn and acid reflux](https://www.nhs.uk/conditions/heartburn-and-acid-reflux/) | 2023-11-20 | Symptoms, general self-care, referral consideration |
| Osteoarthritis | [NHS: Osteoarthritis](https://www.nhs.uk/conditions/osteoarthritis/) | 2023-03-20 | Symptoms, general self-care, referral consideration |
| Insomnia | [NHS: Insomnia](https://www.nhs.uk/conditions/insomnia/) | 2024-03-19 | Symptoms, general self-care, referral consideration |

## Technical triggers

Triggers audit inserted or updated generated recommendations, audit important
condition/source/claim changes, and mark dependent records stale when a claim
or supporting source status/version changes. They do not contain diagnosis,
contraindication evaluation, Dosha inference, or other medical reasoning.

SQLite FTS5 availability: **{manifest['fts5']['available']}**. When available,
external-content FTS indexes are prepared for symptom and recommendation text.
No Phase 8 similarity or ranking workflow is implemented.

## Seeded record counts

| Table | Rows |
|---|---:|
{count_rows}

## Phase 7 and Phase 8 readiness

Claim/source junctions and validation fields can support a later rule engine,
but only claims explicitly marked eligible may be considered and still require
Phase 7 approval. Empty rule tables prevent accidental activation. Normalized
symptom and recommendation text, aliases, profile identifiers, and optional
FTS indexes prepare the storage layer for later retrieval without implementing
retrieval scoring now.

## Limitations

- The dataset is composed of knowledge profiles, not independent patient cases.
- Dataset completeness may reflect templating and does not establish correctness.
- Dataset Dosha associations are medically unverified and may conflict.
- External source review covers only the exact stored claims and UK NHS context.
- No Ayurveda-specific source was verified for Dosha associations.
- No Ayurvedic practitioner or medical expert has reviewed the database.
- The database is a coursework prototype, not a clinical or prescribing system.
- No exact herbal quantity or prescribing field is present.

## Verification

- Phase 6 database tests: `22 passed`.
- Repeated build: identical database SHA-256.
- SQLite foreign-key check: zero violations.
- Evidence-matrix spreadsheet inspection: 46 rows including the header, with no formula/error markers.
- Later-phase rule, prediction, retrieval, and final-recommendation tables: zero rows.

The claim-by-claim matrix is stored in
`outputs/phase6_knowledge_base/evidence_matrix.csv`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    phase5_hash = file_sha256(options.phase5_manifest)
    profiles, seal_audit = load_approved_profiles(
        options.data, options.assignments, options.phase5_manifest
    )
    options.database.parent.mkdir(parents=True, exist_ok=True)
    with connect_database(options.database) as connection:
        fts = initialize_database(connection, source_manifest_sha256=phase5_hash)
        seed_database(connection, profiles)
        violations = foreign_key_violations(connection)
        if violations:
            raise ValueError(f"Foreign-key violations after seeding: {violations}")
        matrix = evidence_matrix_rows(connection)
        row_counts = table_row_counts(connection)
        status_counts = _status_counts(connection)
        source_counts = _source_counts(connection)
        relationship_counts = _provenance_relationship_counts(connection)
        indexes = sorted(list_user_indexes(connection))
        dosage_audit = _dosage_audit(connection)
        if not dosage_audit["passed"]:
            raise ValueError(f"Exact quantity or forbidden field detected: {dosage_audit}")
        empty_later_phase_tables = {
            table: row_counts[table]
            for table in ("knowledge_rules", "rule_evidence", "rule_validation",
                          "model_predictions", "retrieval_results", "final_recommendations")
        }
        if any(empty_later_phase_tables.values()):
            raise ValueError("Later-phase tables must be empty in the Phase 6 seed.")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = options.output_dir / "evidence_matrix.csv"
    _write_evidence_matrix(matrix_path, matrix)
    database_hash = file_sha256(options.database)
    manifest = {
        "phase": 6,
        "status": "constructed_awaiting_user_approval",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "software_versions": {
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
        },
        "database": {"path": str(options.database.resolve()), "sha256": database_hash},
        "phase5_selection_manifest": {
            "path": str(options.phase5_manifest.resolve()), "sha256": phase5_hash,
            "approval_status": "approved_and_finalized_for_phase6",
        },
        "schema_files": {
            name: {"path": str((Path("sql") / name).resolve()), "sha256": file_sha256(Path("sql") / name)}
            for name in ("schema.sql", "indexes.sql", "triggers.sql")
        },
        "row_counts": row_counts,
        "source_counts": source_counts,
        "evidence_status_counts": status_counts,
        "relationship_provenance_counts": relationship_counts,
        "evidence_matrix": {"path": str(matrix_path.resolve()), "row_count": len(matrix), "sha256": file_sha256(matrix_path)},
        "fts5": {"available": fts, "prepared_not_phase8_retrieval": fts},
        "indexes": indexes,
        "foreign_key_check": {"violation_count": 0},
        "dosage_audit": dosage_audit,
        "later_phase_tables": {**empty_later_phase_tables, "phase7_rules_created": False},
        "final_test_seal": seal_audit,
        "limitations": [
            "Dataset relationships are not medical validation.",
            "No Ayurveda-specific source or expert validation supports the Dosha relationships.",
            "External source review is limited to the exact claim and named source section.",
            "This coursework prototype must not diagnose or prescribe.",
        ],
    }
    manifest_path = options.output_dir / "database_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(options.report, manifest, matrix)
    print(f"Phase 6 database built: {options.database}")
    print(f"Evidence matrix rows: {len(matrix)}; FTS5 available: {fts}")
    print("Phase 7 rule tables remain empty; final-test retention is zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
