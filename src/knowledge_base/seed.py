"""Idempotent Phase 6 seeding from approved unsealed profiles and reviewed sources."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from knowledge_base.repository import normalize_lookup_text


DATASET_VERSION = "AyurGenixAI-cleaned-phase2"
CLAIM_VERSION = "1.0"
ACCESS_DATE = "2026-08-06"
REVIEWER = "Codex AI-assisted source check; no clinical expert review"

APPROVED_CONDITIONS: dict[str, dict[str, Any]] = {
    "acne": {
        "canonical_name": "Acne",
        "description": "Approved dermatology knowledge profile for coursework demonstration.",
        "aliases": [],
    },
    "common cold": {
        "canonical_name": "Common Cold",
        "description": "Approved respiratory knowledge profile for coursework demonstration.",
        "aliases": [],
    },
    "gastroesophageal reflux disease gerd": {
        "canonical_name": "Gastroesophageal Reflux Disease",
        "description": "Approved digestive knowledge profile for coursework demonstration.",
        "aliases": [
            ("Gastro-oesophageal Reflux Disease", "spelling_variant"),
            ("GERD", "abbreviation"),
            ("GORD", "abbreviation"),
            ("Acid reflux", "common_name"),
        ],
    },
    "osteoarthritis": {
        "canonical_name": "Osteoarthritis",
        "description": "Approved musculoskeletal knowledge profile for coursework demonstration.",
        "aliases": [],
    },
    "insomnia": {
        "canonical_name": "Insomnia",
        "description": "Approved sleep-domain knowledge profile for coursework demonstration.",
        "aliases": [],
    },
}


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    page_title: str
    url: str
    review_date: str
    source_version: str


SOURCES: tuple[SourceRecord, ...] = (
    SourceRecord("SRC-NHS-ACNE", "Acne", "https://www.nhs.uk/conditions/acne/", "2023-01-03", "NHS page reviewed 2023-01-03"),
    SourceRecord("SRC-NHS-COLD", "Common cold", "https://www.nhs.uk/conditions/common-cold/", "2024-03-22", "NHS page reviewed 2024-03-22"),
    SourceRecord("SRC-NHS-REFLUX", "Heartburn and acid reflux", "https://www.nhs.uk/conditions/heartburn-and-acid-reflux/", "2023-11-20", "NHS page reviewed 2023-11-20"),
    SourceRecord("SRC-NHS-OA", "Osteoarthritis", "https://www.nhs.uk/conditions/osteoarthritis/", "2023-03-20", "NHS page reviewed 2023-03-20"),
    SourceRecord("SRC-NHS-INSOMNIA", "Insomnia", "https://www.nhs.uk/conditions/insomnia/", "2024-03-19", "NHS page reviewed 2024-03-19"),
)


EXTERNAL_CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "condition": "acne", "claim_id": "KC-ACNE-REF-SYM-001", "claim_type": "symptom",
        "summary": "Acne can involve spots and oily skin; affected skin may sometimes feel hot or painful.",
        "source_id": "SRC-NHS-ACNE", "locator": "Overview and Symptoms of acne",
        "safety": "none", "eligible": False,
        "limitations": "General symptom summary; it is not a diagnostic criterion.",
        "symptoms": ["spots", "oily skin", "skin that may feel hot or painful"],
    },
    {
        "condition": "acne", "claim_id": "KC-ACNE-REF-SELF-001", "claim_type": "general_self_care",
        "summary": "Avoid picking or squeezing acne spots because this can worsen them and contribute to scarring.",
        "source_id": "SRC-NHS-ACNE", "locator": "Things you can try if you have acne",
        "safety": "general", "eligible": True,
        "limitations": "General self-care only; it does not replace treatment advice.",
    },
    {
        "condition": "acne", "claim_id": "KC-ACNE-REF-SAFE-001", "claim_type": "referral_consideration",
        "summary": "NHS advises GP assessment for moderate or severe acne, or when nodules or cysts develop.",
        "source_id": "SRC-NHS-ACNE", "locator": "When to seek medical advice",
        "safety": "referral", "eligible": True,
        "limitations": "UK NHS referral wording; local pathways may differ.",
    },
    {
        "condition": "common cold", "claim_id": "KC-COLD-REF-SYM-001", "claim_type": "symptom",
        "summary": "Common cold symptoms commonly include a blocked or runny nose, sneezing, sore throat, hoarse voice, cough, and feeling tired or unwell.",
        "source_id": "SRC-NHS-COLD", "locator": "Symptoms of a cold",
        "safety": "none", "eligible": False,
        "limitations": "Symptoms overlap with other respiratory infections and are not diagnostic alone.",
        "symptoms": ["blocked or runny nose", "sneezing", "sore throat", "hoarse voice", "cough", "feeling tired or unwell"],
    },
    {
        "condition": "common cold", "claim_id": "KC-COLD-REF-SELF-001", "claim_type": "general_self_care",
        "summary": "Rest and adequate fluids are general self-care measures listed by NHS for a common cold.",
        "source_id": "SRC-NHS-COLD", "locator": "How you can treat a cold yourself",
        "safety": "general", "eligible": True,
        "limitations": "General self-care only; individual fluid restrictions are not assessed here.",
    },
    {
        "condition": "common cold", "claim_id": "KC-COLD-REF-SAFE-001", "claim_type": "referral_consideration",
        "summary": "NHS advises GP assessment when cold symptoms worsen, include shortness of breath or chest pain, or have not improved after 10 days.",
        "source_id": "SRC-NHS-COLD", "locator": "Non-urgent advice: See a GP if",
        "safety": "referral", "eligible": True,
        "limitations": "Selected NHS referral examples; this is not an exhaustive triage protocol.",
    },
    {
        "condition": "gastroesophageal reflux disease gerd", "claim_id": "KC-GERD-REF-SYM-001", "claim_type": "symptom",
        "summary": "The main acid-reflux symptoms listed by NHS are heartburn and an unpleasant sour taste in the mouth.",
        "source_id": "SRC-NHS-REFLUX", "locator": "Symptoms of acid reflux",
        "safety": "none", "eligible": False,
        "limitations": "Symptoms are not specific and do not establish a diagnosis.",
        "symptoms": ["heartburn", "unpleasant sour taste in the mouth"],
    },
    {
        "condition": "gastroesophageal reflux disease gerd", "claim_id": "KC-GERD-REF-SELF-001", "claim_type": "general_self_care",
        "summary": "NHS lists smaller, more frequent meals and avoiding personally triggering foods or drinks as measures that may reduce heartburn.",
        "source_id": "SRC-NHS-REFLUX", "locator": "How you can ease heartburn and acid reflux yourself",
        "safety": "general", "eligible": True,
        "limitations": "General measures only; persistent symptoms require assessment.",
    },
    {
        "condition": "gastroesophageal reflux disease gerd", "claim_id": "KC-GERD-REF-SAFE-001", "claim_type": "referral_consideration",
        "summary": "NHS advises GP assessment when self-care and pharmacy measures do not help, heartburn occurs most days, or swallowing difficulty, frequent vomiting, or unexplained weight loss occurs.",
        "source_id": "SRC-NHS-REFLUX", "locator": "Non-urgent advice: See a GP if",
        "safety": "referral", "eligible": True,
        "limitations": "UK NHS referral wording; chest symptoms still require separate emergency assessment outside this claim.",
    },
    {
        "condition": "osteoarthritis", "claim_id": "KC-OA-REF-SYM-001", "claim_type": "symptom",
        "summary": "Osteoarthritis commonly involves joint pain, stiffness, and problems moving the affected joint.",
        "source_id": "SRC-NHS-OA", "locator": "Symptoms of osteoarthritis",
        "safety": "none", "eligible": False,
        "limitations": "Symptoms overlap with other joint conditions and are not diagnostic alone.",
        "symptoms": ["joint pain", "joint stiffness", "problems moving the affected joint"],
    },
    {
        "condition": "osteoarthritis", "claim_id": "KC-OA-REF-SELF-001", "claim_type": "general_self_care",
        "summary": "NHS lists regular exercise and, when applicable, weight reduction among measures that can help manage mild osteoarthritis symptoms.",
        "source_id": "SRC-NHS-OA", "locator": "Treating osteoarthritis",
        "safety": "general", "eligible": True,
        "limitations": "Exercise should be individualized; this claim does not specify an exercise prescription.",
    },
    {
        "condition": "osteoarthritis", "claim_id": "KC-OA-REF-SAFE-001", "claim_type": "referral_consideration",
        "summary": "NHS advises GP assessment for persistent symptoms of osteoarthritis.",
        "source_id": "SRC-NHS-OA", "locator": "Symptoms of osteoarthritis",
        "safety": "referral", "eligible": True,
        "limitations": "General UK NHS advice; it does not define urgent joint presentations.",
    },
    {
        "condition": "insomnia", "claim_id": "KC-INSOMNIA-REF-SYM-001", "claim_type": "symptom",
        "summary": "Insomnia may involve difficulty falling asleep, repeated waking, early waking, tiredness after waking, daytime irritability, or difficulty concentrating because of tiredness.",
        "source_id": "SRC-NHS-INSOMNIA", "locator": "Symptoms of insomnia",
        "safety": "none", "eligible": False,
        "limitations": "A symptom summary only; causes and other sleep disorders require assessment.",
        "symptoms": ["difficulty falling asleep", "waking repeatedly at night", "waking early and not returning to sleep", "tiredness after waking", "daytime tiredness or irritability", "difficulty concentrating due to tiredness"],
    },
    {
        "condition": "insomnia", "claim_id": "KC-INSOMNIA-REF-SELF-001", "claim_type": "general_self_care",
        "summary": "NHS sleep-habit advice includes keeping a consistent wake time, relaxing before bed, and making the bedroom dark and quiet.",
        "source_id": "SRC-NHS-INSOMNIA", "locator": "How you can treat insomnia yourself",
        "safety": "general", "eligible": True,
        "limitations": "General sleep-habit advice; it does not address every cause of insomnia.",
    },
    {
        "condition": "insomnia", "claim_id": "KC-INSOMNIA-REF-SAFE-001", "claim_type": "referral_consideration",
        "summary": "NHS advises GP assessment when sleep-habit changes have not helped, sleeping difficulty has lasted for months, or insomnia makes daily life hard to cope with.",
        "source_id": "SRC-NHS-INSOMNIA", "locator": "Non-urgent advice: See a GP if",
        "safety": "referral", "eligible": True,
        "limitations": "General UK NHS referral advice; it is not an emergency or mental-health triage rule.",
    },
)


DATA_COLUMNS: tuple[str, ...] = (
    "knowledge_profile_id", "raw_disease", "disease", "normalized_disease",
    "raw_symptoms", "symptoms", "symptom_tokens", "raw_doshas", "doshas",
    "raw_diet_and_lifestyle_recommendations", "diet_and_lifestyle_recommendations",
    "raw_prevention", "prevention", "raw_complications", "complications",
)

DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|milligrams?|g|grams?|mcg|µg|ml|millilit(?:er|re)s?|"
    r"tablets?|capsules?|drops?)\b",
    flags=re.IGNORECASE,
)


def contains_exact_dosage(text: str) -> bool:
    """Detect common exact medicine quantity patterns that Phase 6 must reject."""

    return bool(DOSAGE_PATTERN.search(text))


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_approved_profiles(
    data_path: Path,
    assignments_path: Path,
    phase5_manifest_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load only approved train/validation profiles and retain no test fields."""

    manifest = json.loads(phase5_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "approved_and_finalized_for_phase6":
        raise ValueError("Phase 5 approval has not been finalized.")
    approved = set(manifest["approval"]["approved_conditions"])
    if approved != set(APPROVED_CONDITIONS):
        raise ValueError("Phase 5 approval does not match the Phase 6 condition allowlist.")

    assignments = pd.read_csv(
        assignments_path,
        usecols=["knowledge_profile_id", "normalized_disease", "split"],
        dtype="string",
        keep_default_na=False,
    )
    allowed = assignments[
        assignments["split"].isin(["train", "validation"])
        & assignments["normalized_disease"].isin(approved)
    ].copy()
    allowed_ids = set(allowed["knowledge_profile_id"])

    rows: list[dict[str, str]] = []
    with data_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = sorted(set(DATA_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Cleaned data is missing Phase 6 fields: {missing}")
        for row in reader:
            if row["knowledge_profile_id"] not in allowed_ids:
                continue
            retained = {column: row[column] for column in DATA_COLUMNS}
            if retained["normalized_disease"] not in approved:
                raise ValueError("Approved profile condition does not match its split assignment.")
            rows.append(retained)
    if {row["knowledge_profile_id"] for row in rows} != allowed_ids:
        raise ValueError("Approved source profiles do not match the train/validation allowlist.")

    return rows, {
        "allowed_partitions": ["train", "validation"],
        "approved_profile_count": len(rows),
        "approved_condition_count": len({row["normalized_disease"] for row in rows}),
        "retained_profile_ids": sorted(allowed_ids),
        "final_test_profiles_retained": 0,
        "final_test_fields_retained": 0,
        "final_test_predictions_retained": 0,
        "final_test_errors_retained": 0,
        "final_test_metrics_calculated": False,
    }


def _condition_ids(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("SELECT condition_id, normalized_name FROM conditions")
    return {row["normalized_name"]: int(row["condition_id"]) for row in rows}


def _insert_claim(
    connection: sqlite3.Connection,
    *,
    claim_id: str,
    condition_id: int,
    claim_type: str,
    summary: str,
    original_text: str | None,
    source_profile_id: str | None,
    evidence_status: str,
    safety_relevance: str,
    limitations: str,
    phase7_eligible: bool,
) -> None:
    connection.execute(
        """
        INSERT INTO knowledge_claims(
            claim_id, condition_id, claim_type, claim_summary, original_text,
            normalized_text, source_profile_id, evidence_status, claim_version,
            safety_relevance, limitations, phase7_eligible
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(claim_id) DO NOTHING
        """,
        (
            claim_id, condition_id, claim_type, summary, original_text,
            normalize_lookup_text(summary), source_profile_id, evidence_status,
            CLAIM_VERSION, safety_relevance, limitations, int(phase7_eligible),
        ),
    )


def _insert_symptom(
    connection: sqlite3.Connection,
    condition_id: int,
    symptom_text: str,
    status: str,
    source_profile_id: str,
    source_text: str,
    claim_id: str,
) -> None:
    normalized = normalize_lookup_text(symptom_text)
    connection.execute(
        """
        INSERT INTO symptoms(symptom_text, normalized_text, provenance_status, record_version)
        VALUES (?, ?, ?, ?) ON CONFLICT(normalized_text, provenance_status) DO NOTHING
        """,
        (symptom_text, normalized, status, CLAIM_VERSION),
    )
    symptom_id = connection.execute(
        "SELECT symptom_id FROM symptoms WHERE normalized_text = ? AND provenance_status = ?",
        (normalized, status),
    ).fetchone()[0]
    connection.execute(
        """
        INSERT INTO condition_symptoms(
            condition_id, symptom_id, claim_id, source_profile_id, source_text,
            relationship_status, relationship_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(condition_id, symptom_id, source_profile_id, relationship_status) DO NOTHING
        """,
        (condition_id, symptom_id, claim_id, source_profile_id, source_text, status, CLAIM_VERSION),
    )


def seed_database(
    connection: sqlite3.Connection,
    profiles: Iterable[dict[str, str]],
) -> None:
    """Seed exactly five conditions, dataset provenance, and checked NHS claims."""

    profiles = list(profiles)
    for normalized, config in APPROVED_CONDITIONS.items():
        connection.execute(
            """
            INSERT INTO conditions(
                canonical_name, normalized_name, description, provenance_status, record_version
            ) VALUES (?, ?, ?, 'dataset_derived', ?)
            ON CONFLICT(normalized_name) DO NOTHING
            """,
            (config["canonical_name"], normalize_lookup_text(config["canonical_name"]), config["description"], CLAIM_VERSION),
        )
    ids = _condition_ids(connection)
    # The GERD dataset grouping includes the abbreviation; map it to the canonical normalized name.
    ids["gastroesophageal reflux disease gerd"] = ids["gastroesophageal reflux disease"]

    for normalized, config in APPROVED_CONDITIONS.items():
        condition_id = ids[normalized]
        canonical = config["canonical_name"]
        predefined = [(canonical, "canonical"), *config["aliases"]]
        for alias, alias_type in predefined:
            connection.execute(
                """
                INSERT INTO condition_aliases(
                    condition_id, alias_text, normalized_alias, alias_type,
                    source_profile_id, provenance_status, record_version
                ) VALUES (?, ?, ?, ?, '', 'draft', ?)
                ON CONFLICT(condition_id, normalized_alias, source_profile_id) DO NOTHING
                """,
                (condition_id, alias, normalize_lookup_text(alias), alias_type, CLAIM_VERSION),
            )

    for dosha in ("Vata", "Pitta", "Kapha"):
        connection.execute(
            """
            INSERT INTO doshas(dosha_name, description)
            VALUES (?, ?) ON CONFLICT(dosha_name) DO NOTHING
            """,
            (dosha, "Dataset target label; not a clinically verified patient Dosha."),
        )
    for code, name, description in (
        ("dataset_diet_lifestyle", "Dataset diet and lifestyle", "Unverified dataset-derived text."),
        ("dataset_prevention", "Dataset prevention", "Unverified dataset-derived text."),
        ("reference_self_care", "Reference-checked general self-care", "Concise paraphrase supported by a checked source."),
    ):
        connection.execute(
            """
            INSERT INTO recommendation_categories(category_code, category_name, description)
            VALUES (?, ?, ?) ON CONFLICT(category_code) DO NOTHING
            """,
            (code, name, description),
        )

    for source in SOURCES:
        connection.execute(
            """
            INSERT INTO evidence_sources(
                source_id, publisher, page_title, url, access_date,
                publication_or_review_date, jurisdiction, source_type,
                source_version, validation_status, reviewer, notes
            ) VALUES (?, 'NHS', ?, ?, ?, ?, 'United Kingdom', 'official_health',
                      ?, 'reference_checked', ?, ?)
            ON CONFLICT(source_id) DO NOTHING
            """,
            (
                source.source_id, source.page_title, source.url, ACCESS_DATE,
                source.review_date, source.source_version, REVIEWER,
                "Exact Phase 6 claim content checked; no Ayurvedic Dosha support inferred.",
            ),
        )

    dosha_ids = {
        row["dosha_name"]: int(row["dosha_id"])
        for row in connection.execute("SELECT dosha_id, dosha_name FROM doshas")
    }
    category_ids = {
        row["category_code"]: int(row["category_id"])
        for row in connection.execute("SELECT category_id, category_code FROM recommendation_categories")
    }
    insomnia_sets = {
        row["doshas"] for row in profiles if row["normalized_disease"] == "insomnia"
    }
    insomnia_conflict = len(insomnia_sets) > 1

    for row in sorted(profiles, key=lambda item: item["knowledge_profile_id"]):
        condition_key = row["normalized_disease"]
        condition_id = ids[condition_key]
        profile_id = row["knowledge_profile_id"]
        original_name = _clean_text(row["raw_disease"]) or _clean_text(row["disease"])
        connection.execute(
            """
            INSERT INTO condition_aliases(
                condition_id, alias_text, normalized_alias, alias_type,
                source_profile_id, provenance_status, record_version
            ) VALUES (?, ?, ?, 'dataset_original', ?, 'dataset_derived', ?)
            ON CONFLICT(condition_id, normalized_alias, source_profile_id) DO NOTHING
            """,
            (condition_id, original_name, normalize_lookup_text(original_name), profile_id, DATASET_VERSION),
        )

        claim_specs = (
            ("SYM", "dataset_symptom_profile", "Dataset symptom text", row["raw_symptoms"] or row["symptoms"], "none"),
            ("DOSHA", "dataset_dosha_association", f"Dataset-assigned Dosha tags: {row['doshas']}", row["raw_doshas"] or row["doshas"], "none"),
            ("LIFESTYLE", "dataset_lifestyle_text", "Dataset diet and lifestyle text", row["raw_diet_and_lifestyle_recommendations"] or row["diet_and_lifestyle_recommendations"], "general"),
            ("PREVENT", "dataset_prevention_text", "Dataset prevention text", row["raw_prevention"] or row["prevention"], "general"),
            ("COMPLICATION", "dataset_complication_text", "Dataset complication text", row["raw_complications"] or row["complications"], "caution"),
        )
        for suffix, claim_type, summary_prefix, original, safety in claim_specs:
            original = _clean_text(original)
            summary = f"{summary_prefix} retained from knowledge profile {profile_id}."
            _insert_claim(
                connection, claim_id=f"KC-DATA-{profile_id.upper()}-{suffix}",
                condition_id=condition_id, claim_type=claim_type, summary=summary,
                original_text=original, source_profile_id=profile_id,
                evidence_status="dataset_derived", safety_relevance=safety,
                limitations="Dataset-derived and medically unverified; not eligible for a clinical rule.",
                phase7_eligible=False,
            )

        symptom_claim_id = f"KC-DATA-{profile_id.upper()}-SYM"
        try:
            tokens = json.loads(row["symptom_tokens"])
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid symptom tokens for {profile_id}") from error
        for token in tokens:
            token_text = _clean_text(token)
            if token_text:
                _insert_symptom(
                    connection, condition_id, token_text, "dataset_derived", profile_id,
                    _clean_text(row["raw_symptoms"]), symptom_claim_id,
                )

        dosha_claim_id = f"KC-DATA-{profile_id.upper()}-DOSHA"
        for dosha in (part.strip() for part in row["doshas"].split(",")):
            connection.execute(
                """
                INSERT INTO condition_doshas(
                    condition_id, dosha_id, claim_id, source_profile_id,
                    original_dosha_text, relationship_status, relationship_version,
                    conflict_note
                ) VALUES (?, ?, ?, ?, ?, 'dataset_derived', ?, ?)
                ON CONFLICT(condition_id, dosha_id, source_profile_id) DO NOTHING
                """,
                (
                    condition_id, dosha_ids[dosha], dosha_claim_id, profile_id,
                    _clean_text(row["raw_doshas"]), DATASET_VERSION,
                    "Conflicting profile-level Dosha combinations retained."
                    if condition_key == "insomnia" and insomnia_conflict else None,
                ),
            )

        for category_code, field, suffix in (
            ("dataset_diet_lifestyle", "diet_and_lifestyle_recommendations", "LIFESTYLE"),
            ("dataset_prevention", "prevention", "PREVENT"),
        ):
            recommendation = _clean_text(row[field])
            if not recommendation:
                continue
            if contains_exact_dosage(recommendation):
                raise ValueError(f"Exact quantity-like content rejected for {profile_id}:{field}")
            connection.execute(
                """
                INSERT INTO recommendations(
                    condition_id, category_id, claim_id, recommendation_text,
                    normalized_text, source_profile_id, provenance_status,
                    recommendation_version
                ) VALUES (?, ?, ?, ?, ?, ?, 'dataset_derived', ?)
                ON CONFLICT(condition_id, category_id, normalized_text, source_profile_id, provenance_status) DO NOTHING
                """,
                (
                    condition_id, category_ids[category_code],
                    f"KC-DATA-{profile_id.upper()}-{suffix}", recommendation,
                    normalize_lookup_text(recommendation), profile_id, DATASET_VERSION,
                ),
            )

        complication_claim = f"KC-DATA-{profile_id.upper()}-COMPLICATION"
        connection.execute(
            """
            INSERT INTO safety_claims(
                condition_id, claim_id, safety_level, safety_summary,
                provenance_status, safety_version
            ) VALUES (?, ?, 'information', ?, 'dataset_derived', ?)
            ON CONFLICT(claim_id) DO NOTHING
            """,
            (
                condition_id, complication_claim,
                f"Unverified dataset complication text retained for profile {profile_id}.",
                DATASET_VERSION,
            ),
        )

    for claim in EXTERNAL_CLAIMS:
        condition_id = ids[claim["condition"]]
        _insert_claim(
            connection, claim_id=claim["claim_id"], condition_id=condition_id,
            claim_type=claim["claim_type"], summary=claim["summary"],
            original_text=None, source_profile_id=None,
            evidence_status="reference_checked", safety_relevance=claim["safety"],
            limitations=claim["limitations"], phase7_eligible=bool(claim["eligible"]),
        )
        connection.execute(
            """
            INSERT INTO claim_evidence(
                claim_id, source_id, source_locator, supports_complete_claim,
                validation_status, reviewer, notes, evidence_version
            ) VALUES (?, ?, ?, 1, 'reference_checked', ?, ?, ?)
            ON CONFLICT(claim_id, source_id, source_locator) DO NOTHING
            """,
            (
                claim["claim_id"], claim["source_id"], claim["locator"], REVIEWER,
                "Concise paraphrase checked against the named section; no Dosha inference.",
                CLAIM_VERSION,
            ),
        )
        for symptom in claim.get("symptoms", []):
            _insert_symptom(
                connection, condition_id, symptom, "reference_checked", "",
                claim["summary"], claim["claim_id"],
            )
        if claim["claim_type"] == "general_self_care":
            if contains_exact_dosage(claim["summary"]):
                raise ValueError(f"Exact quantity-like external content rejected: {claim['claim_id']}")
            connection.execute(
                """
                INSERT INTO recommendations(
                    condition_id, category_id, claim_id, recommendation_text,
                    normalized_text, source_profile_id, provenance_status,
                    recommendation_version
                ) VALUES (?, ?, ?, ?, ?, '', 'reference_checked', ?)
                ON CONFLICT(condition_id, category_id, normalized_text, source_profile_id, provenance_status) DO NOTHING
                """,
                (
                    condition_id, category_ids["reference_self_care"], claim["claim_id"],
                    claim["summary"], normalize_lookup_text(claim["summary"]), CLAIM_VERSION,
                ),
            )
        if claim["claim_type"] == "referral_consideration":
            connection.execute(
                """
                INSERT INTO safety_claims(
                    condition_id, claim_id, safety_level, safety_summary,
                    provenance_status, safety_version
                ) VALUES (?, ?, 'referral', ?, 'reference_checked', ?)
                ON CONFLICT(claim_id) DO NOTHING
                """,
                (condition_id, claim["claim_id"], claim["summary"], CLAIM_VERSION),
            )

    connection.commit()


def evidence_matrix_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return one auditable evidence-matrix row per knowledge claim."""

    rows = connection.execute(
        """
        SELECT c.canonical_name AS condition, kc.claim_id, kc.claim_type,
               kc.claim_summary, kc.evidence_status, kc.safety_relevance,
               kc.limitations, kc.phase7_eligible, es.publisher, es.page_title,
               es.url, ce.supports_complete_claim, ce.validation_status,
               ce.source_locator
          FROM knowledge_claims AS kc
          JOIN conditions AS c ON c.condition_id = kc.condition_id
          LEFT JOIN claim_evidence AS ce ON ce.claim_id = kc.claim_id
          LEFT JOIN evidence_sources AS es ON es.source_id = ce.source_id
         ORDER BY c.canonical_name, kc.claim_id
        """
    )
    matrix: list[dict[str, Any]] = []
    for row in rows:
        dataset_derived = row["evidence_status"] == "dataset_derived"
        supporting_source = ""
        if row["url"]:
            supporting_source = f"{row['publisher']}: {row['page_title']} | {row['url']}"
        matrix.append(
            {
                "condition": row["condition"],
                "claim_id": row["claim_id"],
                "claim_type": row["claim_type"],
                "claim_summary": row["claim_summary"],
                "dataset_derived_status": "dataset_derived" if dataset_derived else "not_dataset_derived",
                "external_source_status": row["validation_status"] or "not_externally_checked",
                "supporting_source": supporting_source,
                "source_locator_or_section": row["source_locator"] or "",
                "source_supports_complete_claim": (
                    "yes" if row["supports_complete_claim"] == 1 else
                    "no" if row["supports_complete_claim"] == 0 else "not_applicable"
                ),
                "safety_relevance": row["safety_relevance"],
                "limitations": row["limitations"],
                "eligible_for_phase7_rule_creation": "yes" if row["phase7_eligible"] else "no",
            }
        )
    return matrix
