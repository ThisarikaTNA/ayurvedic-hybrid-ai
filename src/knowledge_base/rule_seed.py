"""Reproducible Phase 7 production rule definitions and idempotent seeding."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from knowledge_base.rule_schema import validate_rule_definition


RULE_VERSION = "1.0.0"
VALIDATION_DATE = "2026-08-06"
VALIDATOR = "Phase 7 structural and claim-evidence validation; no clinical expert"


def _action(action_type: str, message: str, group: str, value: str, prominence: str) -> dict[str, str]:
    return {
        "type": action_type,
        "message": message,
        "conflict_group": group,
        "value": value,
        "prominence": prominence,
    }


PRODUCTION_RULES: tuple[dict[str, Any], ...] = (
    {
        "rule_id": "R-ACNE-REFERRAL-001", "rule_key": "R-ACNE-REFERRAL-001",
        "rule_version": RULE_VERSION, "rule_name": "Acne professional-assessment information",
        "condition": "Acne", "rule_type": "professional_referral",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"operator": "any", "conditions": [
            {"field": "acne_severity", "operator": "in", "value": ["moderate", "severe"]},
            {"field": "has_nodules", "operator": "eq", "value": True},
            {"field": "has_cysts", "operator": "eq", "value": True},
        ]},
        "action": _action(
            "professional_referral",
            "Reference-checked NHS information advises GP assessment for moderate or severe acne, or when nodules or cysts develop.",
            "care_pathway", "professional_assessment", "high",
        ),
        "priority": 1,
        "explanation_template": "The supplied Acne context matched a specifically checked NHS professional-assessment condition.",
        "claim_ids": ["KC-ACNE-REF-SAFE-001"], "source_ids": ["SRC-NHS-ACNE"],
        "source_locator": "When to seek medical advice", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "UK NHS referral wording; local pathways may differ.",
        "safety_notes": "Candidate referral information only; this engine does not diagnose acne.",
    },
    {
        "rule_id": "R-ACNE-SELFCARE-001", "rule_key": "R-ACNE-SELFCARE-001",
        "rule_version": RULE_VERSION, "rule_name": "Acne spot-handling information",
        "condition": "Acne", "rule_type": "general_recommendation",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"field": "picks_or_squeezes_spots", "operator": "eq", "value": True},
        "action": _action(
            "general_recommendation",
            "Reference-checked NHS information advises avoiding picking or squeezing acne spots because this can worsen them and contribute to scarring.",
            "supportive_information", "avoid_picking_or_squeezing", "normal",
        ),
        "priority": 5,
        "explanation_template": "The supplied input reported picking or squeezing spots, matching the checked self-care statement.",
        "claim_ids": ["KC-ACNE-REF-SELF-001"], "source_ids": ["SRC-NHS-ACNE"],
        "source_locator": "Things you can try if you have acne", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "General information only; it does not replace treatment advice.",
        "safety_notes": "Suppressed when a referral rule fires or cannot be evaluated safely.",
    },
    {
        "rule_id": "R-COLD-REFERRAL-001", "rule_key": "R-COLD-REFERRAL-001",
        "rule_version": RULE_VERSION, "rule_name": "Common-cold professional-assessment information",
        "condition": "Common Cold", "rule_type": "professional_referral",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"operator": "any", "conditions": [
            {"field": "cold_symptoms_worsening", "operator": "eq", "value": True},
            {"field": "shortness_of_breath", "operator": "eq", "value": True},
            {"field": "chest_pain", "operator": "eq", "value": True},
            {"operator": "all", "conditions": [
                {"field": "cold_symptom_duration_days", "operator": "gte", "value": 10},
                {"field": "cold_not_improving", "operator": "eq", "value": True}
            ]}
        ]},
        "action": _action(
            "professional_referral",
            "Reference-checked NHS information advises GP assessment when cold symptoms worsen, include shortness of breath or chest pain, or have not improved after 10 days.",
            "care_pathway", "professional_assessment", "high",
        ),
        "priority": 1,
        "explanation_template": "The supplied Common Cold context matched a checked NHS professional-assessment condition.",
        "claim_ids": ["KC-COLD-REF-SAFE-001"], "source_ids": ["SRC-NHS-COLD"],
        "source_locator": "Non-urgent advice: See a GP if", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "Selected NHS referral examples; not an exhaustive triage protocol.",
        "safety_notes": "Candidate referral information only; breathing or chest symptoms need professional assessment.",
    },
    {
        "rule_id": "R-COLD-SELFCARE-001", "rule_key": "R-COLD-SELFCARE-001",
        "rule_version": RULE_VERSION, "rule_name": "Common-cold general self-care information",
        "condition": "Common Cold", "rule_type": "general_recommendation",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"field": "general_self_care_requested", "operator": "eq", "value": True},
        "action": _action(
            "general_recommendation",
            "Reference-checked NHS information lists rest and adequate fluids as general self-care measures for a common cold.",
            "supportive_information", "show_general_self_care", "normal",
        ),
        "priority": 5,
        "explanation_template": "General self-care information was requested within a supplied Common Cold context.",
        "claim_ids": ["KC-COLD-REF-SELF-001"], "source_ids": ["SRC-NHS-COLD"],
        "source_locator": "How you can treat a cold yourself", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "General information only; individual fluid restrictions are not assessed.",
        "safety_notes": "Suppressed when referral status is positive or not evaluable from supplied facts.",
    },
    {
        "rule_id": "R-GERD-REFERRAL-001", "rule_key": "R-GERD-REFERRAL-001",
        "rule_version": RULE_VERSION, "rule_name": "Reflux professional-assessment information",
        "condition": "Gastroesophageal Reflux Disease", "rule_type": "professional_referral",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"operator": "any", "conditions": [
            {"field": "reflux_self_care_not_helping", "operator": "eq", "value": True},
            {"field": "heartburn_most_days", "operator": "eq", "value": True},
            {"field": "food_sticking", "operator": "eq", "value": True},
            {"field": "frequent_vomiting", "operator": "eq", "value": True},
            {"field": "unexplained_weight_loss", "operator": "eq", "value": True}
        ]},
        "action": _action(
            "professional_referral",
            "Reference-checked NHS information advises GP assessment when self-care does not help, heartburn occurs most days, or swallowing difficulty, frequent vomiting, or unexplained weight loss occurs.",
            "care_pathway", "professional_assessment", "high",
        ),
        "priority": 1,
        "explanation_template": "The supplied reflux context matched a checked NHS professional-assessment condition.",
        "claim_ids": ["KC-GERD-REF-SAFE-001"], "source_ids": ["SRC-NHS-REFLUX"],
        "source_locator": "Non-urgent advice: See a GP if", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "UK NHS wording; chest symptoms require separate emergency assessment outside this rule.",
        "safety_notes": "Candidate referral information only; no condition is inferred from symptoms.",
    },
    {
        "rule_id": "R-GERD-SELFCARE-001", "rule_key": "R-GERD-SELFCARE-001",
        "rule_version": RULE_VERSION, "rule_name": "Reflux general self-care information",
        "condition": "Gastroesophageal Reflux Disease", "rule_type": "general_recommendation",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"field": "general_self_care_requested", "operator": "eq", "value": True},
        "action": _action(
            "general_recommendation",
            "Reference-checked NHS information lists smaller, more frequent meals and avoiding personally triggering foods or drinks as general measures that may reduce heartburn.",
            "supportive_information", "show_general_self_care", "normal",
        ),
        "priority": 5,
        "explanation_template": "General self-care information was requested within a supplied reflux context.",
        "claim_ids": ["KC-GERD-REF-SELF-001"], "source_ids": ["SRC-NHS-REFLUX"],
        "source_locator": "How you can ease heartburn and acid reflux yourself", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "General information only; persistent symptoms require assessment.",
        "safety_notes": "Suppressed when referral status is positive or not evaluable from supplied facts.",
    },
    {
        "rule_id": "R-OA-REFERRAL-001", "rule_key": "R-OA-REFERRAL-001",
        "rule_version": RULE_VERSION, "rule_name": "Osteoarthritis professional-assessment information",
        "condition": "Osteoarthritis", "rule_type": "professional_referral",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"field": "persistent_joint_symptoms", "operator": "eq", "value": True},
        "action": _action(
            "professional_referral",
            "Reference-checked NHS information advises GP assessment for persistent symptoms of osteoarthritis.",
            "care_pathway", "professional_assessment", "high",
        ),
        "priority": 1,
        "explanation_template": "The supplied Osteoarthritis context reported persistent joint symptoms.",
        "claim_ids": ["KC-OA-REF-SAFE-001"], "source_ids": ["SRC-NHS-OA"],
        "source_locator": "Symptoms of osteoarthritis", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "General UK NHS advice; urgent joint presentations are not defined here.",
        "safety_notes": "Candidate referral information only; no diagnosis is generated.",
    },
    {
        "rule_id": "R-OA-SELFCARE-001", "rule_key": "R-OA-SELFCARE-001",
        "rule_version": RULE_VERSION, "rule_name": "Osteoarthritis general self-care information",
        "condition": "Osteoarthritis", "rule_type": "general_recommendation",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"field": "general_self_care_requested", "operator": "eq", "value": True},
        "action": _action(
            "general_recommendation",
            "Reference-checked NHS information lists regular exercise and, when applicable, weight reduction among general measures that can help manage mild osteoarthritis symptoms.",
            "supportive_information", "show_general_self_care", "normal",
        ),
        "priority": 5,
        "explanation_template": "General self-care information was requested within a supplied Osteoarthritis context.",
        "claim_ids": ["KC-OA-REF-SELF-001"], "source_ids": ["SRC-NHS-OA"],
        "source_locator": "Treating osteoarthritis", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "Exercise should be individualized; no exercise prescription is produced.",
        "safety_notes": "Suppressed when referral status is positive or not evaluable from supplied facts.",
    },
    {
        "rule_id": "R-INSOMNIA-REFERRAL-001", "rule_key": "R-INSOMNIA-REFERRAL-001",
        "rule_version": RULE_VERSION, "rule_name": "Insomnia professional-assessment information",
        "condition": "Insomnia", "rule_type": "professional_referral",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"operator": "any", "conditions": [
            {"field": "sleep_habit_changes_not_helped", "operator": "eq", "value": True},
            {"field": "trouble_sleeping_for_months", "operator": "eq", "value": True},
            {"field": "daily_life_hard_to_cope", "operator": "eq", "value": True}
        ]},
        "action": _action(
            "professional_referral",
            "Reference-checked NHS information advises GP assessment when sleep-habit changes have not helped, sleeping difficulty has lasted for months, or insomnia makes daily life hard to cope with.",
            "care_pathway", "professional_assessment", "high",
        ),
        "priority": 1,
        "explanation_template": "The supplied Insomnia context matched a checked NHS professional-assessment condition.",
        "claim_ids": ["KC-INSOMNIA-REF-SAFE-001"], "source_ids": ["SRC-NHS-INSOMNIA"],
        "source_locator": "Non-urgent advice: See a GP if", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "General UK NHS advice; not an emergency or mental-health triage protocol.",
        "safety_notes": "No Dosha inference or resolution of the dataset disagreement is performed.",
    },
    {
        "rule_id": "R-INSOMNIA-SELFCARE-001", "rule_key": "R-INSOMNIA-SELFCARE-001",
        "rule_version": RULE_VERSION, "rule_name": "Insomnia general sleep-habit information",
        "condition": "Insomnia", "rule_type": "general_recommendation",
        "lifecycle_status": "active", "evidence_status": "reference_checked",
        "conditions": {"field": "general_self_care_requested", "operator": "eq", "value": True},
        "action": _action(
            "general_recommendation",
            "Reference-checked NHS information includes keeping a consistent wake time, relaxing before bed, and making the bedroom dark and quiet.",
            "supportive_information", "show_general_self_care", "normal",
        ),
        "priority": 5,
        "explanation_template": "General sleep-habit information was requested within a supplied Insomnia context.",
        "claim_ids": ["KC-INSOMNIA-REF-SELF-001"], "source_ids": ["SRC-NHS-INSOMNIA"],
        "source_locator": "How you can treat insomnia yourself", "validation_status": "valid",
        "last_validation_date": VALIDATION_DATE,
        "limitations": "General sleep-habit information; it does not address every cause of insomnia.",
        "safety_notes": "No Dosha inference or resolution of the dataset disagreement is performed.",
    },
)


def _condition_ids(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        row["canonical_name"]: int(row["condition_id"])
        for row in connection.execute("SELECT condition_id, canonical_name FROM conditions")
    }


def seed_rules(connection: sqlite3.Connection) -> None:
    """Validate and insert the ten approved Phase 7 rule versions idempotently."""

    condition_ids = _condition_ids(connection)
    for rule in PRODUCTION_RULES:
        validate_rule_definition(rule)
        condition_id = condition_ids[rule["condition"]]
        connection.execute(
            """
            INSERT INTO knowledge_rules(
                rule_id, condition_id, rule_version, rule_type, conditions_json,
                action_json, priority, explanation, status, last_validation_date,
                rule_key, rule_name, lifecycle_status, evidence_status,
                structural_validation_status, limitations, safety_notes, is_stale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(rule_id) DO NOTHING
            """,
            (
                rule["rule_id"], condition_id, rule["rule_version"], rule["rule_type"],
                json.dumps(rule["conditions"], sort_keys=True),
                json.dumps(rule["action"], sort_keys=True), rule["priority"],
                rule["explanation_template"], rule["evidence_status"],
                rule["last_validation_date"], rule["rule_key"], rule["rule_name"],
                rule["lifecycle_status"], rule["evidence_status"],
                rule["validation_status"], rule["limitations"], rule["safety_notes"],
            ),
        )
        for claim_id, source_id in zip(rule["claim_ids"], rule["source_ids"], strict=True):
            connection.execute(
                """
                INSERT INTO rule_evidence(rule_id, source_id, claim_id, validation_status)
                VALUES (?, ?, ?, ?) ON CONFLICT(rule_id, source_id, claim_id) DO NOTHING
                """,
                (rule["rule_id"], source_id, claim_id, rule["evidence_status"]),
            )
        exists = connection.execute(
            """
            SELECT 1 FROM rule_validation
             WHERE rule_id=? AND validator=? AND validation_date=? AND validation_status=?
            """,
            (rule["rule_id"], VALIDATOR, VALIDATION_DATE, rule["evidence_status"]),
        ).fetchone()
        if not exists:
            connection.execute(
                """
                INSERT INTO rule_validation(
                    rule_id, validation_status, validator, validation_date, notes,
                    structural_validation_status, validation_type
                ) VALUES (?, ?, ?, ?, ?, 'valid', 'structural_and_evidence_link')
                """,
                (
                    rule["rule_id"], rule["evidence_status"], VALIDATOR,
                    VALIDATION_DATE,
                    "Structured JSON and exact claim/source linkage checked; no expert review.",
                ),
            )
        else:
            connection.execute(
                """
                UPDATE rule_validation
                   SET structural_validation_status='valid',
                       validation_type='structural_and_evidence_link'
                 WHERE rule_id=? AND validator=? AND validation_date=?
                   AND validation_status=?
                   AND (
                       structural_validation_status <> 'valid'
                       OR validation_type <> 'structural_and_evidence_link'
                   )
                """,
                (rule["rule_id"], VALIDATOR, VALIDATION_DATE, rule["evidence_status"]),
            )
    connection.commit()
