"""Claim-by-claim Phase 7 eligibility review and rule-inventory validation."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

from knowledge_base.rule_engine import RuleEngine
from knowledge_base.rule_seed import PRODUCTION_RULES


def _ineligibility_reason(claim: sqlite3.Row) -> str:
    """Explain why a Phase 6 claim remains knowledge rather than an active rule."""

    claim_type = str(claim["claim_type"])
    evidence_status = str(claim["evidence_status"])
    if evidence_status == "reference_checked" and claim_type == "symptom":
        return (
            "Reference-checked descriptive symptom knowledge does not itself contain "
            "a complete testable condition-and-action pair."
        )
    if claim_type == "dataset_dosha_association":
        return (
            "Dataset-assigned Dosha association is medically unverified and is retained "
            "for provenance only; Phase 7 does not infer or resolve Dosha."
        )
    if evidence_status == "dataset_derived":
        return (
            "Dataset-derived text has not been externally checked for this complete claim; "
            "it is not converted into clinical, safety, referral, or prescribing logic."
        )
    if int(claim["phase7_eligible"]) != 1:
        return "The Phase 6 evidence review marked this claim ineligible for rule creation."
    return "The claim lacks an unambiguous, safe condition-and-action representation."


def build_claim_reviews(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return one deterministic catalog row for each of the 45 Phase 6 claims."""

    rule_links: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT re.claim_id, re.rule_id, re.source_id
          FROM rule_evidence AS re
         ORDER BY re.claim_id, re.rule_id, re.source_id
        """
    ):
        rule_links[str(row["claim_id"])].append(
            (str(row["rule_id"]), str(row["source_id"]))
        )

    source_details: dict[str, dict[str, str]] = {}
    for row in connection.execute(
        """
        SELECT source_id, publisher, page_title, url
          FROM evidence_sources
         ORDER BY source_id
        """
    ):
        source_details[str(row["source_id"])] = {
            "publisher": str(row["publisher"]),
            "page_title": str(row["page_title"]),
            "url": str(row["url"]),
        }

    claims = connection.execute(
        """
        SELECT c.canonical_name AS condition, kc.*,
               GROUP_CONCAT(DISTINCT ce.source_locator) AS source_locators,
               MIN(ce.supports_complete_claim) AS complete_support
          FROM knowledge_claims AS kc
          JOIN conditions AS c ON c.condition_id = kc.condition_id
          LEFT JOIN claim_evidence AS ce ON ce.claim_id = kc.claim_id
         GROUP BY kc.claim_id
         ORDER BY c.canonical_name, kc.claim_id
        """
    ).fetchall()

    reviews: list[dict[str, Any]] = []
    for claim in claims:
        links = rule_links.get(str(claim["claim_id"]), [])
        rule_ids = sorted({rule_id for rule_id, _ in links})
        source_ids = sorted({source_id for _, source_id in links})
        if not source_ids:
            source_ids = [
                str(row["source_id"])
                for row in connection.execute(
                    "SELECT source_id FROM claim_evidence WHERE claim_id=? ORDER BY source_id",
                    (claim["claim_id"],),
                )
            ]
        sources = [source_details[source_id] for source_id in source_ids]
        converted = bool(rule_ids)
        reviews.append(
            {
                "condition": str(claim["condition"]),
                "claim_id": str(claim["claim_id"]),
                "claim_type": str(claim["claim_type"]),
                "claim_summary": str(claim["claim_summary"]),
                "evidence_status": str(claim["evidence_status"]),
                "phase6_rule_eligible": bool(claim["phase7_eligible"]),
                "source_profile_id": claim["source_profile_id"],
                "safety_relevance": str(claim["safety_relevance"]),
                "source_ids": source_ids,
                "supporting_source_urls": [source["url"] for source in sources],
                "source_locators": (
                    sorted(str(claim["source_locators"]).split(","))
                    if claim["source_locators"] else []
                ),
                "complete_claim_supported": (
                    bool(claim["complete_support"])
                    if claim["complete_support"] is not None else None
                ),
                "converted_to_rule": converted,
                "rule_ids": rule_ids,
                "conversion_reason": (
                    "Converted because the exact reference-checked claim supplies a "
                    "bounded, testable condition and safe information action."
                    if converted else _ineligibility_reason(claim)
                ),
                "limitations": str(claim["limitations"]),
            }
        )
    return reviews


def validate_rule_inventory(connection: sqlite3.Connection) -> dict[str, Any]:
    """Validate every stored rule's structure and claim/source relationships."""

    engine = RuleEngine(connection)
    rows = connection.execute(
        """
        SELECT kr.*, c.canonical_name
          FROM knowledge_rules AS kr
          JOIN conditions AS c ON c.condition_id = kr.condition_id
         ORDER BY kr.rule_id, kr.rule_version
        """
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        definition, reason = engine._definition_and_support(row)
        results.append(
            {
                "rule_id": str(row["rule_id"]),
                "rule_version": str(row["rule_version"]),
                "condition": str(row["canonical_name"]),
                "valid": reason is None,
                "reason": reason,
                "claim_ids": definition["claim_ids"] if definition else [],
                "source_ids": definition["source_ids"] if definition else [],
            }
        )

    return {
        "rule_count": len(results),
        "valid_rule_count": sum(item["valid"] for item in results),
        "invalid_rule_count": sum(not item["valid"] for item in results),
        "all_rules_valid": all(item["valid"] for item in results),
        "results": results,
    }


def rule_catalog(connection: sqlite3.Connection) -> dict[str, Any]:
    """Build the complete machine-readable Phase 7 rule and claim catalog."""

    reviews = build_claim_reviews(connection)
    return {
        "phase": 7,
        "scope": "structured deterministic rules for caller-supplied condition contexts",
        "rules": list(PRODUCTION_RULES),
        "claim_reviews": reviews,
        "claim_review_summary": {
            "total_claims_reviewed": len(reviews),
            "converted_claims": sum(item["converted_to_rule"] for item in reviews),
            "knowledge_only_claims": sum(not item["converted_to_rule"] for item in reviews),
            "reference_checked_claims": sum(
                item["evidence_status"] == "reference_checked" for item in reviews
            ),
            "dataset_derived_claims": sum(
                item["evidence_status"] == "dataset_derived" for item in reviews
            ),
        },
    }
