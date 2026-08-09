"""Small read-oriented repository for Phase 6 knowledge-base inspection."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from typing import Any


def normalize_lookup_text(value: str) -> str:
    """Normalize case, apostrophes, punctuation, and whitespace for alias lookup."""

    text = unicodedata.normalize("NFKC", value).replace("’", "'").casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class KnowledgeRepository:
    """Provide explicit queries without embedding medical reasoning."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def resolve_condition(self, name_or_alias: str) -> sqlite3.Row | None:
        normalized = normalize_lookup_text(name_or_alias)
        return self.connection.execute(
            """
            SELECT DISTINCT c.*
              FROM conditions AS c
              LEFT JOIN condition_aliases AS a ON a.condition_id = c.condition_id
             WHERE c.normalized_name = ? OR a.normalized_alias = ?
             ORDER BY c.condition_id
             LIMIT 1
            """,
            (normalized, normalized),
        ).fetchone()

    def list_conditions(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM conditions ORDER BY canonical_name"
            )
        )

    def list_dosha_relationships(self, condition_id: int) -> list[sqlite3.Row]:
        """Return profile-specific dataset labels so disagreements stay visible."""

        return list(
            self.connection.execute(
                """
                SELECT cd.source_profile_id, d.dosha_name, cd.original_dosha_text,
                       cd.relationship_status, cd.conflict_note
                  FROM condition_doshas AS cd
                  JOIN doshas AS d ON d.dosha_id = cd.dosha_id
                 WHERE cd.condition_id = ?
                 ORDER BY cd.source_profile_id, d.dosha_name
                """,
                (condition_id,),
            )
        )

    def list_claims(self, condition_id: int) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT claim_id, claim_type, claim_summary, evidence_status,
                       source_profile_id, safety_relevance, claim_version
                  FROM knowledge_claims
                 WHERE condition_id = ?
                 ORDER BY claim_id
                """,
                (condition_id,),
            )
        )

    def condition_summary(self, name_or_alias: str) -> dict[str, Any] | None:
        condition = self.resolve_condition(name_or_alias)
        if condition is None:
            return None
        condition_id = int(condition["condition_id"])
        return {
            "condition": dict(condition),
            "dosha_relationships": [
                dict(row) for row in self.list_dosha_relationships(condition_id)
            ],
            "claims": [dict(row) for row in self.list_claims(condition_id)],
        }


class RuleRepository:
    """Load structured rules and their traceable claim/source support."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.knowledge = KnowledgeRepository(connection)

    def resolve_condition(self, name_or_alias: str) -> sqlite3.Row | None:
        return self.knowledge.resolve_condition(name_or_alias)

    def list_condition_rules(self, condition_id: int) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT kr.*, c.canonical_name
                  FROM knowledge_rules AS kr
                  JOIN conditions AS c ON c.condition_id = kr.condition_id
                 WHERE kr.condition_id = ?
                 ORDER BY kr.priority, kr.rule_key, kr.rule_version, kr.rule_id
                """,
                (condition_id,),
            )
        )

    def rule_evidence(self, rule_id: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT re.rule_id, re.claim_id, re.source_id,
                       re.validation_status AS link_validation_status,
                       kc.claim_id AS verified_claim_id,
                       es.source_id AS verified_source_id,
                       kc.condition_id AS claim_condition_id, kc.claim_type,
                       kc.evidence_status AS claim_evidence_status,
                       kc.claim_version, kc.is_active AS claim_is_active,
                       es.validation_status AS source_validation_status,
                       es.source_version, ce.source_locator,
                       ce.supports_complete_claim,
                       ce.validation_status AS claim_evidence_link_status
                  FROM rule_evidence AS re
                  LEFT JOIN knowledge_claims AS kc ON kc.claim_id = re.claim_id
                  LEFT JOIN evidence_sources AS es ON es.source_id = re.source_id
                  LEFT JOIN claim_evidence AS ce
                    ON ce.claim_id = re.claim_id AND ce.source_id = re.source_id
                 WHERE re.rule_id = ?
                 ORDER BY re.claim_id, re.source_id
                """,
                (rule_id,),
            )
        )

    def rule_validations(self, rule_id: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT validation_status, structural_validation_status,
                       validation_type, validator, validation_date, notes
                  FROM rule_validation
                 WHERE rule_id = ?
                 ORDER BY validation_date DESC, rule_validation_id DESC
                """,
                (rule_id,),
            )
        )
