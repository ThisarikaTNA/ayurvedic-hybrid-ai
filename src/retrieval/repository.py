"""Read-only, parameterized SQLite queries for Phase 8 retrieval."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

from knowledge_base.repository import normalize_lookup_text


USABLE_STATUSES: tuple[str, ...] = (
    "dataset_derived", "reference_checked", "expert_reviewed"
)


class RetrievalRepository:
    """Expose retrieval collections without invoking models or medical rules."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def resolve_condition(self, supplied_name: str) -> dict[str, Any]:
        """Resolve exact normalized canonical/alias matches and expose ambiguity."""

        normalized = normalize_lookup_text(supplied_name)
        rows = self.connection.execute(
            """
            SELECT c.condition_id, c.canonical_name, c.normalized_name,
                   c.provenance_status AS condition_status,
                   CASE WHEN c.normalized_name = ? THEN 'canonical' ELSE 'alias' END
                       AS match_type,
                   CASE WHEN c.normalized_name = ? THEN c.canonical_name ELSE a.alias_text END
                       AS matched_name,
                   a.alias_type, a.provenance_status AS alias_status
              FROM conditions AS c
              LEFT JOIN condition_aliases AS a
                ON a.condition_id = c.condition_id AND a.normalized_alias = ?
             WHERE (c.normalized_name = ? OR a.normalized_alias = ?)
               AND c.provenance_status IN ('dataset_derived', 'reference_checked', 'expert_reviewed')
               AND (a.alias_id IS NULL OR a.provenance_status <> 'inactive')
             ORDER BY CASE WHEN c.normalized_name = ? THEN 0 ELSE 1 END,
                      c.condition_id, a.alias_id
            """,
            (normalized, normalized, normalized, normalized, normalized, normalized),
        ).fetchall()
        distinct: dict[int, sqlite3.Row] = {}
        for row in rows:
            distinct.setdefault(int(row["condition_id"]), row)
        if not distinct:
            return {
                "status": "no_match", "normalized_input": normalized,
                "supplied_name": supplied_name, "matches": [],
            }
        if len(distinct) > 1:
            return {
                "status": "ambiguous", "normalized_input": normalized,
                "supplied_name": supplied_name,
                "matches": [
                    {
                        "condition_id": int(row["condition_id"]),
                        "canonical_name": str(row["canonical_name"]),
                        "match_type": str(row["match_type"]),
                        "matched_name": str(row["matched_name"]),
                    }
                    for row in distinct.values()
                ],
            }
        row = next(iter(distinct.values()))
        return {
            "status": "resolved",
            "normalized_input": normalized,
            "supplied_name": supplied_name,
            "condition_id": int(row["condition_id"]),
            "canonical_name": str(row["canonical_name"]),
            "match_type": str(row["match_type"]),
            "matched_name": str(row["matched_name"]),
            "alias_type": row["alias_type"],
            "mapping_status": (
                str(row["condition_status"])
                if row["match_type"] == "canonical"
                else str(row["alias_status"])
            ),
        }

    def list_profiles(self, condition_id: int) -> list[dict[str, Any]]:
        """Aggregate active dataset-derived claims by retained source profile."""

        claim_rows = self.connection.execute(
            """
            SELECT claim_id, claim_type, claim_summary, original_text,
                   normalized_text, source_profile_id, evidence_status,
                   claim_version, safety_relevance, limitations
              FROM knowledge_claims
             WHERE condition_id = ?
               AND source_profile_id IS NOT NULL
               AND TRIM(source_profile_id) <> ''
               AND evidence_status = 'dataset_derived'
               AND is_active = 1
             ORDER BY source_profile_id, claim_id
            """,
            (condition_id,),
        ).fetchall()
        claims_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in claim_rows:
            claims_by_profile[str(row["source_profile_id"])].append(dict(row))

        doshas_by_profile: dict[str, list[str]] = defaultdict(list)
        dosha_original: dict[str, str] = {}
        conflict_notes: dict[str, set[str]] = defaultdict(set)
        for row in self.connection.execute(
            """
            SELECT cd.source_profile_id, d.dosha_name, cd.original_dosha_text,
                   cd.relationship_status, cd.conflict_note, cd.claim_id
              FROM condition_doshas AS cd
              JOIN doshas AS d ON d.dosha_id = cd.dosha_id
              JOIN knowledge_claims AS kc ON kc.claim_id = cd.claim_id
             WHERE cd.condition_id = ?
               AND cd.relationship_status = 'dataset_derived'
               AND kc.is_active = 1
             ORDER BY cd.source_profile_id,
                      CASE d.dosha_name WHEN 'Vata' THEN 1 WHEN 'Pitta' THEN 2 ELSE 3 END
            """,
            (condition_id,),
        ):
            profile_id = str(row["source_profile_id"])
            doshas_by_profile[profile_id].append(str(row["dosha_name"]))
            dosha_original[profile_id] = str(row["original_dosha_text"])
            if row["conflict_note"]:
                conflict_notes[profile_id].add(str(row["conflict_note"]))

        profiles: list[dict[str, Any]] = []
        for profile_id in sorted(claims_by_profile):
            claims = claims_by_profile[profile_id]
            profiles.append(
                {
                    "record_id": profile_id,
                    "source_profile_id": profile_id,
                    "claims": claims,
                    "claim_ids": [str(claim["claim_id"]) for claim in claims],
                    "dataset_assigned_dosha_tags": doshas_by_profile.get(profile_id, []),
                    "original_dosha_text": dosha_original.get(profile_id),
                    "conflict_notes": sorted(conflict_notes.get(profile_id, set())),
                    "search_text": " ".join(
                        str(claim["original_text"] or claim["claim_summary"])
                        for claim in claims
                    ),
                    "evidence_status": "dataset_derived",
                    "lifecycle_status": "active",
                    "is_stale": False,
                    "limitations": (
                        "Dataset-derived knowledge profile; content and assigned Dosha "
                        "tags are not medically or clinically validated."
                    ),
                }
            )
        return profiles

    def list_reference_claims(self, condition_id: int) -> list[dict[str, Any]]:
        """Return only active claims with complete checked source support."""

        rows = self.connection.execute(
            """
            SELECT kc.claim_id AS record_id, kc.claim_id, kc.claim_type,
                   kc.claim_summary, kc.normalized_text, kc.evidence_status,
                   kc.claim_version, kc.safety_relevance, kc.limitations,
                   ce.source_id, ce.source_locator, ce.evidence_version,
                   es.publisher, es.page_title, es.url, es.validation_status,
                   es.source_version
              FROM knowledge_claims AS kc
              JOIN claim_evidence AS ce ON ce.claim_id = kc.claim_id
              JOIN evidence_sources AS es ON es.source_id = ce.source_id
             WHERE kc.condition_id = ?
               AND kc.evidence_status = 'reference_checked'
               AND kc.is_active = 1
               AND ce.validation_status = 'reference_checked'
               AND ce.supports_complete_claim = 1
               AND es.validation_status = 'reference_checked'
             ORDER BY kc.claim_id, ce.source_id
            """,
            (condition_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "source_profile_id": None,
                "supporting_claim_ids": [str(row["claim_id"])],
                "supporting_source_ids": [str(row["source_id"])],
                "lifecycle_status": "active",
                "is_stale": False,
                "search_text": str(row["claim_summary"]),
            }
            for row in rows
        ]

    def list_symptoms(self, condition_id: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT 'SYM-' || s.symptom_id AS record_id, s.symptom_id,
                   s.symptom_text, s.normalized_text, s.provenance_status,
                   s.record_version, cs.relationship_status,
                   NULLIF(cs.source_profile_id, '') AS source_profile_id,
                   cs.claim_id, kc.is_active AS claim_is_active,
                   (SELECT GROUP_CONCAT(ce.source_id)
                      FROM claim_evidence AS ce
                      JOIN evidence_sources AS es ON es.source_id = ce.source_id
                     WHERE ce.claim_id = cs.claim_id
                       AND ce.supports_complete_claim = 1
                       AND ce.validation_status = 'reference_checked'
                       AND es.validation_status = 'reference_checked') AS source_ids,
                   COALESCE(kc.limitations, 'Structured symptom relationship only.') AS limitations
              FROM condition_symptoms AS cs
              JOIN symptoms AS s ON s.symptom_id = cs.symptom_id
              LEFT JOIN knowledge_claims AS kc ON kc.claim_id = cs.claim_id
             WHERE cs.condition_id = ?
               AND s.provenance_status IN
                   ('dataset_derived', 'reference_checked', 'expert_reviewed')
               AND cs.relationship_status IN
                   ('dataset_derived', 'reference_checked', 'expert_reviewed')
               AND (kc.claim_id IS NULL OR kc.is_active = 1)
               AND (
                   cs.relationship_status <> 'reference_checked'
                   OR EXISTS (
                       SELECT 1
                         FROM claim_evidence AS ce
                         JOIN evidence_sources AS es ON es.source_id = ce.source_id
                        WHERE ce.claim_id = cs.claim_id
                          AND ce.supports_complete_claim = 1
                          AND ce.validation_status = 'reference_checked'
                          AND es.validation_status = 'reference_checked'
                   )
               )
             ORDER BY s.symptom_id, cs.condition_symptom_id
            """,
            (condition_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "evidence_status": str(row["relationship_status"]),
                "supporting_claim_ids": [str(row["claim_id"])] if row["claim_id"] else [],
                "supporting_source_ids": (
                    sorted(str(row["source_ids"]).split(",")) if row["source_ids"] else []
                ),
                "lifecycle_status": "active",
                "is_stale": False,
                "search_text": str(row["symptom_text"]),
            }
            for row in rows
        ]

    def list_dataset_recommendations(self, condition_id: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT 'REC-' || r.recommendation_id AS record_id,
                   r.recommendation_id, r.recommendation_text, r.normalized_text,
                   r.source_profile_id, r.provenance_status, r.recommendation_version,
                   r.claim_id, rc.category_code, rc.category_name,
                   kc.limitations
              FROM recommendations AS r
              JOIN recommendation_categories AS rc ON rc.category_id = r.category_id
              LEFT JOIN knowledge_claims AS kc ON kc.claim_id = r.claim_id
             WHERE r.condition_id = ?
               AND r.provenance_status = 'dataset_derived'
               AND r.is_generated = 0
               AND r.is_stale = 0
               AND (kc.claim_id IS NULL OR kc.is_active = 1)
             ORDER BY r.recommendation_id
            """,
            (condition_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "evidence_status": "dataset_derived",
                "supporting_claim_ids": [str(row["claim_id"])] if row["claim_id"] else [],
                "supporting_source_ids": [],
                "lifecycle_status": "active",
                "is_stale": False,
                "search_text": str(row["recommendation_text"]),
                "limitations": str(row["limitations"] or (
                    "Dataset-derived recommendation text; not medically verified and not a prescription."
                )),
            }
            for row in rows
        ]

    def list_sources(self, condition_id: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT es.source_id AS record_id, es.source_id, es.publisher,
                   es.page_title, es.url, es.access_date,
                   es.publication_or_review_date, es.jurisdiction,
                   es.source_type, es.source_version, es.validation_status,
                   es.reviewer, es.notes,
                   GROUP_CONCAT(DISTINCT kc.claim_id) AS claim_ids
              FROM evidence_sources AS es
              JOIN claim_evidence AS ce ON ce.source_id = es.source_id
              JOIN knowledge_claims AS kc ON kc.claim_id = ce.claim_id
             WHERE kc.condition_id = ?
               AND kc.is_active = 1
               AND kc.evidence_status = 'reference_checked'
               AND ce.validation_status = 'reference_checked'
               AND ce.supports_complete_claim = 1
               AND es.validation_status = 'reference_checked'
             GROUP BY es.source_id
             ORDER BY es.source_id
            """,
            (condition_id,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            claim_ids = sorted(str(row["claim_ids"]).split(","))
            results.append(
                {
                    **dict(row),
                    "evidence_status": "reference_checked",
                    "source_profile_id": None,
                    "supporting_claim_ids": claim_ids,
                    "supporting_source_ids": [str(row["source_id"])],
                    "lifecycle_status": "active",
                    "is_stale": False,
                    "search_text": " ".join(
                        [str(row["publisher"]), str(row["page_title"]), str(row["url"])]
                    ),
                    "limitations": (
                        "Source supports only the linked checked claims; source ranking is "
                        "not a measure of medical authority beyond that review."
                    ),
                }
            )
        return results

    def search_symptom_bm25(
        self, condition_id: int, fts_query: str
    ) -> dict[str, float]:
        rows = self.connection.execute(
            """
            SELECT 'SYM-' || s.symptom_id AS record_id,
                   bm25(symptom_search_fts) AS raw_bm25
              FROM symptom_search_fts
              JOIN symptoms AS s ON s.symptom_id = symptom_search_fts.rowid
              JOIN condition_symptoms AS cs ON cs.symptom_id = s.symptom_id
              LEFT JOIN knowledge_claims AS kc ON kc.claim_id = cs.claim_id
             WHERE symptom_search_fts MATCH ?
               AND cs.condition_id = ?
               AND s.provenance_status IN
                   ('dataset_derived', 'reference_checked', 'expert_reviewed')
               AND cs.relationship_status IN
                   ('dataset_derived', 'reference_checked', 'expert_reviewed')
               AND (kc.claim_id IS NULL OR kc.is_active = 1)
             ORDER BY raw_bm25, s.symptom_id
            """,
            (fts_query, condition_id),
        ).fetchall()
        return {str(row["record_id"]): float(row["raw_bm25"]) for row in rows}

    def search_recommendation_bm25(
        self, condition_id: int, fts_query: str
    ) -> dict[str, float]:
        rows = self.connection.execute(
            """
            SELECT 'REC-' || r.recommendation_id AS record_id,
                   bm25(recommendation_search_fts) AS raw_bm25
              FROM recommendation_search_fts
              JOIN recommendations AS r
                ON r.recommendation_id = recommendation_search_fts.rowid
              LEFT JOIN knowledge_claims AS kc ON kc.claim_id = r.claim_id
             WHERE recommendation_search_fts MATCH ?
               AND r.condition_id = ?
               AND r.is_stale = 0
               AND r.is_generated = 0
               AND r.provenance_status IN
                   ('dataset_derived', 'reference_checked', 'expert_reviewed')
               AND (kc.claim_id IS NULL OR kc.is_active = 1)
             ORDER BY raw_bm25, r.recommendation_id
            """,
            (fts_query, condition_id),
        ).fetchall()
        return {str(row["record_id"]): float(row["raw_bm25"]) for row in rows}

    def profile_bm25(
        self,
        condition_id: int,
        symptom_scores: dict[str, float],
        recommendation_scores: dict[str, float],
    ) -> dict[str, float]:
        """Map matched FTS records back to their dataset source profiles."""

        scores: dict[str, list[float]] = defaultdict(list)
        if symptom_scores:
            for row in self.connection.execute(
                """
                SELECT 'SYM-' || symptom_id AS record_id, source_profile_id
                  FROM condition_symptoms
                 WHERE condition_id = ? AND source_profile_id <> ''
                """,
                (condition_id,),
            ):
                record_id = str(row["record_id"])
                if record_id in symptom_scores:
                    scores[str(row["source_profile_id"])].append(symptom_scores[record_id])
        if recommendation_scores:
            for row in self.connection.execute(
                """
                SELECT 'REC-' || recommendation_id AS record_id, source_profile_id
                  FROM recommendations
                 WHERE condition_id = ? AND source_profile_id <> ''
                   AND provenance_status='dataset_derived' AND is_stale=0
                """,
                (condition_id,),
            ):
                record_id = str(row["record_id"])
                if record_id in recommendation_scores:
                    scores[str(row["source_profile_id"])].append(
                        recommendation_scores[record_id]
                    )
        return {profile_id: min(values) for profile_id, values in scores.items()}

    def fts5_available(self) -> bool:
        tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        return {
            "symptom_search_fts", "recommendation_search_fts"
        }.issubset(tables)
