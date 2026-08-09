"""Deterministic, explainable, condition-scoped Phase 8 retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any, Mapping, Sequence

from retrieval.query_schema import (
    RetrievalInputError,
    ValidatedRetrievalQuery,
    build_safe_fts_query,
    load_retrieval_config,
    tokenize_query,
    validate_retrieval_query,
)
from retrieval.repository import RetrievalRepository


DISCLAIMER = (
    "Educational research prototype only; retrieval does not diagnose, infer a true "
    "Dosha, prescribe, fire medical rules, or replace professional healthcare."
)
DOSHA_NOTE = (
    "Caller-supplied Dosha tags are retrieval signals only and are not inferred, "
    "confirmed, or clinically validated."
)
LEXICAL_NOTE = (
    "Lexical relevance measures text similarity, not evidence quality or medical correctness."
)
DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|milligrams?|g|grams?|mcg|µg|ml|"
    r"millilit(?:er|re)s?|tablets?|capsules?|drops?)\b",
    flags=re.IGNORECASE,
)


def _run_id(request: Any, config_version: str) -> str:
    payload = json.dumps(
        {"request": request, "config_version": config_version},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "RET-" + hashlib.sha256(payload).hexdigest()[:16].upper()


def _text_terms(text: str) -> set[str]:
    return set(tokenize_query(text, maximum_terms=10000))


def _matched_terms(query_terms: Sequence[str], text: str) -> list[str]:
    document_terms = _text_terms(text)
    return [term for term in query_terms if term in document_terms]


def _jaccard(left: Sequence[str], right: Sequence[str]) -> tuple[float, list[str], list[str]]:
    left_set = set(left)
    right_set = set(right)
    intersection = sorted(left_set & right_set)
    union = sorted(left_set | right_set)
    return (
        (len(intersection) / len(union)) if union else 0.0,
        intersection,
        union,
    )


def _bm25_normalized(raw_by_id: Mapping[str, float]) -> dict[str, float]:
    """Normalize smaller-is-better BM25 scores within one result collection."""

    if not raw_by_id:
        return {}
    best = min(raw_by_id.values())
    worst = max(raw_by_id.values())
    if abs(worst - best) < 1e-15:
        return {record_id: 1.0 for record_id in raw_by_id}
    return {
        record_id: (worst - raw) / (worst - best)
        for record_id, raw in raw_by_id.items()
    }


def _safe_public_content(item: Mapping[str, Any], result_type: str) -> dict[str, Any]:
    if result_type == "dataset_knowledge_profile":
        return {
            "profile_claims": [
                {
                    "claim_id": claim["claim_id"],
                    "claim_type": claim["claim_type"],
                    "claim_summary": claim["claim_summary"],
                    "original_text": claim["original_text"],
                    "evidence_status": claim["evidence_status"],
                }
                for claim in item["claims"]
            ],
            "dataset_assigned_dosha_tags": item["dataset_assigned_dosha_tags"],
            "original_dosha_text": item["original_dosha_text"],
            "conflict_notes": item["conflict_notes"],
        }
    if result_type == "reference_checked_claim":
        return {
            "claim_type": item["claim_type"],
            "claim_summary": item["claim_summary"],
            "safety_relevance": item["safety_relevance"],
            "source_locator": item["source_locator"],
            "source_url": item["url"],
        }
    if result_type == "symptom":
        return {"symptom_text": item["symptom_text"]}
    if result_type == "dataset_recommendation":
        return {
            "category": item["category_name"],
            "recommendation_text": item["recommendation_text"],
        }
    if result_type == "evidence_source":
        return {
            "publisher": item["publisher"],
            "page_title": item["page_title"],
            "url": item["url"],
            "access_date": item["access_date"],
            "jurisdiction": item["jurisdiction"],
        }
    return {
        "source_profile_id": item["source_profile_id"],
        "provenance_statement": (
            "Dataset-derived knowledge-profile provenance; not a historical patient record."
        ),
    }


class ProfileRetriever:
    """Retrieve SQLite knowledge profiles and claims without diagnostic inference."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.connection = connection
        self.repository = RetrievalRepository(connection)
        self.config = dict(config or load_retrieval_config())

    def _base_response(
        self,
        run_id: str,
        status: str,
        request: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "retrieval_run_id": run_id,
            "status": status,
            "request": request,
            "condition_context_source": "caller_supplied_not_inferred",
            "resolved_condition": None,
            "matched_alias": None,
            "collections": {
                "profiles": [], "claims": [], "symptoms": [],
                "recommendations": [], "sources": [],
            },
            "collection_counts": {
                "profiles": 0, "claims": 0, "symptoms": 0,
                "recommendations": 0, "sources": 0,
            },
            "disclosures": [],
            "persistence": "disabled_non_persistent",
            "ml_model_loaded_or_invoked": False,
            "rule_engine_loaded_or_invoked": False,
            "diagnosis_performed": False,
            "dosha_inference_performed": False,
            "lexical_relevance_note": LEXICAL_NOTE,
            "caller_supplied_dosha_note": DOSHA_NOTE,
            "disclaimer": DISCLAIMER,
        }

    def _score_collection(
        self,
        *,
        items: list[dict[str, Any]],
        query: ValidatedRetrievalQuery,
        raw_bm25: Mapping[str, float] | None,
        result_type: str,
        condition_resolution: Mapping[str, Any],
        use_dosha: bool = False,
    ) -> list[dict[str, Any]]:
        raw_scores = dict(raw_bm25 or {})
        normalized_bm25 = _bm25_normalized(raw_scores)
        query_supplied = query.free_text is not None
        dosha_supplied = bool(query.caller_supplied_dosha_tags)
        lexical_config = self.config["lexical_scoring"]
        scored: list[dict[str, Any]] = []

        for item in items:
            item_result_type = (
                "dataset_provenance"
                if result_type == "evidence_source"
                and str(item["record_id"]).startswith("PROV-")
                else result_type
            )
            public_content = _safe_public_content(item, item_result_type)
            if DOSAGE_PATTERN.search(json.dumps(public_content, ensure_ascii=False)):
                continue
            record_id = str(item["record_id"])
            matched = _matched_terms(query.query_terms, str(item.get("search_text", "")))
            coverage = (
                len(matched) / len(query.query_terms) if query.query_terms else 0.0
            )
            raw = raw_scores.get(record_id)
            bm25_score = normalized_bm25.get(record_id, 0.0)
            if raw is None:
                lexical_score = coverage
                lexical_method = "safe_token_coverage"
            else:
                lexical_score = (
                    float(lexical_config["bm25_normalized_weight"]) * bm25_score
                    + float(lexical_config["matched_term_coverage_weight"]) * coverage
                )
                lexical_method = "fts5_bm25_plus_safe_token_coverage"

            dosha_score = 0.0
            intersection: list[str] = []
            union: list[str] = []
            dataset_tags = list(item.get("dataset_assigned_dosha_tags", []))
            if use_dosha and dosha_supplied:
                dosha_score, intersection, union = _jaccard(
                    query.caller_supplied_dosha_tags, dataset_tags
                )

            if use_dosha:
                if query_supplied and dosha_supplied:
                    mode = "query_and_dosha"
                elif query_supplied:
                    mode = "query_only"
                elif dosha_supplied:
                    mode = "dosha_only"
                else:
                    mode = "no_optional_signals"
                weights = self.config["profile_ranking"][mode]
                final_score = (
                    float(weights["lexical_relevance_weight"]) * lexical_score
                    + float(weights["caller_supplied_dosha_overlap_weight"]) * dosha_score
                )
            else:
                final_score = lexical_score if query_supplied else 0.0

            inclusion_signal = (
                not query_supplied and (not use_dosha or not dosha_supplied)
            ) or bool(matched) or (use_dosha and dosha_supplied and dosha_score > 0)
            if not inclusion_signal:
                continue

            reasons: list[str] = [
                "Record is active, non-stale, and restricted to the resolved condition."
            ]
            if matched:
                reasons.append("Safe lexical terms matched: " + ", ".join(matched) + ".")
            if use_dosha and dosha_supplied:
                reasons.append(
                    "Caller-supplied Dosha overlap was used only as a ranking signal."
                )
            if not query_supplied and not dosha_supplied:
                reasons.append("No optional signals were supplied; stable identifier order applies.")

            scored.append(
                {
                    "retrieval_run_id": _run_id(query.as_dict(), self.config["config_version"]),
                    "resolved_condition": condition_resolution["canonical_name"],
                    "matched_alias": {
                        "supplied": condition_resolution["supplied_name"],
                        "matched_name": condition_resolution["matched_name"],
                        "match_type": condition_resolution["match_type"],
                        "mapping_status": condition_resolution["mapping_status"],
                    },
                    "record_id": record_id,
                    "result_type": item_result_type,
                    "rank": 0,
                    "component_scores": {
                        "lexical_method": lexical_method,
                        "raw_bm25_smaller_is_better": raw,
                        "bm25_normalized": round(bm25_score, 8),
                        "matched_term_coverage": round(coverage, 8),
                        "lexical_relevance": round(lexical_score, 8),
                        "caller_supplied_dosha_jaccard": round(dosha_score, 8),
                        "evidence_quality_score": None,
                    },
                    "final_retrieval_score": round(final_score, 8),
                    "matched_query_terms": matched,
                    "dosha_overlap_details": {
                        "used": use_dosha and dosha_supplied,
                        "caller_supplied_tags": list(query.caller_supplied_dosha_tags),
                        "dataset_assigned_tags": dataset_tags,
                        "intersection": intersection,
                        "union": union,
                        "jaccard_score": round(dosha_score, 8),
                        "interpretation": "retrieval_signal_not_inference",
                    },
                    "evidence_status": item["evidence_status"],
                    "lifecycle_status": item["lifecycle_status"],
                    "is_stale": bool(item["is_stale"]),
                    "source_profile_identifier": item.get("source_profile_id"),
                    "supporting_claim_ids": list(
                        item.get("supporting_claim_ids", item.get("claim_ids", []))
                    ),
                    "supporting_source_ids": list(item.get("supporting_source_ids", [])),
                    "reason_included": " ".join(reasons),
                    "limitations": item["limitations"],
                    "content": public_content,
                    "lexical_relevance_note": LEXICAL_NOTE,
                    "caller_supplied_dosha_note": DOSHA_NOTE,
                    "disclaimer": DISCLAIMER,
                }
            )

        scored.sort(
            key=lambda result: (
                -float(result["final_retrieval_score"]),
                -float(result["component_scores"]["lexical_relevance"]),
                (
                    float(result["component_scores"]["raw_bm25_smaller_is_better"])
                    if result["component_scores"]["raw_bm25_smaller_is_better"] is not None
                    else float("inf")
                ),
                str(result["record_id"]),
            )
        )
        selected = scored[: query.top_k]
        for rank, result in enumerate(selected, start=1):
            result["rank"] = rank
        return selected

    @staticmethod
    def _insomnia_disclosure(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        combinations = {
            str(profile["record_id"]): list(profile["dataset_assigned_dosha_tags"])
            for profile in profiles
        }
        unique = {tuple(tags) for tags in combinations.values()}
        if len(unique) < 2:
            return None
        return {
            "type": "dataset_association_disagreement",
            "condition": "Insomnia",
            "profile_associations": combinations,
            "resolution": "preserved_not_resolved",
            "explanation": (
                "The retained Insomnia profiles contain Vata and Vata/Pitta dataset "
                "assignments. Ranking may reflect caller-supplied overlap, but no voting, "
                "clinical interpretation, or Dosha inference is performed."
            ),
        }

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Validate, resolve, retrieve, rank, and trace one non-persistent request."""

        initial_run_id = _run_id(request, self.config["config_version"])
        try:
            query = validate_retrieval_query(request, self.config)
        except RetrievalInputError as error:
            response = self._base_response(
                initial_run_id,
                "invalid_request",
                {
                    "provided_fields": (
                        sorted(str(key) for key in request)
                        if isinstance(request, Mapping) else []
                    )
                },
            )
            response["reason"] = str(error)
            return response

        run_id = _run_id(query.as_dict(), self.config["config_version"])
        response = self._base_response(run_id, "pending", query.as_dict())
        if query.condition_context is None:
            response["status"] = "clarification_required"
            response["reason"] = (
                "A canonical condition name or approved alias is required; the condition "
                "will not be inferred from query text."
            )
            return response

        resolution = self.repository.resolve_condition(query.condition_context)
        response["condition_resolution"] = resolution
        if resolution["status"] == "no_match":
            response["status"] = "no_match"
            response["reason"] = "Unknown condition or alias; no fallback condition was used."
            return response
        if resolution["status"] == "ambiguous":
            response["status"] = "clarification_required"
            response["reason"] = "Condition input matched more than one canonical condition."
            return response

        condition_id = int(resolution["condition_id"])
        response["resolved_condition"] = resolution["canonical_name"]
        response["matched_alias"] = {
            "supplied": resolution["supplied_name"],
            "matched_name": resolution["matched_name"],
            "match_type": resolution["match_type"],
            "mapping_status": resolution["mapping_status"],
        }
        profiles = self.repository.list_profiles(condition_id)
        claims = self.repository.list_reference_claims(condition_id)
        symptoms = self.repository.list_symptoms(condition_id)
        recommendations = self.repository.list_dataset_recommendations(condition_id)
        sources = self.repository.list_sources(condition_id)
        provenance_records = [
            {
                "record_id": f"PROV-{profile['record_id']}",
                "source_profile_id": profile["source_profile_id"],
                "supporting_claim_ids": profile["claim_ids"],
                "supporting_source_ids": [],
                "search_text": profile["search_text"],
                "evidence_status": "dataset_derived",
                "lifecycle_status": "active",
                "is_stale": False,
                "limitations": profile["limitations"],
            }
            for profile in profiles
        ]
        sources.extend(provenance_records)

        fts_query = build_safe_fts_query(query.query_terms)
        symptom_bm25: dict[str, float] = {}
        recommendation_bm25: dict[str, float] = {}
        if fts_query and self.repository.fts5_available():
            symptom_bm25 = self.repository.search_symptom_bm25(condition_id, fts_query)
            recommendation_bm25 = self.repository.search_recommendation_bm25(
                condition_id, fts_query
            )
        profile_bm25 = self.repository.profile_bm25(
            condition_id, symptom_bm25, recommendation_bm25
        )

        if "profiles" in query.categories:
            response["collections"]["profiles"] = self._score_collection(
                items=profiles, query=query, raw_bm25=profile_bm25,
                result_type="dataset_knowledge_profile",
                condition_resolution=resolution, use_dosha=True,
            )
        if "claims" in query.categories:
            response["collections"]["claims"] = self._score_collection(
                items=claims, query=query, raw_bm25=None,
                result_type="reference_checked_claim",
                condition_resolution=resolution,
            )
        if "symptoms" in query.categories:
            response["collections"]["symptoms"] = self._score_collection(
                items=symptoms, query=query, raw_bm25=symptom_bm25,
                result_type="symptom", condition_resolution=resolution,
            )
        if "recommendations" in query.categories:
            response["collections"]["recommendations"] = self._score_collection(
                items=recommendations, query=query, raw_bm25=recommendation_bm25,
                result_type="dataset_recommendation", condition_resolution=resolution,
            )
        if "sources" in query.categories:
            response["collections"]["sources"] = self._score_collection(
                items=sources, query=query, raw_bm25=None,
                result_type="evidence_source",
                condition_resolution=resolution,
            )
        response["collection_counts"] = {
            name: len(results) for name, results in response["collections"].items()
        }
        if resolution["canonical_name"] == "Insomnia":
            disclosure = self._insomnia_disclosure(profiles)
            if disclosure:
                response["disclosures"].append(disclosure)
        total_results = sum(response["collection_counts"].values())
        response["status"] = "success" if total_results else "no_match"
        if not total_results:
            response["reason"] = (
                "No active, non-stale record matched the supplied lexical or Dosha "
                "retrieval signals within the resolved condition."
            )
        response["fts5"] = {
            "available": self.repository.fts5_available(),
            "safe_query": fts_query,
            "parameterized": True,
        }
        return response
