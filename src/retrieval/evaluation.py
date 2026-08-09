"""Transparent, approved-record evaluation cases and Phase 8 retrieval metrics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Mapping

from retrieval.profile_retriever import ProfileRetriever


EVALUATION_QUERIES: tuple[dict[str, Any], ...] = (
    {
        "query_id": "EVAL-CANON-ACNE",
        "query_type": "canonical_name",
        "request": {"condition": "Acne", "categories": ["profiles"], "top_k": 3},
        "expected_status": "success",
        "expected_condition": "Acne",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0016"],
        "relevance_assignment": "Exact condition scope and the sole approved Acne profile.",
    },
    {
        "query_id": "EVAL-CANON-COLD",
        "query_type": "canonical_name",
        "request": {"condition": "Common Cold", "categories": ["profiles"], "top_k": 3},
        "expected_status": "success",
        "expected_condition": "Common Cold",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0006"],
        "relevance_assignment": "Exact condition scope and the sole approved Common Cold profile.",
    },
    {
        "query_id": "EVAL-CANON-GERD",
        "query_type": "canonical_name",
        "request": {
            "condition": "Gastroesophageal Reflux Disease",
            "categories": ["profiles"], "top_k": 3,
        },
        "expected_status": "success",
        "expected_condition": "Gastroesophageal Reflux Disease",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0114"],
        "relevance_assignment": "Exact condition scope and the sole approved reflux profile.",
    },
    {
        "query_id": "EVAL-CANON-OA",
        "query_type": "canonical_name",
        "request": {"condition": "Osteoarthritis", "categories": ["profiles"], "top_k": 3},
        "expected_status": "success",
        "expected_condition": "Osteoarthritis",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0226"],
        "relevance_assignment": "Exact condition scope and the sole approved Osteoarthritis profile.",
    },
    {
        "query_id": "EVAL-CANON-INSOMNIA",
        "query_type": "canonical_name",
        "request": {"condition": "Insomnia", "categories": ["profiles"], "top_k": 3},
        "expected_status": "success",
        "expected_condition": "Insomnia",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0010", "kp_0150"],
        "relevance_assignment": (
            "Both approved Insomnia profiles are relevant and must remain visible in stable ID order."
        ),
    },
    {
        "query_id": "EVAL-ALIAS-GERD",
        "query_type": "alias",
        "request": {"condition": "GERD", "categories": ["profiles"], "top_k": 3},
        "expected_status": "success",
        "expected_condition": "Gastroesophageal Reflux Disease",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0114"],
        "relevance_assignment": "GERD is an approved abbreviation for the reflux condition.",
    },
    {
        "query_id": "EVAL-ALIAS-GORD",
        "query_type": "alias",
        "request": {"condition": "GORD", "categories": ["profiles"], "top_k": 3},
        "expected_status": "success",
        "expected_condition": "Gastroesophageal Reflux Disease",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0114"],
        "relevance_assignment": "GORD is an approved abbreviation for the reflux condition.",
    },
    {
        "query_id": "EVAL-ALIAS-ACID-REFLUX",
        "query_type": "alias",
        "request": {"condition": "Acid reflux", "categories": ["profiles"], "top_k": 3},
        "expected_status": "success",
        "expected_condition": "Gastroesophageal Reflux Disease",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0114"],
        "relevance_assignment": "Acid reflux is an approved common-name alias.",
    },
    {
        "query_id": "EVAL-ALIAS-GASTRO-OESOPHAGEAL",
        "query_type": "alias",
        "request": {
            "condition": "Gastro-oesophageal Reflux Disease",
            "categories": ["profiles"], "top_k": 3,
        },
        "expected_status": "success",
        "expected_condition": "Gastroesophageal Reflux Disease",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0114"],
        "relevance_assignment": "Approved British spelling and punctuation variant.",
    },
    {
        "query_id": "EVAL-TEXT-ACNE-SELFCARE",
        "query_type": "within_condition_text",
        "request": {
            "condition": "Acne", "free_text": "picking squeezing scarring",
            "categories": ["claims"], "top_k": 3,
        },
        "expected_status": "success",
        "expected_condition": "Acne",
        "target_collection": "claims",
        "expected_record_ids": ["KC-ACNE-REF-SELF-001"],
        "relevance_assignment": "Terms are stated in the checked Acne self-care claim.",
    },
    {
        "query_id": "EVAL-TEXT-COLD-REFERRAL",
        "query_type": "within_condition_text",
        "request": {
            "condition": "Common Cold", "free_text": "shortness breath chest pain",
            "categories": ["claims"], "top_k": 3,
        },
        "expected_status": "success",
        "expected_condition": "Common Cold",
        "target_collection": "claims",
        "expected_record_ids": ["KC-COLD-REF-SAFE-001"],
        "relevance_assignment": "Terms are stated in the checked Common Cold referral-information claim.",
    },
    {
        "query_id": "EVAL-TEXT-INSOMNIA-SELFCARE",
        "query_type": "within_condition_text",
        "request": {
            "condition": "Insomnia", "free_text": "consistent wake time dark quiet",
            "categories": ["claims"], "top_k": 3,
        },
        "expected_status": "success",
        "expected_condition": "Insomnia",
        "target_collection": "claims",
        "expected_record_ids": ["KC-INSOMNIA-REF-SELF-001"],
        "relevance_assignment": "Terms are stated in the checked Insomnia self-care claim.",
    },
    {
        "query_id": "EVAL-FTS-COLD-SYMPTOMS",
        "query_type": "fts5_lexical",
        "request": {
            "condition": "Common Cold", "free_text": "blocked runny nose cough",
            "categories": ["symptoms"], "top_k": 3,
        },
        "expected_status": "success",
        "expected_condition": "Common Cold",
        "target_collection": "symptoms",
        "expected_record_ids": ["SYM-20", "SYM-1", "SYM-24"],
        "expected_first_record_id": "SYM-20",
        "relevance_assignment": (
            "All three stored symptom records contain supplied terms; the checked phrase "
            "'blocked or runny nose' covers three terms and is expected first by BM25/coverage."
        ),
    },
    {
        "query_id": "EVAL-DOSHA-INSOMNIA-PITTA",
        "query_type": "caller_dosha_ranking",
        "request": {
            "condition": "Insomnia", "caller_supplied_dosha_tags": ["Pitta"],
            "categories": ["profiles"], "top_k": 2,
        },
        "expected_status": "success",
        "expected_condition": "Insomnia",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0010"],
        "expected_first_record_id": "kp_0010",
        "relevance_assignment": (
            "Only kp_0010 contains the caller-supplied Pitta tag; this is a retrieval signal, not validation."
        ),
    },
    {
        "query_id": "EVAL-CONFLICT-INSOMNIA-VATA",
        "query_type": "insomnia_disagreement",
        "request": {
            "condition": "Insomnia", "caller_supplied_dosha_tags": ["Vata"],
            "categories": ["profiles"], "top_k": 2,
        },
        "expected_status": "success",
        "expected_condition": "Insomnia",
        "target_collection": "profiles",
        "expected_record_ids": ["kp_0150", "kp_0010"],
        "expected_first_record_id": "kp_0150",
        "relevance_assignment": (
            "Both profiles overlap Vata; exact Vata-set overlap ranks kp_0150 first while both remain disclosed."
        ),
    },
    {
        "query_id": "EVAL-NO-RESULT",
        "query_type": "no_result",
        "request": {
            "condition": "Acne", "free_text": "quantum telescope nebula",
            "categories": ["profiles", "claims", "symptoms", "recommendations"],
            "top_k": 3,
        },
        "expected_status": "no_match",
        "expected_condition": "Acne",
        "target_collection": "profiles",
        "expected_record_ids": [],
        "relevance_assignment": "Terms are absent from the condition-scoped approved records.",
    },
    {
        "query_id": "EVAL-UNKNOWN-CONDITION",
        "query_type": "unknown_condition",
        "request": {"condition": "Unlisted Condition", "free_text": "pain", "top_k": 3},
        "expected_status": "no_match",
        "expected_condition": None,
        "target_collection": None,
        "expected_record_ids": [],
        "relevance_assignment": "No approved canonical condition or alias has this normalized name.",
    },
    {
        "query_id": "EVAL-MISSING-CONDITION",
        "query_type": "missing_condition",
        "request": {"free_text": "heartburn", "top_k": 3},
        "expected_status": "clarification_required",
        "expected_condition": None,
        "target_collection": None,
        "expected_record_ids": [],
        "relevance_assignment": "Condition context is required and must not be inferred from text.",
    },
)


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def evaluate_retrieval(retriever: ProfileRetriever) -> dict[str, Any]:
    """Execute transparent cases and calculate ranking/resolution metrics."""

    cases: list[dict[str, Any]] = []
    for specification in EVALUATION_QUERIES:
        result = retriever.retrieve(specification["request"])
        collection = specification["target_collection"]
        retrieved_ids = (
            [item["record_id"] for item in result["collections"][collection]]
            if collection else []
        )
        expected_ids = list(specification["expected_record_ids"])
        positions = [
            retrieved_ids.index(record_id) + 1
            for record_id in expected_ids if record_id in retrieved_ids
        ]
        hit = bool(positions) if expected_ids else not retrieved_ids
        recall = (
            len(set(retrieved_ids) & set(expected_ids)) / len(expected_ids)
            if expected_ids else float(not retrieved_ids)
        )
        reciprocal_rank = (1.0 / min(positions)) if positions else 0.0
        exact = set(retrieved_ids) == set(expected_ids)
        first_expected = specification.get("expected_first_record_id")
        first_rank_correct = first_expected is None or (
            bool(retrieved_ids) and retrieved_ids[0] == first_expected
        )
        status_correct = result["status"] == specification["expected_status"]
        condition_correct = result["resolved_condition"] == specification["expected_condition"]
        cases.append(
            {
                **specification,
                "actual_status": result["status"],
                "actual_condition": result["resolved_condition"],
                "retrieved_record_ids": retrieved_ids,
                "hit_at_k": float(hit),
                "recall_at_k": round(recall, 8),
                "reciprocal_rank": round(reciprocal_rank, 8),
                "exact_expected_record_retrieval": exact,
                "first_rank_correct": first_rank_correct,
                "status_correct": status_correct,
                "condition_resolution_correct": condition_correct,
                "passed": (
                    status_correct and condition_correct and exact and first_rank_correct
                ),
                "retrieval_run_id": result["retrieval_run_id"],
            }
        )

    ranking_cases = [case for case in cases if case["target_collection"]]
    resolution_cases = [
        case for case in cases
        if case["query_type"] in {"canonical_name", "alias", "unknown_condition", "missing_condition"}
    ]
    overall = {
        "evaluation_query_count": len(cases),
        "ranking_query_count": len(ranking_cases),
        "alias_resolution_accuracy": _mean([
            float(case["condition_resolution_correct"] and case["status_correct"])
            for case in cases if case["query_type"] == "alias"
        ]),
        "condition_resolution_accuracy": _mean([
            float(case["condition_resolution_correct"] and case["status_correct"])
            for case in resolution_cases
        ]),
        "hit_at_k": _mean([case["hit_at_k"] for case in ranking_cases]),
        "recall_at_k": _mean([case["recall_at_k"] for case in ranking_cases]),
        "mean_reciprocal_rank": _mean([
            case["reciprocal_rank"] for case in ranking_cases
            if case["expected_record_ids"]
        ]),
        "exact_expected_record_retrieval": _mean([
            float(case["exact_expected_record_retrieval"]) for case in ranking_cases
        ]),
        "passed_query_count": sum(case["passed"] for case in cases),
        "all_queries_passed": all(case["passed"] for case in cases),
    }

    by_type_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_type_values[case["query_type"]].append(case)
    by_type: dict[str, dict[str, Any]] = {}
    for query_type, type_cases in sorted(by_type_values.items()):
        type_ranking = [case for case in type_cases if case["target_collection"]]
        by_type[query_type] = {
            "query_count": len(type_cases),
            "pass_rate": _mean([float(case["passed"]) for case in type_cases]),
            "hit_at_k": _mean([case["hit_at_k"] for case in type_ranking]),
            "recall_at_k": _mean([case["recall_at_k"] for case in type_ranking]),
            "mean_reciprocal_rank": _mean([
                case["reciprocal_rank"] for case in type_ranking
                if case["expected_record_ids"]
            ]),
            "exact_expected_record_retrieval": _mean([
                float(case["exact_expected_record_retrieval"])
                for case in type_ranking
            ]),
        }
    return {"metrics": {"overall": overall, "by_query_type": by_type}, "cases": cases}
