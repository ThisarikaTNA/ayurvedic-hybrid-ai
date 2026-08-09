"""Provenance-separated composition for Phase 9 hybrid results."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PROTOTYPE_DISCLAIMER = (
    "Educational research prototype only. The condition was selected by the caller and "
    "was not diagnosed from symptoms. Model labels predict dataset-assigned Dosha tags, "
    "not a medically true Dosha. Results are not prescriptions, exact dosage instructions, "
    "claims of clinical validation, or a replacement for professional healthcare."
)


def empty_result(*, run_id: str, pipeline_version: str) -> dict[str, Any]:
    """Return all required result sections even for early failures."""

    return {
        "result_type": "hybrid_decision_support_result",
        "run_id": run_id,
        "pipeline_version": pipeline_version,
        "orchestration_state": "blocked_component_failure",
        "condition_context": {},
        "safety_gate_result": {},
        "rule_trace": [],
        "model_prediction": {},
        "reference_checked_information": [],
        "dataset_derived_profiles": [],
        "dataset_derived_recommendations": [],
        "retrieval_trace": {},
        "agreements_and_disagreements": [],
        "suppressed_items": [],
        "limitations": [],
        "prototype_disclaimer": PROTOTYPE_DISCLAIMER,
        "component_versions": {},
        "component_integrity": {},
        "invocation_trace": [],
    }


def compose(
    *,
    result: dict[str, Any],
    safety: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    retrieval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Populate required sections without creating an artificial hybrid score."""

    if safety:
        result["orchestration_state"] = safety["state"]
        result["safety_gate_result"] = {
            key: value for key, value in safety.items() if key != "rule_engine_result"
        }
        engine = safety.get("rule_engine_result", {})
        result["rule_trace"] = [
            {**trace, "provenance": trace.get("evidence_status", "unknown")}
            for trace in engine.get("explanation_trace", [])
        ]
        trace_by_rule = {
            trace.get("rule_id"): trace for trace in result["rule_trace"]
        }
        result["suppressed_items"].extend(
            _action_with_trace(item, trace_by_rule.get(item.get("rule_id")))
            for item in engine.get("suppressed_actions", [])
        )
        for action in engine.get("candidate_actions", []):
            if action.get("type") in {"professional_referral", "general_recommendation"}:
                result["reference_checked_information"].append(
                    {
                        **_action_with_trace(
                            action, trace_by_rule.get(action.get("rule_id"))
                        ),
                        "provenance": "reference_checked",
                        "inclusion_reason": "active Phase 7 rule action with checked support",
                    }
                )
    if model:
        result["model_prediction"] = dict(model)
    if retrieval:
        collections = retrieval.get("collections", {})
        result["reference_checked_information"].extend(
            _with_provenance(collections.get("claims", []), "reference_checked")
        )
        result["dataset_derived_profiles"] = _with_provenance(
            collections.get("profiles", []), "dataset_derived"
        )
        result["dataset_derived_recommendations"] = _with_provenance(
            collections.get("recommendations", []), "dataset_derived"
        )
        result["retrieval_trace"] = {
            "status": retrieval.get("status"),
            "run_id": retrieval.get("retrieval_run_id"),
            "matched_alias": retrieval.get("matched_alias"),
            "collection_counts": retrieval.get("collection_counts", {}),
            "fts5": retrieval.get("fts5", {}),
            "adapter_trace": retrieval.get("adapter_trace", {}),
            "disclosures": retrieval.get("disclosures", []),
            "score_interpretation": (
                "Retrieval scores represent query/text relevance and optional set overlap, "
                "not evidence strength or medical correctness."
            ),
            "supporting_symptoms": _from_evidence(collections.get("symptoms", [])),
            "source_and_provenance_records": _from_evidence(
                collections.get("sources", [])
            ),
        }
        result["agreements_and_disagreements"] = compare_dosha_outputs(
            model.get("model_predicted_dosha_labels", []) if model else [],
            result["dataset_derived_profiles"],
            result.get("condition_context", {}).get("canonical_name"),
        )
    return result


def _with_provenance(items: Sequence[Mapping[str, Any]], provenance: str) -> list[dict[str, Any]]:
    return [
        {
            **dict(item),
            "provenance": provenance,
            "authority_note": (
                "Reference checking is claim-specific and is not expert review or clinical validation."
                if provenance == "reference_checked"
                else "Dataset demonstration only; not verified Ayurvedic medical advice."
            ),
        }
        for item in items
    ]


def _from_evidence(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**dict(item), "provenance": item.get("evidence_status", "unknown")}
        for item in items
    ]


def _action_with_trace(
    action: Mapping[str, Any], trace: Mapping[str, Any] | None
) -> dict[str, Any]:
    supporting = trace or {}
    return {
        **dict(action),
        "evidence_status": supporting.get("evidence_status", "reference_checked"),
        "provenance": supporting.get("evidence_status", "reference_checked"),
        "supporting_claim_ids": list(
            supporting.get("supporting_claim_ids", [])
        ),
        "supporting_source_ids": list(
            supporting.get("supporting_source_ids", [])
        ),
        "source_locator": supporting.get("source_locator"),
        "limitations": supporting.get("limitations"),
    }


def compare_dosha_outputs(
    predicted_labels: Sequence[str],
    profiles: Sequence[Mapping[str, Any]],
    condition: str | None,
) -> list[dict[str, Any]]:
    """Preserve agreement and disagreement without choosing medical truth."""

    predicted = set(predicted_labels)
    comparisons: list[dict[str, Any]] = []
    for profile in profiles:
        dataset_tags = profile.get("dataset_assigned_dosha_tags")
        if dataset_tags is None:
            dataset_tags = profile.get("content", {}).get(
                "dataset_assigned_dosha_tags", []
            )
        dataset = set(dataset_tags)
        if not predicted:
            relationship = "not_compared_due_to_ml_abstention"
        elif predicted == dataset:
            relationship = "set_agreement"
        elif predicted & dataset:
            relationship = "partial_disagreement"
        else:
            relationship = "disagreement"
        comparisons.append(
            {
                "source_profile_id": (
                    profile.get("source_profile_identifier")
                    or profile.get("source_profile_id")
                ),
                "model_predicted_dosha_labels": sorted(predicted),
                "dataset_assigned_dosha_tags": list(
                    dataset_tags
                ),
                "relationship": relationship,
                "interpretation": (
                    "Neither output is treated as clinically correct; both are retained."
                ),
            }
        )
    if condition == "Insomnia":
        comparisons.append(
            {
                "type": "retained_dataset_disagreement",
                "associations": [["Vata"], ["Vata", "Pitta"]],
                "resolution": "none",
                "interpretation": (
                    "Both source-profile associations remain visible; the model, majority "
                    "vote and retrieval rank do not resolve them."
                ),
            }
        )
    if predicted and profiles and not any(
        predicted
        & set(
            profile.get("dataset_assigned_dosha_tags")
            or profile.get("content", {}).get("dataset_assigned_dosha_tags", [])
        )
        for profile in profiles
    ):
        comparisons.append(
            {
                "type": "no_matching_dosha_profile",
                "model_predicted_dosha_labels": sorted(predicted),
                "fallback": "condition_scoped_lexical_results_retained",
            }
        )
    return comparisons
