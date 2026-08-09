"""Phase 9 adapter around the unchanged Phase 8 condition-scoped retriever."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


class RetrievalAdapterError(RuntimeError):
    """Raised when condition-scoped retrieval fails at its component boundary."""


class RetrievalAdapter:
    """Expose ML labels as ranking signals without calling them caller-supplied."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self.invocation_count = 0

    def retrieve(
        self,
        *,
        condition: str,
        symptom_text: str,
        predicted_labels: Sequence[str],
        categories: Sequence[str],
        top_k: int,
    ) -> dict[str, Any]:
        tag_source = "ml_prediction" if predicted_labels else "none"
        internal_request = {
            "condition": condition,
            "free_text": symptom_text,
            "categories": list(categories),
            # Phase 8's validated signal slot is reused internally. The adapter removes
            # its caller-oriented wording before the Phase 9 result boundary.
            "caller_supplied_dosha_tags": list(predicted_labels),
            "top_k": top_k,
        }
        try:
            raw = self.retriever.retrieve(internal_request)
        except Exception as error:
            raise RetrievalAdapterError(f"Condition-scoped retrieval failed: {error}") from error
        self.invocation_count += 1
        result = self._translate(copy.deepcopy(raw), tag_source, predicted_labels)
        result["adapter_trace"] = {
            "tag_source": tag_source,
            "ranking_dosha_tags": list(predicted_labels),
            "lexical_query_source": "validated_symptom_text",
            "condition_scope_source": "caller_selected_and_resolved",
            "ranking_configuration": "unchanged_phase8_config",
            "no_combined_hybrid_score": True,
        }
        return result

    def retrieve_reference_only(
        self, *, condition: str, categories: Sequence[str], top_k: int
    ) -> dict[str, Any]:
        permitted = [category for category in categories if category in {"claims", "sources"}]
        if not permitted:
            permitted = ["claims", "sources"]
        try:
            raw = self.retriever.retrieve(
                {
                    "condition": condition,
                    "categories": permitted,
                    "top_k": top_k,
                }
            )
        except Exception as error:
            raise RetrievalAdapterError(f"Reference-only retrieval failed: {error}") from error
        self.invocation_count += 1
        result = self._translate(copy.deepcopy(raw), "none", [])
        result["adapter_trace"] = {
            "tag_source": "none",
            "ranking_dosha_tags": [],
            "lexical_query_source": "none_component_degraded",
            "condition_scope_source": "caller_selected_and_resolved",
            "ranking_configuration": "unchanged_phase8_config",
            "reference_only_degraded_mode": True,
            "no_combined_hybrid_score": True,
        }
        return result

    def _translate(
        self, value: Any, tag_source: str, tags: Sequence[str]
    ) -> Any:
        if isinstance(value, dict):
            translated: dict[str, Any] = {}
            for key, item in value.items():
                if key == "caller_supplied_dosha_tags":
                    translated["ranking_dosha_tags"] = list(tags)
                elif key == "caller_supplied_tags":
                    translated["ranking_tags"] = list(tags)
                elif key == "caller_supplied_dosha_jaccard":
                    translated["dosha_ranking_signal_jaccard"] = item
                elif key == "caller_supplied_dosha_note":
                    translated["dosha_ranking_signal_note"] = (
                        "ML-predicted dataset-label outputs are secondary retrieval "
                        "signals only; they are not inferred medical truth."
                        if tag_source == "ml_prediction"
                        else "No Dosha ranking signal was used."
                    )
                else:
                    translated[key] = self._translate(item, tag_source, tags)
            if "dosha_overlap_details" in translated and isinstance(
                translated["dosha_overlap_details"], dict
            ):
                translated["dosha_overlap_details"]["tag_source"] = tag_source
            return translated
        if isinstance(value, list):
            return [self._translate(item, tag_source, tags) for item in value]
        if isinstance(value, str) and tag_source == "ml_prediction":
            return value.replace(
                "Caller-supplied Dosha", "ML-derived Dosha-label"
            ).replace(
                "caller-supplied Dosha", "ML-derived Dosha-label"
            )
        return value
