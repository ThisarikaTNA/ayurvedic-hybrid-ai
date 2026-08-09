"""Strict input validation for Phase 9 hybrid orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from knowledge_base.rule_schema import RuleValidationError, validate_structured_input
from retrieval.query_schema import CONTROL_PATTERN, DOSAGE_LIKE_PATTERN


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class HybridInputError(ValueError):
    """Raised when a Phase 9 input violates the documented contract."""


@dataclass(frozen=True)
class ValidatedBaseInput:
    """Input validated before condition resolution."""

    condition: str
    symptom_text: str
    safety_facts: dict[str, Any]
    requested_information_categories: tuple[str, ...]
    top_k: int
    request_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "symptom_text": self.symptom_text,
            "safety_facts": self.safety_facts,
            "requested_information_categories": list(
                self.requested_information_categories
            ),
            "top_k": self.top_k,
            "request_id": self.request_id,
        }


def _categories(value: Any, allowed: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(allowed)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise HybridInputError("requested_information_categories must be a list.")
    requested: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise HybridInputError("Every information category must be text.")
        normalized = item.strip().casefold()
        if normalized not in allowed:
            raise HybridInputError(f"Unknown information category: {item!r}")
        if normalized not in requested:
            requested.append(normalized)
    if not requested:
        raise HybridInputError("At least one information category is required.")
    return tuple(item for item in allowed if item in requested)


def validate_base_input(
    request: Mapping[str, Any], config: Mapping[str, Any]
) -> ValidatedBaseInput:
    """Validate non-medical input fields before resolving the supplied condition."""

    if not isinstance(request, Mapping):
        raise HybridInputError("Hybrid input must be an object.")
    allowed_fields = {
        "condition",
        "symptom_text",
        "safety_facts",
        "requested_information_categories",
        "top_k",
        "request_id",
    }
    unknown = sorted(set(request) - allowed_fields)
    if unknown:
        raise HybridInputError(f"Unknown hybrid input fields: {unknown}")

    limits = config["input_limits"]
    condition = request.get("condition")
    if not isinstance(condition, str) or not condition.strip():
        raise HybridInputError("A caller-selected condition or alias is required.")
    condition = condition.strip()
    if len(condition) > int(limits["condition_characters"]):
        raise HybridInputError("Condition input is too long.")
    if CONTROL_PATTERN.search(condition):
        raise HybridInputError("Condition input contains control characters.")

    symptom_text = request.get("symptom_text")
    if not isinstance(symptom_text, str) or not symptom_text.strip():
        raise HybridInputError("Non-empty symptom_text is required for the ML component.")
    symptom_text = symptom_text.strip()
    if len(symptom_text) > int(limits["symptom_characters"]):
        raise HybridInputError("Symptom text is too long.")
    if CONTROL_PATTERN.search(symptom_text):
        raise HybridInputError("Symptom text contains control characters.")
    if DOSAGE_LIKE_PATTERN.search(symptom_text):
        raise HybridInputError("Exact dosage-like text is outside the Phase 9 scope.")

    safety_facts = request.get("safety_facts")
    if not isinstance(safety_facts, dict):
        raise HybridInputError("safety_facts must be an object.")

    top_k = request.get("top_k", limits["top_k_default"])
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise HybridInputError("top_k must be an integer.")
    if not int(limits["top_k_minimum"]) <= top_k <= int(
        limits["top_k_maximum"]
    ):
        raise HybridInputError(
            f"top_k must be between {limits['top_k_minimum']} and "
            f"{limits['top_k_maximum']}."
        )

    request_id = request.get("request_id")
    if request_id is not None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise HybridInputError("request_id must be non-empty text when supplied.")
        request_id = request_id.strip()
        if len(request_id) > int(limits["request_id_characters"]):
            raise HybridInputError("request_id is too long.")
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            raise HybridInputError(
                "request_id may contain only letters, numbers, dot, underscore and hyphen."
            )

    return ValidatedBaseInput(
        condition=condition,
        symptom_text=symptom_text,
        safety_facts=dict(safety_facts),
        requested_information_categories=_categories(
            request.get("requested_information_categories"),
            config["allowed_information_categories"],
        ),
        top_k=top_k,
        request_id=request_id,
    )


def validate_condition_facts(condition: str, facts: Any) -> dict[str, Any]:
    """Apply the unchanged Phase 7 condition-specific field allowlist."""

    try:
        return validate_structured_input(condition, facts)
    except RuleValidationError as error:
        raise HybridInputError(str(error)) from error
