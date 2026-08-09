"""Strict validation for non-executable Phase 7 rule JSON structures."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


class RuleValidationError(ValueError):
    """Raised when a rule or structured input violates the allowlist."""


SUPPORTED_OPERATORS: tuple[str, ...] = (
    "all", "any", "not", "eq", "in", "contains", "gte", "lte", "exists"
)
SUPPORTED_ACTIONS: tuple[str, ...] = (
    "professional_referral", "recommendation_exclusion",
    "contraindication_warning", "dataset_demonstration", "general_recommendation",
)
SUPPORTED_RULE_TYPES: tuple[str, ...] = (
    "professional_referral", "recommendation_exclusion",
    "contraindication_warning", "dataset_personalization_demo",
    "general_recommendation",
)
RULE_PRIORITIES: Mapping[str, int] = {
    "professional_referral": 1,
    "recommendation_exclusion": 2,
    "contraindication_warning": 3,
    "dataset_personalization_demo": 4,
    "general_recommendation": 5,
}
RULE_TO_ACTION: Mapping[str, str] = {
    "professional_referral": "professional_referral",
    "recommendation_exclusion": "recommendation_exclusion",
    "contraindication_warning": "contraindication_warning",
    "dataset_personalization_demo": "dataset_demonstration",
    "general_recommendation": "general_recommendation",
}

CANONICAL_CONDITIONS: tuple[str, ...] = (
    "Acne", "Common Cold", "Gastroesophageal Reflux Disease",
    "Osteoarthritis", "Insomnia",
)
COMMON_FIELDS: Mapping[str, dict[str, Any]] = {
    "general_self_care_requested": {"type": bool},
    "reported_features": {"type": list, "item_type": str},
}
CONDITION_FIELDS: Mapping[str, Mapping[str, dict[str, Any]]] = {
    "Acne": {
        "acne_severity": {"type": str, "choices": {"mild", "moderate", "severe"}},
        "has_nodules": {"type": bool},
        "has_cysts": {"type": bool},
        "picks_or_squeezes_spots": {"type": bool},
    },
    "Common Cold": {
        "cold_symptoms_worsening": {"type": bool},
        "shortness_of_breath": {"type": bool},
        "chest_pain": {"type": bool},
        "cold_symptom_duration_days": {"type": (int, float), "minimum": 0},
        "cold_not_improving": {"type": bool},
    },
    "Gastroesophageal Reflux Disease": {
        "reflux_self_care_not_helping": {"type": bool},
        "heartburn_most_days": {"type": bool},
        "food_sticking": {"type": bool},
        "frequent_vomiting": {"type": bool},
        "unexplained_weight_loss": {"type": bool},
    },
    "Osteoarthritis": {
        "persistent_joint_symptoms": {"type": bool},
        "overweight_context": {"type": bool},
    },
    "Insomnia": {
        "sleep_habit_changes_not_helped": {"type": bool},
        "trouble_sleeping_for_months": {"type": bool},
        "daily_life_hard_to_cope": {"type": bool},
    },
}

RULE_KEYS: frozenset[str] = frozenset({
    "rule_id", "rule_key", "rule_version", "rule_name", "condition", "rule_type",
    "lifecycle_status", "evidence_status", "conditions", "action", "priority",
    "explanation_template", "claim_ids", "source_ids", "source_locator",
    "validation_status", "last_validation_date", "limitations", "safety_notes",
})
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
IDENTIFIER_PATTERN = re.compile(r"^R-[A-Z0-9-]+$")
DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|milligrams?|g|grams?|mcg|µg|ml|millilit(?:er|re)s?|"
    r"tablets?|capsules?|drops?)\b",
    flags=re.IGNORECASE,
)


def allowed_fields(condition: str) -> dict[str, dict[str, Any]]:
    if condition not in CONDITION_FIELDS:
        raise RuleValidationError(f"Unknown condition context: {condition!r}")
    return {**COMMON_FIELDS, **CONDITION_FIELDS[condition]}


def _reject_unknown_keys(node: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(node) - allowed)
    if unknown:
        raise RuleValidationError(f"Unknown {context} fields: {unknown}")


def validate_predicate(node: Any, condition: str, path: str = "conditions") -> None:
    """Validate a recursive predicate against fixed operators and input fields."""

    if not isinstance(node, dict):
        raise RuleValidationError(f"{path} must be an object.")
    operator = node.get("operator")
    if operator not in SUPPORTED_OPERATORS:
        raise RuleValidationError(f"Unknown operator at {path}: {operator!r}")
    if operator in {"all", "any"}:
        _reject_unknown_keys(node, {"operator", "conditions"}, path)
        children = node.get("conditions")
        if not isinstance(children, list) or not children:
            raise RuleValidationError(f"{path}.conditions must be a non-empty list.")
        for index, child in enumerate(children):
            validate_predicate(child, condition, f"{path}.conditions[{index}]")
        return
    if operator == "not":
        _reject_unknown_keys(node, {"operator", "condition"}, path)
        if "condition" not in node:
            raise RuleValidationError(f"{path}.condition is required for not.")
        validate_predicate(node["condition"], condition, f"{path}.condition")
        return

    allowed = {"field", "operator"} if operator == "exists" else {"field", "operator", "value"}
    _reject_unknown_keys(node, allowed, path)
    field = node.get("field")
    field_specs = allowed_fields(condition)
    if field not in field_specs:
        raise RuleValidationError(f"Unknown input field at {path}: {field!r}")
    if operator != "exists" and "value" not in node:
        raise RuleValidationError(f"{path}.value is required for {operator}.")
    value = node.get("value")
    if operator == "in" and (not isinstance(value, list) or not value):
        raise RuleValidationError(f"{path}.value must be a non-empty list for in.")
    if operator in {"gte", "lte"} and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise RuleValidationError(f"{path}.value must be numeric for {operator}.")
    if operator == "contains" and not isinstance(value, (str, int, float, bool)):
        raise RuleValidationError(f"{path}.value must be a scalar for contains.")


def validate_action(action: Any) -> None:
    if not isinstance(action, dict):
        raise RuleValidationError("action must be an object.")
    _reject_unknown_keys(
        action,
        {"type", "message", "conflict_group", "value", "prominence", "target_action_type"},
        "action",
    )
    required = {"type", "message", "conflict_group", "value", "prominence"}
    missing = sorted(required - set(action))
    if missing:
        raise RuleValidationError(f"Missing action fields: {missing}")
    if action["type"] not in SUPPORTED_ACTIONS:
        raise RuleValidationError(f"Unknown action type: {action['type']!r}")
    if action["prominence"] not in {"high", "normal", "low"}:
        raise RuleValidationError("Unknown action prominence.")
    for field in ("message", "conflict_group", "value"):
        if not isinstance(action[field], str) or not action[field].strip():
            raise RuleValidationError(f"action.{field} must be non-empty text.")
    if DOSAGE_PATTERN.search(action["message"]):
        raise RuleValidationError("Exact quantity-like prescribing content is forbidden.")


def validate_rule_definition(rule: Mapping[str, Any]) -> None:
    """Validate a complete rule without evaluating or executing stored code."""

    if not isinstance(rule, Mapping):
        raise RuleValidationError("Rule must be an object.")
    unknown = sorted(set(rule) - RULE_KEYS)
    missing = sorted(RULE_KEYS - set(rule))
    if unknown:
        raise RuleValidationError(f"Unknown rule fields: {unknown}")
    if missing:
        raise RuleValidationError(f"Missing rule fields: {missing}")
    for identifier in ("rule_id", "rule_key"):
        if not isinstance(rule[identifier], str) or not IDENTIFIER_PATTERN.fullmatch(rule[identifier]):
            raise RuleValidationError(f"Invalid {identifier}: {rule[identifier]!r}")
    if not isinstance(rule["rule_version"], str) or not SEMVER_PATTERN.fullmatch(rule["rule_version"]):
        raise RuleValidationError("rule_version must use numeric MAJOR.MINOR.PATCH format.")
    if rule["condition"] not in CANONICAL_CONDITIONS:
        raise RuleValidationError("Rule must use one approved canonical condition.")
    if rule["rule_type"] not in SUPPORTED_RULE_TYPES:
        raise RuleValidationError("Unknown rule type.")
    if rule["lifecycle_status"] not in {"draft", "active", "inactive"}:
        raise RuleValidationError("Unknown lifecycle status.")
    if rule["evidence_status"] not in {"dataset_derived", "reference_checked", "expert_reviewed"}:
        raise RuleValidationError("Unknown evidence status.")
    if rule["evidence_status"] == "expert_reviewed":
        raise RuleValidationError("No genuine expert-reviewed evidence exists in Phase 7.")
    if rule["validation_status"] not in {"pending", "valid", "invalid"}:
        raise RuleValidationError("Unknown structural validation status.")
    if not isinstance(rule["priority"], int) or isinstance(rule["priority"], bool):
        raise RuleValidationError("priority must be an integer.")
    if rule["priority"] != RULE_PRIORITIES[rule["rule_type"]]:
        raise RuleValidationError("Rule priority does not match the controlled priority map.")
    validate_action(rule["action"])
    if rule["action"].get("type") != RULE_TO_ACTION[rule["rule_type"]]:
        raise RuleValidationError("Rule type and action type are incompatible.")
    if rule["rule_type"] in {
        "professional_referral", "recommendation_exclusion", "contraindication_warning"
    } and rule["evidence_status"] != "reference_checked":
        raise RuleValidationError("Safety actions require reference-checked evidence.")
    if rule["evidence_status"] == "dataset_derived" and (
        rule["rule_type"] != "dataset_personalization_demo"
        or rule["action"].get("type") != "dataset_demonstration"
    ):
        raise RuleValidationError("Dataset-derived rules are limited to non-clinical demonstrations.")
    for field in ("rule_name", "explanation_template", "source_locator", "limitations", "safety_notes"):
        if not isinstance(rule[field], str) or not rule[field].strip():
            raise RuleValidationError(f"{field} must be non-empty text.")
    for field in ("claim_ids", "source_ids"):
        values = rule[field]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise RuleValidationError(f"{field} must be a unique list.")
    if not rule["claim_ids"]:
        raise RuleValidationError("At least one supporting claim is required.")
    if rule["evidence_status"] == "reference_checked" and not rule["source_ids"]:
        raise RuleValidationError("Reference-checked rules require a source ID.")
    validate_predicate(rule["conditions"], rule["condition"])
    combined_text = " ".join(
        [rule["explanation_template"], rule["limitations"], rule["safety_notes"]]
    )
    if DOSAGE_PATTERN.search(combined_text):
        raise RuleValidationError("Exact quantity-like content is forbidden in rule text.")


def parse_and_validate_json(value: str, *, kind: str, condition: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuleValidationError(f"Malformed {kind} JSON.") from error
    if kind == "conditions":
        validate_predicate(parsed, condition)
    elif kind == "action":
        validate_action(parsed)
    else:
        raise RuleValidationError(f"Unknown JSON validation kind: {kind}")
    return parsed


def validate_structured_input(condition: str, facts: Any) -> dict[str, Any]:
    """Validate a user-selected condition context and synthetic structured facts."""

    specifications = allowed_fields(condition)
    if not isinstance(facts, dict):
        raise RuleValidationError("facts must be an object.")
    unknown = sorted(set(facts) - set(specifications))
    if unknown:
        raise RuleValidationError(f"Unknown input fields: {unknown}")
    validated: dict[str, Any] = {}
    for field, value in facts.items():
        if value is None:
            validated[field] = None
            continue
        spec = specifications[field]
        expected = spec["type"]
        if not isinstance(value, expected) or (
            expected == (int, float) and isinstance(value, bool)
        ):
            raise RuleValidationError(f"Invalid type for input field {field!r}.")
        if "choices" in spec and value not in spec["choices"]:
            raise RuleValidationError(f"Invalid value for input field {field!r}.")
        if "minimum" in spec and value < spec["minimum"]:
            raise RuleValidationError(f"Input field {field!r} is below its minimum.")
        if isinstance(value, list) and not all(
            isinstance(item, spec.get("item_type", object)) for item in value
        ):
            raise RuleValidationError(f"Invalid list item for input field {field!r}.")
        validated[field] = value
    return validated


def semantic_version_key(version: str) -> tuple[int, int, int]:
    if not SEMVER_PATTERN.fullmatch(version):
        raise RuleValidationError(f"Invalid rule version: {version!r}")
    return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]
