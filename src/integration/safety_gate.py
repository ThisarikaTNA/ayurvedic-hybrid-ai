"""Fail-closed deterministic safety gate over the validated Phase 7 rules."""

from __future__ import annotations

from typing import Any, Mapping


SAFETY_STATES = (
    "blocked_invalid_input",
    "clarification_required",
    "escalation_information",
    "proceed_with_information",
    "blocked_rule_conflict",
    "blocked_component_failure",
)


class SafetyGate:
    """Interpret rule-engine traces as orchestration states, not medical safety."""

    def __init__(self, rule_engine: Any, inventory: Mapping[str, int]) -> None:
        self.rule_engine = rule_engine
        self.inventory = dict(inventory)

    def evaluate(self, condition: str, facts: Mapping[str, Any]) -> dict[str, Any]:
        try:
            engine_result = self.rule_engine.evaluate(condition, facts)
        except Exception as error:  # fail closed at the component boundary
            return self._failure("rule_engine_exception", str(error))

        traces = engine_result.get("explanation_trace")
        if not isinstance(traces, list):
            return self._failure("missing_rule_trace", "Rule trace was not returned.")
        if engine_result.get("ignored_rules"):
            return self._failure(
                "ignored_or_invalid_rule",
                "One or more applicable rules failed lifecycle or evidence checks.",
                engine_result,
            )
        if engine_result.get("conflicts"):
            return {
                "state": "blocked_rule_conflict",
                "personalization_permitted": False,
                "reason": "The rule engine reported an unresolved conflict.",
                "missing_structured_fields": [],
                "prominent_referral_information": [],
                "rule_engine_result": engine_result,
                "limited_inventory_disclosure": self._inventory_disclosure(),
            }

        referral = [t for t in traces if t.get("rule_type") == "professional_referral"]
        if len(referral) != 1:
            return self._failure(
                "referral_rule_inventory_failure",
                f"Expected one applicable referral rule, observed {len(referral)}.",
                engine_result,
            )
        referral_trace = referral[0]
        if referral_trace.get("outcome") == "fired":
            return {
                "state": "escalation_information",
                "personalization_permitted": False,
                "reason": "The condition-specific reference-checked referral rule fired.",
                "missing_structured_fields": referral_trace.get(
                    "missing_required_inputs", []
                ),
                "prominent_referral_information": engine_result.get(
                    "prominent_referral_messages", []
                ),
                "rule_engine_result": engine_result,
                "limited_inventory_disclosure": self._inventory_disclosure(),
            }
        if referral_trace.get("outcome") == "not_evaluable":
            return {
                "state": "clarification_required",
                "personalization_permitted": False,
                "reason": "Required structured safety facts are missing.",
                "missing_structured_fields": referral_trace.get(
                    "missing_required_inputs", []
                ),
                "prominent_referral_information": [],
                "rule_engine_result": engine_result,
                "limited_inventory_disclosure": self._inventory_disclosure(),
            }
        if referral_trace.get("outcome") != "not_fired":
            return self._failure(
                "unknown_referral_outcome",
                f"Unexpected referral outcome: {referral_trace.get('outcome')!r}.",
                engine_result,
            )
        return {
            "state": "proceed_with_information",
            "personalization_permitted": True,
            "reason": (
                "The implemented referral rule was completely evaluated and did not fire; "
                "this is not a determination of medical safety."
            ),
            "missing_structured_fields": [],
            "prominent_referral_information": [],
            "rule_engine_result": engine_result,
            "limited_inventory_disclosure": self._inventory_disclosure(),
        }

    def _inventory_disclosure(self) -> dict[str, Any]:
        return {
            **self.inventory,
            "interpretation": (
                "This limited inventory does not establish comprehensive safety, especially "
                "because contraindication, exclusion and expert-reviewed rules are absent."
            ),
        }

    def _failure(
        self, code: str, reason: str, engine_result: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "state": "blocked_component_failure",
            "personalization_permitted": False,
            "reason": reason,
            "failure_code": code,
            "missing_structured_fields": [],
            "prominent_referral_information": [],
            "rule_engine_result": dict(engine_result or {}),
            "limited_inventory_disclosure": self._inventory_disclosure(),
        }
