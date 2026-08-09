"""Deterministic, explainable evaluation of validated Phase 7 rule JSON."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any, Mapping

from knowledge_base.repository import RuleRepository
from knowledge_base.rule_schema import (
    RuleValidationError,
    parse_and_validate_json,
    semantic_version_key,
    validate_rule_definition,
    validate_structured_input,
)


DISCLAIMER = (
    "Educational research prototype only. The condition context was supplied by the user "
    "or caller; it was not inferred. Outputs are candidate information actions, not a "
    "diagnosis, Dosha inference, prescription, exact dosage, guaranteed benefit, clinical "
    "validation, or replacement for professional healthcare."
)
SAFETY_TYPES = {
    "professional_referral", "recommendation_exclusion", "contraindication_warning"
}


def _evaluate_predicate(
    node: Mapping[str, Any], facts: Mapping[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any], list[str]]:
    operator = node["operator"]
    if operator in {"all", "any"}:
        child_results = [_evaluate_predicate(child, facts) for child in node["conditions"]]
        statuses = [result[0] for result in child_results]
        if operator == "all":
            status = (
                "false" if "false" in statuses else
                "not_evaluable" if "not_evaluable" in statuses else "true"
            )
        else:
            status = (
                "true" if "true" in statuses else
                "not_evaluable" if "not_evaluable" in statuses else "false"
            )
        used: dict[str, Any] = {}
        missing: list[str] = []
        for _, _, child_used, child_missing in child_results:
            used.update(child_used)
            missing.extend(child_missing)
        trace = {
            "operator": operator,
            "result": status,
            "children": [result[1] for result in child_results],
        }
        return status, trace, used, sorted(set(missing))
    if operator == "not":
        child_status, child_trace, used, missing = _evaluate_predicate(
            node["condition"], facts
        )
        status = (
            "not_evaluable" if child_status == "not_evaluable"
            else "false" if child_status == "true" else "true"
        )
        return status, {
            "operator": "not", "result": status, "child": child_trace
        }, used, missing

    field = node["field"]
    if field not in facts or facts[field] is None:
        if operator == "exists":
            return "false", {
                "field": field, "operator": operator, "result": "false",
                "fact_present": False,
            }, {}, [field]
        return "not_evaluable", {
            "field": field, "operator": operator, "result": "not_evaluable",
            "reason": "missing_input",
        }, {}, [field]
    fact = facts[field]
    if operator == "exists":
        result = True
    elif operator == "eq":
        result = fact == node["value"]
    elif operator == "in":
        result = fact in node["value"]
    elif operator == "contains":
        result = node["value"] in fact
    elif operator == "gte":
        result = fact >= node["value"]
    elif operator == "lte":
        result = fact <= node["value"]
    else:  # pragma: no cover - schema validation makes this unreachable.
        raise RuleValidationError(f"Unsupported evaluated operator: {operator}")
    status = "true" if result else "false"
    trace = {
        "field": field,
        "operator": operator,
        "expected": node.get("value"),
        "actual": fact,
        "result": status,
    }
    return status, trace, {field: fact}, []


class RuleEngine:
    """Load, validate, evaluate, order, and trace structured database rules."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.repository = RuleRepository(connection)

    def _definition_and_support(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, Any] | None, str | None]:
        evidence = self.repository.rule_evidence(row["rule_id"])
        validations = self.repository.rule_validations(row["rule_id"])
        if not evidence:
            return None, "missing_rule_evidence"
        if not validations:
            return None, "missing_rule_validation"
        if not any(
            validation["structural_validation_status"] == "valid"
            for validation in validations
        ):
            return None, "missing_valid_structural_validation"
        claim_ids: list[str] = []
        source_ids: list[str] = []
        locators: list[str] = []
        for link in evidence:
            if link["verified_claim_id"] is None or link["verified_source_id"] is None:
                return None, "orphaned_claim_or_source"
            if link["claim_condition_id"] != row["condition_id"]:
                return None, "claim_condition_mismatch"
            if link["claim_is_active"] != 1:
                return None, "inactive_supporting_claim"
            if link["claim_evidence_status"] != row["evidence_status"]:
                return None, "claim_evidence_status_mismatch"
            if link["link_validation_status"] != row["evidence_status"]:
                return None, "rule_evidence_status_mismatch"
            if row["evidence_status"] == "reference_checked":
                if (
                    link["source_validation_status"] != "reference_checked"
                    or link["supports_complete_claim"] != 1
                    or link["claim_evidence_link_status"] != "reference_checked"
                ):
                    return None, "incomplete_reference_support"
            if row["rule_type"] == "professional_referral" and link["claim_type"] != "referral_consideration":
                return None, "referral_rule_requires_referral_claim"
            if row["rule_type"] == "contraindication_warning" and link["claim_type"] != "contraindication":
                return None, "contraindication_rule_requires_checked_contraindication_claim"
            claim_ids.append(link["claim_id"])
            source_ids.append(link["source_id"])
            if link["source_locator"]:
                locators.append(link["source_locator"])
        if row["evidence_status"] == "dataset_derived" and row["rule_type"] in SAFETY_TYPES:
            return None, "dataset_rule_blocked_from_safety_action"
        try:
            conditions = parse_and_validate_json(
                row["conditions_json"], kind="conditions", condition=row["canonical_name"]
            )
            action = parse_and_validate_json(
                row["action_json"], kind="action", condition=row["canonical_name"]
            )
        except RuleValidationError as error:
            return None, f"invalid_structure:{error}"
        definition = {
            "rule_id": row["rule_id"], "rule_key": row["rule_key"],
            "rule_version": row["rule_version"], "rule_name": row["rule_name"],
            "condition": row["canonical_name"], "rule_type": row["rule_type"],
            "lifecycle_status": row["lifecycle_status"],
            "evidence_status": row["evidence_status"], "conditions": conditions,
            "action": action, "priority": row["priority"],
            "explanation_template": row["explanation"], "claim_ids": sorted(set(claim_ids)),
            "source_ids": sorted(set(source_ids)),
            "source_locator": " | ".join(sorted(set(locators))),
            "validation_status": row["structural_validation_status"],
            "last_validation_date": row["last_validation_date"],
            "limitations": row["limitations"], "safety_notes": row["safety_notes"],
        }
        try:
            validate_rule_definition(definition)
        except RuleValidationError as error:
            return None, f"invalid_rule:{error}"
        return definition, None

    def _load_rules(self, condition_id: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        rows = self.repository.list_condition_rules(condition_id)
        eligible_rows: list[sqlite3.Row] = []
        ignored: list[dict[str, str]] = []
        for row in rows:
            reason = None
            if row["lifecycle_status"] != "active":
                reason = f"lifecycle_{row['lifecycle_status']}"
            elif row["is_stale"] != 0:
                reason = "stale_rule"
            elif row["structural_validation_status"] != "valid":
                reason = f"structural_{row['structural_validation_status']}"
            if reason:
                ignored.append({"rule_id": row["rule_id"], "rule_version": row["rule_version"], "reason": reason})
            else:
                eligible_rows.append(row)

        by_key: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in eligible_rows:
            by_key[row["rule_key"]].append(row)
        selected_rows: list[sqlite3.Row] = []
        for key in sorted(by_key):
            versions = sorted(
                by_key[key], key=lambda item: (semantic_version_key(item["rule_version"]), item["rule_id"])
            )
            selected_rows.append(versions[-1])
            for older in versions[:-1]:
                ignored.append({
                    "rule_id": older["rule_id"], "rule_version": older["rule_version"],
                    "reason": f"superseded_by_version_{versions[-1]['rule_version']}",
                })

        definitions: list[dict[str, Any]] = []
        for row in selected_rows:
            definition, reason = self._definition_and_support(row)
            if reason:
                ignored.append({"rule_id": row["rule_id"], "rule_version": row["rule_version"], "reason": reason})
            else:
                definitions.append(definition)
        definitions.sort(key=lambda rule: (rule["priority"], rule["rule_id"], semantic_version_key(rule["rule_version"])))
        ignored.sort(key=lambda item: (item["rule_id"], item["rule_version"], item["reason"]))
        return definitions, ignored

    @staticmethod
    def _apply_suppression(
        traces: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        fired = [trace for trace in traces if trace["outcome"] == "fired"]
        by_priority_group: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for trace in fired:
            action = trace["proposed_action"]
            by_priority_group[(trace["priority"], action["conflict_group"])].append(trace)
        for (priority, group), members in sorted(by_priority_group.items()):
            values = {member["proposed_action"]["value"] for member in members}
            if len(values) > 1:
                conflict = {
                    "type": "equal_priority_incompatible_actions",
                    "priority": priority, "conflict_group": group,
                    "rule_ids": sorted(member["rule_id"] for member in members),
                    "values": sorted(values), "resolved": False,
                }
                conflicts.append(conflict)
                for member in members:
                    member["action_suppressed"] = True
                    member["suppression_reason"] = "unresolved_equal_priority_conflict"
                    member["conflict_details"].append(conflict)

        safety_fired = [
            trace for trace in fired
            if trace["rule_type"] in SAFETY_TYPES and not trace["action_suppressed"]
        ]
        safety_unknown = [
            trace for trace in traces
            if trace["rule_type"] in SAFETY_TYPES and trace["outcome"] == "not_evaluable"
        ]
        for trace in fired:
            if trace["rule_type"] != "general_recommendation" or trace["action_suppressed"]:
                continue
            if safety_fired:
                trace["action_suppressed"] = True
                trace["suppression_reason"] = "higher_priority_safety_action_fired"
            elif safety_unknown:
                trace["action_suppressed"] = True
                trace["suppression_reason"] = "higher_priority_safety_rule_not_evaluable"

        for higher in fired:
            if higher["action_suppressed"]:
                continue
            for lower in fired:
                if lower["action_suppressed"] or lower["priority"] <= higher["priority"]:
                    continue
                high_action = higher["proposed_action"]
                low_action = lower["proposed_action"]
                if (
                    high_action["conflict_group"] == low_action["conflict_group"]
                    and high_action["value"] != low_action["value"]
                ):
                    lower["action_suppressed"] = True
                    lower["suppression_reason"] = f"overridden_by_higher_priority_rule:{higher['rule_id']}"
        return conflicts

    def evaluate(self, condition_context: str, facts: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate rules for an explicitly supplied condition and synthetic facts."""

        condition = self.repository.resolve_condition(condition_context)
        if condition is None:
            raise RuleValidationError("Unknown or unapproved condition context.")
        canonical = condition["canonical_name"]
        validated_facts = validate_structured_input(canonical, facts)
        rules, ignored = self._load_rules(int(condition["condition_id"]))
        traces: list[dict[str, Any]] = []
        for rule in rules:
            status, predicate_trace, used, missing = _evaluate_predicate(
                rule["conditions"], validated_facts
            )
            outcome = (
                "fired" if status == "true" else
                "not_evaluable" if status == "not_evaluable" else "not_fired"
            )
            traces.append({
                "rule_id": rule["rule_id"], "rule_version": rule["rule_version"],
                "rule_type": rule["rule_type"], "priority": rule["priority"],
                "predicate_results": predicate_trace, "input_facts_used": used,
                "missing_required_inputs": missing, "outcome": outcome,
                "fired": outcome == "fired", "proposed_action": rule["action"],
                "action_suppressed": False, "suppression_reason": None,
                "conflict_details": [], "evidence_status": rule["evidence_status"],
                "supporting_claim_ids": rule["claim_ids"],
                "supporting_source_ids": rule["source_ids"],
                "source_locator": rule["source_locator"],
                "human_readable_explanation": rule["explanation_template"],
                "limitations": rule["limitations"], "safety_notes": rule["safety_notes"],
                "disclaimer": DISCLAIMER,
            })
        conflicts = self._apply_suppression(traces)
        candidate_actions = [
            {"rule_id": trace["rule_id"], "rule_version": trace["rule_version"], **trace["proposed_action"]}
            for trace in traces if trace["fired"] and not trace["action_suppressed"]
        ]
        suppressed_actions = [
            {
                "rule_id": trace["rule_id"], "rule_version": trace["rule_version"],
                **trace["proposed_action"], "suppression_reason": trace["suppression_reason"],
                "conflict_details": trace["conflict_details"],
            }
            for trace in traces if trace["fired"] and trace["action_suppressed"]
        ]
        prominent = [
            action["message"] for action in candidate_actions
            if action["type"] == "professional_referral"
        ]
        knowledge_notes: list[str] = []
        if canonical == "Insomnia":
            knowledge_notes.append(
                "The two dataset-derived Insomnia Dosha combinations conflict; they are retained as provenance and are not evaluated or resolved by this engine."
            )
        return {
            "condition_context": canonical,
            "condition_context_source": "externally_supplied_not_inferred",
            "engine_status": "evaluated",
            "candidate_actions": candidate_actions,
            "prominent_referral_messages": prominent,
            "suppressed_actions": suppressed_actions,
            "conflicts": conflicts,
            "evaluated_rule_count": len(traces),
            "ignored_rules": ignored,
            "explanation_trace": traces,
            "knowledge_notes": knowledge_notes,
            "disclaimer": DISCLAIMER,
        }
