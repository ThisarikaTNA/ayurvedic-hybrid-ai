"""Record the user's approved five-condition Phase 5 decision."""

from __future__ import annotations

import json
from pathlib import Path


MANIFEST_PATH = Path("outputs/phase5_condition_selection/phase5_selection.json")
APPROVED = [
    "acne",
    "common cold",
    "gastroesophageal reflux disease gerd",
    "osteoarthritis",
    "insomnia",
]


def main() -> int:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("recommended_conditions") != APPROVED:
        raise ValueError("The approved conditions do not match the Phase 5 recommendations.")
    payload["status"] = "approved_and_finalized_for_phase6"
    payload["approval"] = {
        "approved": True,
        "approval_date": "2026-08-06",
        "approved_conditions": APPROVED,
        "selection_basis": [
            "portfolio diversity",
            "knowledge completeness",
            "explainability",
            "manageable MSc prototype scope",
            "authoritative source availability",
        ],
        "explicitly_not_selection_basis": [
            "medical correctness",
            "clinical validation",
            "ML sample size",
            "final-test results or errors",
        ],
        "next_authorized_phase": 6,
        "phase7_rule_engine_authorized": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Phase 5 approval recorded for five conditions; Phase 7 remains unauthorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
