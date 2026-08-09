"""Audit unsealed disease groups and propose Phase 5 knowledge-base conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from knowledge_base.condition_selection import (
    CANDIDATE_CONFIG,
    WEIGHTS,
    aggregate_condition_scores,
    apply_candidate_review,
    load_unsealed_profiles,
)
from models.selection_manifest import file_sha256


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--data", type=Path, default=Path("data/processed/ayurgenix_cleaned.csv")
    )
    command.add_argument(
        "--assignments", type=Path, default=Path("outputs/splits/split_assignments.csv")
    )
    command.add_argument(
        "--model-selection-manifest", type=Path,
        default=Path("outputs/phase4_validation/model_selection_manifest.json"),
    )
    command.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/phase5_condition_selection"),
    )
    command.add_argument(
        "--report", type=Path,
        default=Path("docs/phase5_condition_selection_report.md"),
    )
    return command


def _write_report(
    path: Path, candidates: pd.DataFrame, access_audit: dict, selection: dict
) -> None:
    rows = [
        "| Condition | Profiles | Dosha combinations | Score | Recommendation | Source status |",
        "|---|---:|---|---:|---|---|",
    ]
    for _, row in candidates.sort_values("selection_score", ascending=False).iterrows():
        dosha_display = str(row["dosha_combinations_present"]).replace(" | ", "<br>")
        rows.append(
            f"| {row['display_name']} | {int(row['profile_count'])} | "
            f"{dosha_display} | {row['selection_score']:.3f} | "
            f"{row['recommendation_status']} | verified authoritative page |"
        )
    details: list[str] = []
    for _, row in candidates.sort_values("selection_score", ascending=False).iterrows():
        details.append(
            f"""### {row['display_name']}

- Normalized name: `{row['normalized_condition_name']}`
- Profiles: {int(row['profile_count'])}
- Dataset Dosha combinations: {row['dosha_combinations_present']}
- Complete fields: `{row['relevant_fields_complete']}`
- Missing/placeholders: `{row['missing_or_placeholder_fields']}`
- Safety/referral opportunity: {row['possible_safety_referral_rule_opportunities']}
- General lifestyle opportunity: {row['possible_general_lifestyle_rule_opportunities']}
- External-source availability: {row['external_source_availability_status']}
- Source checked for availability: [{row['external_source_title']}]({row['external_source_url']})
- Selection score: {row['selection_score']:.3f}
- Decision rationale: {row['inclusion_or_exclusion_reason']}
- Limitations: {row['important_medical_or_data_limitations']}
"""
        )
    recommended = [
        row["display_name"]
        for _, row in candidates.iterrows()
        if row["recommendation_status"] == "recommended_for_user_approval"
    ]
    text = f"""# Phase 5 condition-selection report

## Scope and final-test seal

Phase 5 reviewed all {access_audit['retained_normalized_group_count']} normalized
disease groups available in the training and validation knowledge profiles
({access_audit['retained_profile_count']} profiles). Final-test rows, fields,
predictions, model errors, and metrics were not selected or inspected.

The full dataset contains additional sealed groups. The instruction to inspect
all groups was therefore interpreted as all **unsealed** groups; the final-test
seal takes precedence. Disease profile count is reported but is not part of
the selection score.

## Reproducible rubric

The score weights are `{json.dumps(WEIGHTS, sort_keys=True)}`. Completeness
measures field availability only. It does not validate medical correctness.
Dosha clarity is the majority dataset-combination proportion within a group.
Symptom distinctiveness uses TF-IDF cosine distance. Rule potential is a data-
sufficiency proxy, not a validated medical rule. Source availability means an
authoritative page was found; its content has not yet been translated into
rules. Manageable scope is an explicit coursework judgment, not clinical
approval.

The audited candidate fields are nearly universally populated, which is
consistent with templated profiles and makes completeness weak evidence for
choosing between conditions. It must not be interpreted as content quality.

## Ten candidates

{chr(10).join(rows)}

## Recommended five — awaiting approval

{', '.join(recommended)}

These five are recommendations only. They are not finalized, and no database
or rule engine has been created.

## Candidate details

{chr(10).join(details)}

## Dataset-derived versus external knowledge

Profile counts, Dosha combinations, completeness, and text-based
distinctiveness are dataset-derived. The listed URLs establish only that an
authoritative external page is available. No safety trigger, contraindication,
medical rule, dosage, or treatment recommendation has been imported or
invented. Any later rule must store its source, validation status, and review
date separately from dataset-derived associations.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    if not options.model_selection_manifest.exists():
        raise FileNotFoundError("Freeze the Phase 4 model-selection manifest first.")
    frozen_manifest = json.loads(
        options.model_selection_manifest.read_text(encoding="utf-8")
    )
    if frozen_manifest.get("status") != "frozen_after_user_acceptance":
        raise ValueError("Phase 4 model-selection manifest is not frozen.")
    dataframe, access_audit = load_unsealed_profiles(options.data, options.assignments)
    scores = aggregate_condition_scores(dataframe)
    candidates = apply_candidate_review(scores)
    recommended = candidates[
        candidates["recommendation_status"] == "recommended_for_user_approval"
    ]
    if len(candidates) != 10 or len(recommended) != 5:
        raise ValueError("Phase 5 must produce ten candidates and five recommendations.")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_columns = [
        "normalized_condition_name", "display_name", "profile_count",
        "dosha_combinations_present", "relevant_fields_complete",
        "missing_or_placeholder_fields", "possible_safety_referral_rule_opportunities",
        "possible_general_lifestyle_rule_opportunities",
        "external_source_availability_status", "external_source_title",
        "external_source_url", "selection_score", "recommendation_status",
        "inclusion_or_exclusion_reason", "important_medical_or_data_limitations",
    ]
    candidates.loc[:, candidate_columns].to_csv(
        options.output_dir / "condition_candidates.csv", index=False
    )
    scores.drop(columns=["symptom_document"]).to_csv(
        options.output_dir / "selection_scores.csv", index=False
    )
    selection = {
        "phase": 5,
        "status": "recommendations_only_awaiting_user_approval",
        "model_selection_manifest": {
            "path": str(options.model_selection_manifest.resolve()),
            "sha256": file_sha256(options.model_selection_manifest),
        },
        "data_access_audit": access_audit,
        "scoring_weights": dict(WEIGHTS),
        "candidate_count": len(CANDIDATE_CONFIG),
        "recommended_conditions": recommended["normalized_condition_name"].tolist(),
        "candidates": candidates.loc[:, candidate_columns].to_dict(orient="records"),
        "separation_of_knowledge": {
            "dataset_derived": [
                "profile counts", "Dosha combinations", "field completeness",
                "symptom distinctiveness",
            ],
            "externally_verified_in_phase5": "authoritative page availability only",
            "not_created": [
                "medical rules", "safety triggers", "contraindications", "dosages",
                "treatment recommendations",
            ],
        },
    }
    (options.output_dir / "phase5_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(options.report, candidates, access_audit, selection)
    print("Phase 5 recommendations generated; final test remained sealed.")
    print("Recommended for approval:", ", ".join(recommended["display_name"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
