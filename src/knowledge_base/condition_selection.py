"""Reproducible Phase 5 condition scoring on unsealed knowledge profiles only.

Scores measure coursework suitability and data availability. They do not
measure medical correctness, disease importance, or clinical evidence.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data.preprocessing import PLACEHOLDERS, normalize_disease_name, normalized_key


RELEVANT_FIELDS: tuple[str, ...] = (
    "symptoms",
    "doshas",
    "diet_and_lifestyle_recommendations",
    "prevention",
    "complications",
)

WEIGHTS: Mapping[str, float] = {
    "symptoms_completeness_score": 0.10,
    "dosha_clarity_score": 0.12,
    "diet_lifestyle_completeness_score": 0.10,
    "prevention_completeness_score": 0.08,
    "complications_referral_completeness_score": 0.10,
    "distinctiveness_score": 0.10,
    "rule_potential_score": 0.12,
    "external_source_availability_score": 0.13,
    "manageable_scope_score": 0.15,
}


CANDIDATE_CONFIG: Mapping[str, Mapping[str, Any]] = {
    "acne": {
        "category": "dermatology",
        "manageable_scope_score": 1.0,
        "recommended": True,
        "source_title": "NHS: Acne",
        "source_url": "https://www.nhs.uk/conditions/acne/",
        "safety_opportunity": (
            "Later source review could define severity, scarring, infection, and "
            "appropriate-care escalation topics; no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": (
            "Dataset skin-care and trigger text can nominate non-dosage lifestyle-rule "
            "topics for later source checking."
        ),
        "reason": "Manageable scope, clear dataset Dosha tag, distinct skin-care demonstration.",
        "manual_limitations": "Only one unsealed profile; dataset claims remain unverified.",
    },
    "osteoarthritis": {
        "category": "musculoskeletal",
        "manageable_scope_score": 0.9,
        "recommended": True,
        "source_title": "NHS: Osteoarthritis",
        "source_url": "https://www.nhs.uk/conditions/osteoarthritis/",
        "safety_opportunity": (
            "Later source review could define persistent or function-limiting symptom and "
            "professional-assessment topics; no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": (
            "Dataset activity and lifestyle text can nominate general movement and daily-"
            "living topics for later source checking."
        ),
        "reason": "Distinct musculoskeletal example with explainable lifestyle and referral structure.",
        "manual_limitations": "Only one unsealed profile; condition severity and affected joint vary.",
    },
    "insomnia": {
        "category": "sleep",
        "manageable_scope_score": 0.9,
        "recommended": True,
        "source_title": "NHS: Insomnia",
        "source_url": "https://www.nhs.uk/conditions/insomnia/",
        "safety_opportunity": (
            "Later source review could define persistence, daytime impairment, and other-"
            "sleep-disorder referral topics; no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": (
            "Dataset sleep-routine text can nominate general sleep-habit rule topics for "
            "later source checking."
        ),
        "reason": "Distinct sleep-domain demonstration with non-dosage lifestyle-rule potential.",
        "manual_limitations": "Two profiles have different dataset Dosha combinations.",
    },
    "gastroesophageal reflux disease gerd": {
        "category": "digestive",
        "manageable_scope_score": 0.9,
        "recommended": True,
        "source_title": "NHS: Heartburn and acid reflux",
        "source_url": "https://www.nhs.uk/conditions/heartburn-and-acid-reflux/",
        "safety_opportunity": (
            "Later source review could define persistent, worsening, or alarm-feature "
            "assessment topics; no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": (
            "Dataset meal-pattern and trigger text can nominate general lifestyle-rule "
            "topics for later source checking."
        ),
        "reason": "Distinct digestive example with clear dataset tag and general lifestyle structure.",
        "manual_limitations": "Only one unsealed profile; chest symptoms require careful differential safety design.",
    },
    "common cold": {
        "category": "respiratory_infectious",
        "manageable_scope_score": 1.0,
        "recommended": True,
        "source_title": "NHS: Common cold",
        "source_url": "https://www.nhs.uk/conditions/common-cold/",
        "safety_opportunity": (
            "Later source review could define worsening, prolonged, or breathing-related "
            "assessment topics; no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": (
            "Dataset rest, hygiene, and general self-care text can nominate non-dosage "
            "rule topics for later source checking."
        ),
        "reason": "Manageable demonstration scope with clear separation between general support and escalation.",
        "manual_limitations": "Only one unsealed profile; symptom overlap with other infections is substantial.",
    },
    "psoriasis": {
        "category": "dermatology",
        "manageable_scope_score": 0.7,
        "recommended": False,
        "source_title": "NHS: Psoriasis",
        "source_url": "https://www.nhs.uk/conditions/psoriasis/",
        "safety_opportunity": (
            "Authoritative material supports later review of complications and urgent "
            "presentations; no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": "Dataset trigger and skin-care topics could be reviewed later.",
        "reason": "Strong candidate, but excluded provisionally to reduce dermatology redundancy.",
        "manual_limitations": "Chronic immune-mediated disease with potentially serious presentations and treatment complexity.",
    },
    "eczema atopic dermatitis": {
        "category": "dermatology",
        "manageable_scope_score": 0.7,
        "recommended": False,
        "source_title": "NHS: Atopic eczema",
        "source_url": "https://www.nhs.uk/conditions/atopic-eczema/",
        "safety_opportunity": (
            "Authoritative material supports later infection and urgent-assessment topic "
            "review; no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": "Dataset irritant and skin-care topics could be reviewed later.",
        "reason": "Useful candidate, but redundant with acne in a five-condition portfolio.",
        "manual_limitations": "Only one profile; age-dependent care and infection safety add complexity.",
    },
    "asthma": {
        "category": "respiratory_chronic",
        "manageable_scope_score": 0.25,
        "recommended": False,
        "source_title": "NHS: Asthma",
        "source_url": "https://www.nhs.uk/conditions/asthma/",
        "safety_opportunity": (
            "Authoritative material clearly supports later emergency-escalation review, "
            "but no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": "Dataset trigger-avoidance topics could be reviewed later.",
        "reason": "Excluded because emergency breathing assessment and medication context are high risk.",
        "manual_limitations": (
            "Conflicting dataset Dosha combinations and an apparent unrelated fracture-like "
            "profile indicate serious data-quality concerns."
        ),
    },
    "hypertension": {
        "category": "cardiovascular",
        "manageable_scope_score": 0.2,
        "recommended": False,
        "source_title": "WHO: Hypertension",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "safety_opportunity": (
            "Authoritative material supports later urgent-care and monitoring review, but "
            "no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": "Dataset lifestyle topics could be reviewed later with clinical safeguards.",
        "reason": "Excluded because diagnosis, monitoring, treatment, and escalation are high stakes.",
        "manual_limitations": "Conflicting dataset Dosha combinations; symptoms alone cannot establish hypertension.",
    },
    "diabetes": {
        "category": "metabolic_endocrine",
        "manageable_scope_score": 0.2,
        "recommended": False,
        "source_title": "WHO: Diabetes",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "safety_opportunity": (
            "Authoritative material supports later complication and professional-care "
            "review, but no trigger or rule is defined here."
        ),
        "lifestyle_opportunity": "Dataset lifestyle topics could be reviewed later without prescribing output.",
        "reason": "Excluded because monitoring, medication, complications, and disease subtype are high risk.",
        "manual_limitations": "Generic disease label may combine contexts; dataset medical content is unverified.",
    },
}


def is_missing(value: Any) -> bool:
    """Treat cleaned blanks and known placeholders as missing, not as negatives."""

    if pd.isna(value):
        return True
    return normalized_key(value) in PLACEHOLDERS


def normalize_condition_name(value: Any) -> str:
    """Normalize spelling, apostrophes, whitespace, and punctuation for grouping."""

    normalized = normalize_disease_name(value)
    return "" if pd.isna(normalized) else str(normalized)


def load_unsealed_profiles(
    data_path: Path, assignment_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retain train/validation knowledge only and discard sealed rows immediately."""

    assignments = pd.read_csv(
        assignment_path,
        usecols=["knowledge_profile_id", "split"],
        dtype="string",
        keep_default_na=False,
    )
    if assignments["knowledge_profile_id"].duplicated().any():
        raise ValueError("Split assignments contain duplicate profile IDs.")
    allowed_assignments = assignments[
        assignments["split"].isin(["train", "validation"])
    ].copy()
    allowed_ids = set(allowed_assignments["knowledge_profile_id"])
    selected_columns = [
        "knowledge_profile_id", "disease", "normalized_disease", *RELEVANT_FIELDS
    ]
    rows: list[dict[str, str]] = []
    with Path(data_path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = sorted(set(selected_columns) - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Missing Phase 5 fields: {missing_columns}")
        for row in reader:
            profile_id = row["knowledge_profile_id"]
            if profile_id not in allowed_ids:
                continue
            retained = {column: row[column] for column in selected_columns}
            retained["normalized_disease"] = normalize_condition_name(row["disease"])
            rows.append(retained)
    dataframe = pd.DataFrame(rows, columns=selected_columns)
    if len(dataframe) != len(allowed_assignments):
        raise ValueError("Unsealed knowledge rows do not match locked assignments.")
    return dataframe, {
        "allowed_partitions": ["train", "validation"],
        "retained_profile_count": int(len(dataframe)),
        "retained_normalized_group_count": int(dataframe["normalized_disease"].nunique()),
        "final_test_rows_selected": 0,
        "final_test_fields_retained": 0,
        "final_test_metrics_computed": False,
    }


def _symptom_distinctiveness(group_documents: pd.Series) -> pd.Series:
    if len(group_documents) == 1:
        return pd.Series([1.0], index=group_documents.index)
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(
        group_documents.fillna("").astype(str)
    )
    similarities = cosine_similarity(matrix)
    np.fill_diagonal(similarities, -np.inf)
    maximum = similarities.max(axis=1)
    return pd.Series(np.clip(1.0 - maximum, 0.0, 1.0), index=group_documents.index)


def _within_group_symptom_coherence(values: pd.Series) -> float:
    documents = [str(value) for value in values if not is_missing(value)]
    if len(documents) <= 1:
        return 1.0
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(documents)
    similarities = cosine_similarity(matrix)
    upper = similarities[np.triu_indices_from(similarities, k=1)]
    return float(upper.mean()) if len(upper) else 1.0


def aggregate_condition_scores(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Score every unsealed normalized disease group deterministically."""

    required = {"knowledge_profile_id", "disease", "normalized_disease", *RELEVANT_FIELDS}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Condition scoring input is missing columns: {missing}")

    records: list[dict[str, Any]] = []
    grouped = dataframe.groupby("normalized_disease", sort=True, dropna=False)
    for condition, group in grouped:
        completeness = {
            field: float((~group[field].map(is_missing)).mean())
            for field in RELEVANT_FIELDS
        }
        dosha_values = [
            str(value) for value in group["doshas"] if not is_missing(value)
        ]
        counts = Counter(dosha_values)
        dosha_clarity = max(counts.values()) / len(dosha_values) if dosha_values else 0.0
        complete_fields = [
            field for field in RELEVANT_FIELDS if completeness[field] == 1.0
        ]
        missing_counts = {
            field: int(group[field].map(is_missing).sum())
            for field in RELEVANT_FIELDS
            if group[field].map(is_missing).any()
        }
        records.append(
            {
                "normalized_condition_name": str(condition),
                "display_name": str(group["disease"].iloc[0]),
                "profile_count": int(len(group)),
                "dosha_combinations_present": " | ".join(sorted(counts)),
                "relevant_fields_complete": json.dumps(complete_fields),
                "missing_or_placeholder_fields": json.dumps(missing_counts, sort_keys=True),
                "symptoms_completeness_score": completeness["symptoms"],
                "dosha_clarity_score": float(dosha_clarity),
                "diet_lifestyle_completeness_score": completeness[
                    "diet_and_lifestyle_recommendations"
                ],
                "prevention_completeness_score": completeness["prevention"],
                "complications_referral_completeness_score": completeness[
                    "complications"
                ],
                "within_group_symptom_coherence": _within_group_symptom_coherence(
                    group["symptoms"]
                ),
                "symptom_document": " || ".join(
                    str(value) for value in group["symptoms"] if not is_missing(value)
                ),
            }
        )
    scores = pd.DataFrame(records).set_index("normalized_condition_name", drop=False)
    scores["symptom_distinctiveness_score"] = _symptom_distinctiveness(
        scores["symptom_document"]
    )
    scores["distinctiveness_score"] = scores["symptom_distinctiveness_score"]
    scores["rule_potential_score"] = scores[
        [
            "symptoms_completeness_score",
            "dosha_clarity_score",
            "diet_lifestyle_completeness_score",
            "prevention_completeness_score",
            "complications_referral_completeness_score",
        ]
    ].mean(axis=1)
    scores["external_source_availability_status"] = "not yet verified"
    scores["external_source_availability_score"] = 0.0
    scores["manageable_scope_status"] = "not manually reviewed"
    scores["manageable_scope_score"] = 0.5
    scores["preliminary_data_score"] = sum(
        scores[component] * weight
        for component, weight in WEIGHTS.items()
    )
    scores["selection_score"] = scores["preliminary_data_score"]
    return scores.reset_index(drop=True)


def apply_candidate_review(scores: pd.DataFrame) -> pd.DataFrame:
    """Apply source verification, scope review, and portfolio diversity to ten candidates."""

    candidate_rows: list[pd.Series] = []
    recommended_categories = {
        str(config["category"])
        for config in CANDIDATE_CONFIG.values()
        if bool(config["recommended"])
    }
    for condition, config in CANDIDATE_CONFIG.items():
        matches = scores[scores["normalized_condition_name"] == condition]
        if len(matches) != 1:
            raise ValueError(
                f"Candidate {condition!r} is absent or non-unique in the unsealed groups."
            )
        row = matches.iloc[0].copy()
        category_distinctiveness = (
            1.0
            if bool(config["recommended"]) or str(config["category"]) not in recommended_categories
            else 0.4
        )
        row["portfolio_category"] = config["category"]
        row["portfolio_category_distinctiveness_score"] = category_distinctiveness
        row["distinctiveness_score"] = (
            float(row["symptom_distinctiveness_score"]) + category_distinctiveness
        ) / 2.0
        row["external_source_availability_status"] = (
            "verified: authoritative source page available; content not yet converted to rules"
        )
        row["external_source_availability_score"] = 1.0
        row["external_source_title"] = config["source_title"]
        row["external_source_url"] = config["source_url"]
        row["manageable_scope_status"] = (
            "reviewed for coursework scope; not a clinical-safety approval"
        )
        row["manageable_scope_score"] = float(config["manageable_scope_score"])
        row["possible_safety_referral_rule_opportunities"] = config[
            "safety_opportunity"
        ]
        row["possible_general_lifestyle_rule_opportunities"] = config[
            "lifestyle_opportunity"
        ]
        row["recommendation_status"] = (
            "recommended_for_user_approval"
            if bool(config["recommended"])
            else "not_recommended_in_current_five"
        )
        row["inclusion_or_exclusion_reason"] = config["reason"]
        automatic_limitations: list[str] = [
            "Dataset completeness is not evidence of medical correctness."
        ]
        if int(row["profile_count"]) == 1:
            automatic_limitations.append("Only one unsealed dataset profile is available.")
        if float(row["dosha_clarity_score"]) < 1.0:
            automatic_limitations.append(
                "The normalized disease group contains multiple dataset Dosha combinations."
            )
        if float(row["within_group_symptom_coherence"]) < 0.25:
            automatic_limitations.append(
                "Repeated profiles have low symptom coherence and require data-quality review."
            )
        automatic_limitations.append(str(config["manual_limitations"]))
        row["important_medical_or_data_limitations"] = " ".join(
            automatic_limitations
        )
        row["selection_score"] = sum(
            float(row[component]) * weight for component, weight in WEIGHTS.items()
        )
        candidate_rows.append(row)
    candidates = pd.DataFrame(candidate_rows)
    return candidates.sort_values(
        ["recommendation_status", "selection_score", "normalized_condition_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
