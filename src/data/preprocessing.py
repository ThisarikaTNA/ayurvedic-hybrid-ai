"""Auditable Phase 2 preprocessing for AyurGenixAI knowledge profiles.

The module preserves source values, creates separate cleaned columns, validates
dataset-assigned Dosha tags, and screens candidate model features for leakage.
It does not split data, fit preprocessing estimators, or train a model.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


EXPECTED_SOURCE_SHA256 = (
    "6B722B8784E5947B762D93A0F1A6E86650CA7AF27BBE21A95AB0C84AFD4F37D5"
)
DOSHA_ORDER: tuple[str, ...] = ("vata", "pitta", "kapha")
DOSHA_DISPLAY: Mapping[str, str] = {
    "vata": "Vata",
    "pitta": "Pitta",
    "kapha": "Kapha",
}

EXPECTED_COLUMNS: tuple[str, ...] = (
    "Disease", "Hindi Name", "Marathi Name", "Symptoms", "Diagnosis & Tests",
    "Symptom Severity", "Duration of Treatment", "Medical History",
    "Current Medications", "Risk Factors", "Environmental Factors",
    "Sleep Patterns", "Stress Levels", "Physical Activity Levels",
    "Family History", "Dietary Habits", "Allergies (Food/Env)",
    "Seasonal Variation", "Age Group", "Gender",
    "Occupation and Lifestyle", "Cultural Preferences",
    "Herbal/Alternative Remedies", "Ayurvedic Herbs", "Formulation", "Doshas",
    "Constitution/Prakriti", "Diet and Lifestyle Recommendations",
    "Yoga & Physical Therapy", "Medical Intervention", "Prevention", "Prognosis",
    "Complications", "Patient Recommendations",
)

MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "symptoms", "symptom_severity", "age_group", "gender", "risk_factors",
    "environmental_factors", "sleep_patterns", "stress_levels",
    "physical_activity_levels", "dietary_habits", "seasonal_variation",
    "occupation_and_lifestyle",
)

# These fields are never eligible as model inputs in the initial experiment.
FORBIDDEN_MODEL_COLUMNS: frozenset[str] = frozenset(
    {
        "disease", "hindi_name", "marathi_name", "diagnosis_and_tests",
        "duration_of_treatment", "medical_history", "current_medications",
        "family_history", "allergies_food_env", "cultural_preferences",
        "herbal_alternative_remedies", "ayurvedic_herbs", "formulation", "doshas",
        "constitution_prakriti", "diet_and_lifestyle_recommendations",
        "yoga_and_physical_therapy", "medical_intervention", "prevention",
        "prognosis", "complications", "patient_recommendations",
        "dosha_vata", "dosha_pitta", "dosha_kapha", "normalized_disease",
    }
)

PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "", "-", "--", "?", "n/a", "na", "nil", "none", "null", "unknown",
        "not applicable", "not available", "not known", "not specified",
        "not reported", "not provided", "no data", "no information",
        "no information available",
    }
)

# Only observed, conservative presentation/spelling variants are collapsed.
CATEGORY_VARIANTS: Mapping[str, Mapping[str, str]] = {
    "gender": {"both genders": "all genders", "males": "male"},
    "physical_activity_levels": {
        "moderate to low": "low to moderate",
    },
}


class PreprocessingError(ValueError):
    """Raised when the source or preprocessing policy fails validation."""


def file_sha256(path: Path) -> str:
    """Calculate an uppercase SHA-256 digest without modifying the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def standardize_column_name(name: str) -> str:
    """Convert a source heading into a stable snake-case identifier."""

    text = unicodedata.normalize("NFKC", name).strip().casefold()
    text = text.replace("&", " and ").replace("/", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_apostrophes(text: str) -> str:
    """Convert curly and modifier apostrophes to the ASCII apostrophe."""

    return text.translate(str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "`": "'"}))


def normalized_key(value: Any) -> str:
    """Return a comparison key for placeholders and categorical values."""

    if pd.isna(value):
        return ""
    text = normalize_apostrophes(unicodedata.normalize("NFKC", str(value)))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    text = re.sub(r"\s*([,;/|])\s*", r"\1", text)
    return text


def clean_text(value: Any, *, categorical: bool = False) -> Any:
    """Clean text while converting ambiguous placeholders to ``pd.NA``.

    Explicit negative phrases such as ``No known allergies`` are retained.
    The raw source value remains available in a separate ``raw_`` column.
    """

    key = normalized_key(value)
    if key in PLACEHOLDERS:
        return pd.NA
    text = normalize_apostrophes(unicodedata.normalize("NFKC", str(value)))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([,;/|])\s*", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold() if categorical else text


def normalize_disease_name(value: Any) -> Any:
    """Normalize disease names only for grouping and future data splitting."""

    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return pd.NA
    text = str(cleaned).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_dosha(value: Any) -> tuple[str, ...]:
    """Parse and validate an unordered, possibly multi-label Dosha value."""

    if pd.isna(value) or not str(value).strip():
        raise PreprocessingError("Doshas contains a missing or empty value.")
    tokens = [
        normalized_key(token)
        for token in re.split(r"[,;/|+&-]+", str(value))
        if normalized_key(token)
    ]
    unknown = sorted(set(tokens) - set(DOSHA_ORDER))
    if unknown:
        raise PreprocessingError(f"Unknown Dosha label(s): {unknown}")
    if not tokens:
        raise PreprocessingError("Doshas contains no recognized label.")
    return tuple(label for label in DOSHA_ORDER if label in set(tokens))


def canonical_dosha(value: Any) -> str:
    """Return labels in stable Vata, Pitta, Kapha order."""

    return ", ".join(DOSHA_DISPLAY[label] for label in parse_dosha(value))


def tokenize_symptoms(value: Any) -> tuple[str, ...]:
    """Tokenize the observed comma-led symptom-list format conservatively."""

    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return ()
    items = re.split(r"[,;/|]+", str(cleaned))
    tokens: list[str] = []
    for item in items:
        token = re.sub(r"\s+", " ", item).strip().casefold().strip(" .:-")
        if token and token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def validate_feature_policy(feature_columns: Iterable[str]) -> None:
    """Fail fast if target, disease, or post-diagnosis fields enter features."""

    supplied = tuple(feature_columns)
    forbidden = sorted(set(supplied) & FORBIDDEN_MODEL_COLUMNS)
    unknown = sorted(set(supplied) - set(MODEL_FEATURE_COLUMNS))
    if forbidden or unknown:
        raise PreprocessingError(
            f"Unsafe feature policy. Forbidden={forbidden}; not allowlisted={unknown}"
        )


def load_source(path: Path) -> pd.DataFrame:
    """Load CSV or Excel safely as strings without automatic NA assumptions."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        dataframe = pd.read_csv(
            path, dtype="string", keep_default_na=False, encoding="utf-8-sig"
        )
    elif suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path, dtype="string", keep_default_na=False)
    else:
        raise PreprocessingError(f"Unsupported input format: {path.suffix}")

    actual = tuple(map(str, dataframe.columns))
    if actual != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(actual))
        unexpected = sorted(set(actual) - set(EXPECTED_COLUMNS))
        raise PreprocessingError(
            f"Unexpected schema or column order. Missing={missing}; unexpected={unexpected}"
        )
    return dataframe


def preprocess_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned table while preserving each source value separately."""

    actual = tuple(map(str, dataframe.columns))
    if actual != EXPECTED_COLUMNS:
        raise PreprocessingError("Dataframe columns do not match the expected source schema.")
    validate_feature_policy(MODEL_FEATURE_COLUMNS)

    output = pd.DataFrame(index=dataframe.index)
    output["knowledge_profile_id"] = [f"kp_{index + 1:04d}" for index in range(len(dataframe))]
    missing_columns_by_row: list[list[str]] = [[] for _ in range(len(dataframe))]
    categorical = {
        "symptom_severity", "sleep_patterns", "stress_levels",
        "physical_activity_levels", "age_group", "gender", "seasonal_variation",
    }

    for source_name in EXPECTED_COLUMNS:
        clean_name = standardize_column_name(source_name)
        output[f"raw_{clean_name}"] = dataframe[source_name].astype("string")
        cleaned = dataframe[source_name].map(
            lambda value: clean_text(value, categorical=clean_name in categorical)
        ).astype("string")
        if clean_name in CATEGORY_VARIANTS:
            cleaned = cleaned.replace(CATEGORY_VARIANTS[clean_name])
        output[clean_name] = cleaned
        for position, is_missing in enumerate(cleaned.isna().tolist()):
            if is_missing:
                missing_columns_by_row[position].append(clean_name)

    output["normalized_disease"] = dataframe["Disease"].map(normalize_disease_name).astype("string")
    output["symptoms"] = dataframe["Symptoms"].map(clean_text).astype("string")
    output["symptom_tokens"] = dataframe["Symptoms"].map(
        lambda value: json.dumps(tokenize_symptoms(value), ensure_ascii=False)
    )
    output["symptom_token_count"] = dataframe["Symptoms"].map(
        lambda value: len(tokenize_symptoms(value))
    )
    output["doshas"] = dataframe["Doshas"].map(canonical_dosha).astype("string")
    parsed = dataframe["Doshas"].map(parse_dosha)
    for label in DOSHA_ORDER:
        output[f"dosha_{label}"] = parsed.map(lambda labels: int(label in labels))
    output["placeholder_missing_columns"] = [
        json.dumps(columns) for columns in missing_columns_by_row
    ]
    return output


def _target_combo(row: pd.Series) -> str:
    return "+".join(
        label for label in DOSHA_ORDER if int(row[f"dosha_{label}"]) == 1
    )


def leakage_screen(cleaned: pd.DataFrame) -> dict[str, Any]:
    """Screen direct answer terms and indirect disease/target proxies."""

    validate_feature_policy(MODEL_FEATURE_COLUMNS)
    dosha_pattern = re.compile(r"\b(?:vata|pitta|kapha)\b", re.IGNORECASE)
    rows: list[dict[str, Any]] = []
    combos = cleaned.apply(_target_combo, axis=1)

    for column in MODEL_FEATURE_COLUMNS:
        values = cleaned[column].fillna("").astype(str)
        direct_rows = [int(i) for i, value in values.items() if dosha_pattern.search(value)]
        disease_rows: list[int] = []
        for index, (disease, value) in enumerate(
            zip(cleaned["normalized_disease"].fillna(""), values, strict=True)
        ):
            disease_key = str(disease)
            value_key = normalize_disease_name(value)
            if disease_key and len(disease_key) >= 4 and disease_key in str(value_key):
                disease_rows.append(index)

        repeated = pd.DataFrame({"value": values, "target": combos})
        repeated = repeated[repeated["value"] != ""]
        counts = repeated["value"].value_counts()
        repeated = repeated[repeated["value"].isin(counts[counts >= 2].index)]
        if repeated.empty:
            purity = None
        else:
            majority = repeated.groupby("value")["target"].value_counts().groupby(level=0).max()
            purity = round(float(majority.sum() / len(repeated)), 4)
        rows.append(
            {
                "column": column,
                "direct_dosha_term_count": len(direct_rows),
                "direct_dosha_term_row_indices": direct_rows,
                "disease_name_overlap_count": len(disease_rows),
                "disease_name_overlap_row_indices": disease_rows,
                "unique_value_count": int(values.nunique()),
                "unique_value_ratio": round(float(values.nunique() / len(values)), 4),
                "repeated_value_target_majority_purity": purity,
            }
        )

    family_values = cleaned["family_history"].fillna("").astype(str)
    family_overlap = sum(
        bool(disease) and len(str(disease)) >= 4 and str(disease) in str(normalize_disease_name(value))
        for disease, value in zip(cleaned["normalized_disease"].fillna(""), family_values, strict=True)
    )
    medical_values = cleaned["medical_history"].fillna("").astype(str)
    medical_overlap = sum(
        bool(disease) and len(str(disease)) >= 4 and str(disease) in str(normalize_disease_name(value))
        for disease, value in zip(cleaned["normalized_disease"].fillna(""), medical_values, strict=True)
    )
    return {
        "feature_policy_passed": True,
        "allowed_features": list(MODEL_FEATURE_COLUMNS),
        "forbidden_features": sorted(FORBIDDEN_MODEL_COLUMNS),
        "candidate_feature_screen": rows,
        "excluded_field_checks": {
            "family_history_disease_name_overlap_count": int(family_overlap),
            "medical_history_disease_name_overlap_count": int(medical_overlap),
        },
        "interpretation": (
            "String overlap and repeated-value purity are screening heuristics, not proof "
            "that a field is safe or causal. High-cardinality and templated fields require review."
        ),
    }


def build_report(source: pd.DataFrame, cleaned: pd.DataFrame, source_path: Path) -> dict[str, Any]:
    """Build the complete machine-readable preprocessing and audit report."""

    placeholder_counts: dict[str, int] = {}
    for column in EXPECTED_COLUMNS:
        count = int(source[column].map(lambda value: normalized_key(value) in PLACEHOLDERS).sum())
        if count:
            placeholder_counts[standardize_column_name(column)] = count

    normalized_duplicates = cleaned[
        [standardize_column_name(column) for column in EXPECTED_COLUMNS]
    ].duplicated()
    disease_counts = cleaned["normalized_disease"].value_counts()
    target_counts = {label: int(cleaned[f"dosha_{label}"].sum()) for label in DOSHA_ORDER}
    combination_counts = Counter(cleaned.apply(_target_combo, axis=1))
    category_changes: dict[str, dict[str, int]] = {}
    for column, mappings in CATEGORY_VARIANTS.items():
        raw_keys = source[EXPECTED_COLUMNS[
            [standardize_column_name(name) for name in EXPECTED_COLUMNS].index(column)
        ]].map(normalized_key)
        changes = {f"{old} -> {new}": int((raw_keys == old).sum()) for old, new in mappings.items()}
        category_changes[column] = {key: value for key, value in changes.items() if value}

    return {
        "phase": 2,
        "scope": "audit and preprocessing only; no split or model training",
        "source": {
            "path": str(Path(source_path).resolve()),
            "sha256": file_sha256(source_path),
            "rows": int(len(source)),
            "columns": int(len(source.columns)),
            "encoding": "utf-8-sig" if Path(source_path).suffix.casefold() == ".csv" else None,
        },
        "column_mapping": {
            name: standardize_column_name(name) for name in EXPECTED_COLUMNS
        },
        "missing_and_placeholders": {
            "true_missing_cells_in_source": int(source.isna().sum().sum()),
            "placeholder_counts_by_clean_column": placeholder_counts,
            "policy": (
                "Placeholders become pd.NA only in cleaned columns. Raw values are preserved. "
                "'None' is not interpreted as an explicit negative or as no allergy."
            ),
        },
        "duplicates": {
            "exact_duplicate_extra_rows": int(source.duplicated().sum()),
            "normalized_duplicate_extra_rows": int(normalized_duplicates.sum()),
            "normalized_disease_count": int(cleaned["normalized_disease"].nunique()),
            "repeated_disease_group_count": int((disease_counts > 1).sum()),
            "maximum_profiles_per_disease": int(disease_counts.max()),
        },
        "symptoms": {
            "delimiter_inspection": {
                "rows_with_comma": int(source["Symptoms"].str.contains(",", regex=False).sum()),
                "rows_with_slash": int(source["Symptoms"].str.contains("/", regex=False).sum()),
                "rows_with_semicolon": int(source["Symptoms"].str.contains(";", regex=False).sum()),
            },
            "token_count_distribution": {
                str(key): int(value)
                for key, value in cleaned["symptom_token_count"].value_counts().sort_index().items()
            },
            "policy": "Split on comma, semicolon, slash, or vertical bar; casefold and deduplicate per profile.",
        },
        "dosha": {
            "allowed_labels": list(DOSHA_ORDER),
            "invalid_value_count": 0,
            "per_label_positive_counts": target_counts,
            "combination_counts": dict(sorted(combination_counts.items())),
        },
        "categorical_variant_changes": category_changes,
        "leakage": leakage_screen(cleaned),
        "model_feature_policy": {
            "allowed_initial_features": list(MODEL_FEATURE_COLUMNS),
            "excluded_fields": sorted(FORBIDDEN_MODEL_COLUMNS),
            "medical_history_status": "excluded pending manual leakage review",
            "family_history_status": "excluded because disease-name leakage is frequent",
        },
        "limitations": [
            "Rows are knowledge profiles, not verified independent patient cases.",
            "Dataset provenance, medical accuracy, and labeling process are unverified.",
            "String-based leakage tests cannot detect every semantic or templated proxy.",
            "Cleaning does not make dataset-derived Dosha tags clinically valid.",
        ],
    }
