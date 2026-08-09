"""Strict validation and safe tokenization for standalone Phase 8 retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "retrieval_config.v1.json"
ALLOWED_CATEGORIES: tuple[str, ...] = (
    "profiles", "claims", "symptoms", "recommendations", "sources"
)
DOSHA_ORDER: tuple[str, ...] = ("Vata", "Pitta", "Kapha")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
DOSAGE_LIKE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|milligrams?|g|grams?|mcg|µg|ml|"
    r"millilit(?:er|re)s?|tablets?|capsules?|drops?)\b",
    flags=re.IGNORECASE,
)


class RetrievalInputError(ValueError):
    """Raised when a retrieval request violates the Phase 8 allowlist."""


@dataclass(frozen=True)
class ValidatedRetrievalQuery:
    """Normalized, immutable retrieval input."""

    condition_context: str | None
    free_text: str | None
    query_terms: tuple[str, ...]
    categories: tuple[str, ...]
    caller_supplied_dosha_tags: tuple[str, ...]
    top_k: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_context": self.condition_context,
            "free_text": self.free_text,
            "query_terms": list(self.query_terms),
            "categories": list(self.categories),
            "caller_supplied_dosha_tags": list(self.caller_supplied_dosha_tags),
            "top_k": self.top_k,
        }


def load_retrieval_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the versioned ranking configuration."""

    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("config_version") != "1.0.0":
        raise RetrievalInputError("Unsupported retrieval configuration version.")
    lexical = config.get("lexical_scoring", {})
    if abs(
        float(lexical.get("bm25_normalized_weight", -1))
        + float(lexical.get("matched_term_coverage_weight", -1))
        - 1.0
    ) > 1e-9:
        raise RetrievalInputError("Lexical scoring weights must sum to one.")
    for mode, weights in config.get("profile_ranking", {}).items():
        if mode == "no_optional_signals":
            continue
        if abs(
            float(weights.get("lexical_relevance_weight", -1))
            + float(weights.get("caller_supplied_dosha_overlap_weight", -1))
            - 1.0
        ) > 1e-9:
            raise RetrievalInputError(f"Profile weights for {mode} must sum to one.")
    return config


def tokenize_query(text: str, *, maximum_terms: int) -> tuple[str, ...]:
    """Return unique lowercase alphanumeric terms safe for quoted FTS usage."""

    terms: list[str] = []
    for token in TOKEN_PATTERN.findall(text.casefold()):
        if token not in terms:
            terms.append(token)
        if len(terms) > maximum_terms:
            raise RetrievalInputError(
                f"Query contains more than the allowed {maximum_terms} unique terms."
            )
    return tuple(terms)


def build_safe_fts_query(terms: Sequence[str]) -> str | None:
    """Build FTS syntax only from validated alphanumeric tokens."""

    if not terms:
        return None
    if any(not re.fullmatch(r"[a-z0-9]+", term) for term in terms):
        raise RetrievalInputError("Unsafe FTS token encountered.")
    return " OR ".join(f'"{term}"' for term in terms)


def _validate_categories(value: Any) -> tuple[str, ...]:
    if value is None:
        return ALLOWED_CATEGORIES
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RetrievalInputError("categories must be a list of approved category names.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RetrievalInputError("Every category must be text.")
        category = item.strip().casefold()
        if category not in ALLOWED_CATEGORIES:
            raise RetrievalInputError(f"Unknown retrieval category: {item!r}")
        if category not in normalized:
            normalized.append(category)
    if not normalized:
        raise RetrievalInputError("At least one retrieval category is required.")
    return tuple(category for category in ALLOWED_CATEGORIES if category in normalized)


def _validate_doshas(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RetrievalInputError("caller_supplied_dosha_tags must be a list.")
    mapping = {tag.casefold(): tag for tag in DOSHA_ORDER}
    normalized: set[str] = set()
    for item in value:
        if not isinstance(item, str) or item.strip().casefold() not in mapping:
            raise RetrievalInputError(
                "Only Vata, Pitta and Kapha are allowed as caller-supplied Dosha tags."
            )
        normalized.add(mapping[item.strip().casefold()])
    return tuple(tag for tag in DOSHA_ORDER if tag in normalized)


def validate_retrieval_query(
    request: Mapping[str, Any], config: Mapping[str, Any]
) -> ValidatedRetrievalQuery:
    """Reject unknown fields and normalize a Phase 8 retrieval request."""

    if not isinstance(request, Mapping):
        raise RetrievalInputError("Retrieval request must be an object.")
    allowed = {
        "condition", "free_text", "categories", "caller_supplied_dosha_tags", "top_k"
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise RetrievalInputError(f"Unknown retrieval input fields: {unknown}")

    limits = config["input_limits"]
    condition_value = request.get("condition")
    if condition_value is None or (
        isinstance(condition_value, str) and not condition_value.strip()
    ):
        condition = None
    elif not isinstance(condition_value, str):
        raise RetrievalInputError("condition must be text.")
    else:
        condition = condition_value.strip()
        if len(condition) > int(limits["condition_characters"]):
            raise RetrievalInputError("Condition input is too long.")
        if CONTROL_PATTERN.search(condition):
            raise RetrievalInputError("Condition input contains control characters.")

    free_text_value = request.get("free_text")
    if free_text_value is None or (
        isinstance(free_text_value, str) and not free_text_value.strip()
    ):
        free_text = None
        terms: tuple[str, ...] = ()
    elif not isinstance(free_text_value, str):
        raise RetrievalInputError("free_text must be text.")
    else:
        free_text = free_text_value.strip()
        if len(free_text) > int(limits["query_characters"]):
            raise RetrievalInputError("Free-text query is too long.")
        if CONTROL_PATTERN.search(free_text):
            raise RetrievalInputError("Free-text query contains control characters.")
        if DOSAGE_LIKE_PATTERN.search(free_text):
            raise RetrievalInputError(
                "Exact dosage-like text is outside the Phase 8 retrieval scope."
            )
        terms = tokenize_query(free_text, maximum_terms=int(limits["query_terms"]))

    top_k = request.get("top_k", config["top_k"]["default"])
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise RetrievalInputError("top_k must be an integer.")
    if not int(config["top_k"]["minimum"]) <= top_k <= int(config["top_k"]["maximum"]):
        raise RetrievalInputError(
            f"top_k must be between {config['top_k']['minimum']} and "
            f"{config['top_k']['maximum']}."
        )

    return ValidatedRetrievalQuery(
        condition_context=condition,
        free_text=free_text,
        query_terms=terms,
        categories=_validate_categories(request.get("categories")),
        caller_supplied_dosha_tags=_validate_doshas(
            request.get("caller_supplied_dosha_tags")
        ),
        top_k=top_k,
    )
