"""Generate the Phase 2 cleaned dataset and audit outputs without modelling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.preprocessing import (
    EXPECTED_SOURCE_SHA256,
    build_report,
    file_sha256,
    leakage_screen,
    load_source,
    preprocess_dataframe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/AyurGenixAI_Dataset.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/ayurgenix_cleaned.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/processed/preprocessing_report.json")
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    digest = file_sha256(options.input)
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"Source hash changed: expected {EXPECTED_SOURCE_SHA256}, observed {digest}"
        )

    source = load_source(options.input)
    cleaned = preprocess_dataframe(source)
    report = build_report(source, cleaned, options.input)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(options.output, index=False, encoding="utf-8")
    options.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    leakage_rows = pd.DataFrame(leakage_screen(cleaned)["candidate_feature_screen"])
    leakage_rows.to_csv(
        options.report.with_name("leakage_screen.csv"), index=False, encoding="utf-8"
    )
    pd.DataFrame(
        {
            "column": list(report["missing_and_placeholders"]["placeholder_counts_by_clean_column"]),
            "placeholder_count": list(report["missing_and_placeholders"]["placeholder_counts_by_clean_column"].values()),
        }
    ).to_csv(options.report.with_name("missing_values.csv"), index=False, encoding="utf-8")

    print(f"Cleaned knowledge profiles: {cleaned.shape[0]} rows x {cleaned.shape[1]} columns")
    print(f"Cleaned CSV: {options.output}")
    print(f"Preprocessing report: {options.report}")
    print("No data split or model training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
