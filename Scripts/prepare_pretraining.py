"""Prepare locked partitions and run the validation-only dummy baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ayurvedic_hybrid_ai.data_audit import EXPECTED_SHEET, file_sha256
from ayurvedic_hybrid_ai.pretraining import (
    TARGET_DEFINITION,
    create_dataset_card,
    encode_dosha_targets,
    fit_training_only_dummy_baseline,
    lock_or_validate_partitions,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Create the locked group split and validation dummy baseline."
    )
    command.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/AyurGenixAI_Dataset.xlsx"),
    )
    command.add_argument(
        "--output", type=Path, default=Path("outputs/pretraining")
    )
    command.add_argument("--random-state", type=int, default=42)
    command.add_argument("--kaggle-url")
    command.add_argument("--kaggle-claimed-rows", type=int)
    command.add_argument("--kaggle-claimed-columns", type=int)
    return command


def main(arguments: list[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    workbook_path = options.input.resolve()
    dataframe = pd.read_excel(
        workbook_path, sheet_name=EXPECTED_SHEET, dtype=object, engine="openpyxl"
    )
    dataframe.index = pd.RangeIndex(len(dataframe))
    digest = file_sha256(workbook_path)
    targets = encode_dosha_targets(dataframe["Doshas"])

    manifest_path = options.output / "locked_split_manifest.json"
    partitions, manifest, created = lock_or_validate_partitions(
        manifest_path,
        dataframe,
        targets,
        source_sha256=digest,
        random_state=options.random_state,
    )
    _, _, baseline_report = fit_training_only_dummy_baseline(
        dataframe, targets, partitions, random_state=options.random_state
    )
    baseline_report["target_definition"] = TARGET_DEFINITION
    write_json(options.output / "dummy_validation_report.json", baseline_report)

    card = create_dataset_card(
        workbook_path,
        dataframe,
        kaggle_dataset_url=options.kaggle_url,
        kaggle_claimed_rows=options.kaggle_claimed_rows,
        kaggle_claimed_columns=options.kaggle_claimed_columns,
    )
    Path("docs/dataset_card.md").write_text(card, encoding="utf-8")

    print("Pre-training preparation complete.")
    print(f"Target definition: {TARGET_DEFINITION}")
    print(f"Split manifest: {'created and locked' if created else 'validated and reused'}")
    print(f"Partition summary: {manifest['summary']}")
    print("Dummy baseline evaluated on validation only; final test was not accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

