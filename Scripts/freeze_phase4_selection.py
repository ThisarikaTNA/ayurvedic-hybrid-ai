"""Freeze the user-accepted Phase 4 model decision before Phase 5."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.selection_manifest import (
    build_selection_manifest,
    write_or_validate_frozen_manifest,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--split-manifest", type=Path, default=Path("outputs/splits/split_manifest.json")
    )
    command.add_argument(
        "--preprocessing-output", type=Path,
        default=Path("data/processed/ayurgenix_cleaned.csv"),
    )
    command.add_argument(
        "--model-bundle", type=Path,
        default=Path("models/phase4_validation/provisional_best_bundle.joblib"),
    )
    command.add_argument(
        "--output", type=Path,
        default=Path("outputs/phase4_validation/model_selection_manifest.json"),
    )
    return command


def main(arguments: list[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    payload = build_selection_manifest(
        options.split_manifest, options.preprocessing_output, options.model_bundle
    )
    created = write_or_validate_frozen_manifest(options.output, payload)
    print(f"Model-selection manifest: {'created and frozen' if created else 'validated and unchanged'}")
    print(f"Path: {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
