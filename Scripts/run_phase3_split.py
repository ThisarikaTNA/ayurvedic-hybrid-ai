"""Create or validate the locked Phase 3 disease-group-disjoint split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from models.splitting import (
    SplitConfig,
    build_manifest,
    create_grouped_multilabel_split,
    file_sha256,
    verify_assignments,
    write_or_validate_locked_manifest,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--input", type=Path, default=Path("data/processed/ayurgenix_cleaned.csv")
    )
    command.add_argument("--output-dir", type=Path, default=Path("outputs/splits"))
    command.add_argument("--random-state", type=int, default=42)
    command.add_argument("--candidate-count", type=int, default=5000)
    return command


def main(arguments: list[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    dataframe = pd.read_csv(options.input, dtype="string", encoding="utf-8")
    config = SplitConfig(
        random_state=options.random_state, candidate_count=options.candidate_count
    )
    assignments, search_report = create_grouped_multilabel_split(dataframe, config)
    verify_assignments(dataframe, assignments)
    manifest = build_manifest(options.input, assignments, search_report)

    options.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = options.output_dir / "split_manifest.json"
    created = write_or_validate_locked_manifest(manifest_path, manifest)
    assignments.to_csv(options.output_dir / "split_assignments.csv", index=False)
    (options.output_dir / "split_report.json").write_text(
        json.dumps(search_report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Cleaned dataset SHA-256: {file_sha256(options.input)}")
    print(f"Locked manifest: {'created' if created else 'validated and reused'}")
    for partition, summary in search_report["summary"].items():
        print(
            f"{partition}: {summary['profile_count']} profiles, "
            f"{summary['disease_group_count']} disease groups, "
            f"labels={summary['positive_label_counts']}"
        )
    print("Disease-group overlaps: 0")
    print("No preprocessing estimator or model was fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
