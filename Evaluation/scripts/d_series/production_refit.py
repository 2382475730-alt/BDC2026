from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import read_json, write_json


def build_production_refit_manifest(
    *,
    experiment_id: str,
    gate_decision: dict[str, Any],
    split_metadata: dict[str, Any],
    champion_config: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    accepted = gate_decision.get("decision") == "ACCEPT_AS_CHAMPION"
    production = split_metadata.get("production_refit", {})
    manifest = {
        "experiment_id": experiment_id,
        "status": "planned" if accepted else "skipped",
        "reason": "Champion Gate accepted" if accepted else f"Champion Gate decision was {gate_decision.get('decision')}",
        "champion_config": str(champion_config),
        "raw_data_end": production.get("raw_data_end"),
        "train_sample_end": production.get("train_sample_end"),
        "feature_end_for_inference": production.get("feature_end_for_inference"),
        "rules": {
            "no_validation_set": True,
            "no_early_stopping": True,
            "no_hyperparameter_search": True,
            "fixed_epoch_from_champion_program": True,
        },
    }
    write_json(output_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create D-series production refit manifest.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gate-decision", required=True)
    parser.add_argument("--split-metadata", required=True)
    parser.add_argument("--champion-config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_production_refit_manifest(
        experiment_id=args.experiment_id,
        gate_decision=read_json(args.gate_decision, default={}) or {},
        split_metadata=read_json(args.split_metadata, default={}) or {},
        champion_config=args.champion_config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

