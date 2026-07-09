from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_json, write_json


def upsert_experiment(registry_path: str | Path, entry: dict[str, Any]) -> dict[str, Any]:
    registry = read_json(registry_path, default=None)
    if not registry:
        registry = {"family": "D-Series Market Intelligence Ranking System", "updated": "", "experiments": []}
    experiments = registry.setdefault("experiments", [])
    experiment_id = entry.get("id")
    if not experiment_id:
        raise ValueError("registry entry requires id")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {**entry, "registry_updated_at": now}
    for index, existing in enumerate(experiments):
        if existing.get("id") == experiment_id:
            experiments[index] = {**existing, **entry}
            break
    else:
        experiments.append(entry)
    registry["updated"] = now[:10]
    write_json(registry_path, registry)
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert a D-series experiment registry entry.")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--entry", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upsert_experiment(args.registry, read_json(args.entry, default={}) or {})


if __name__ == "__main__":
    main()

