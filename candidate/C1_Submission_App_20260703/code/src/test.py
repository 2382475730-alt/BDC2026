from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_OUTPUT = "output/result.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the official output/result.csv file.")
    parser.add_argument("--work-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def _resolve_path(value: str | Path, work_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return work_dir / path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = Path(args.work_dir).resolve()
    output_path = _resolve_path(args.output, work_dir)
    default_output = work_dir / DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    default_output.parent.mkdir(parents=True, exist_ok=True)

    src_dir = work_dir / "code" / "src"
    sys.path.insert(0, str(src_dir))
    os.chdir(work_dir)

    import predict  # noqa: PLC0415

    mp.set_start_method("spawn", force=True)
    predict.main()

    if output_path.resolve() != default_output.resolve():
        shutil.copyfile(default_output, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
