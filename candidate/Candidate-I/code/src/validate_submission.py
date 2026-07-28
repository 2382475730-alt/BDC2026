from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from submission_spec import (
    DEFAULT_MAX_TOTAL_BYTES,
    FORBIDDEN_REVIEW_PATTERNS,
    MANIFEST_EXTENSIONS,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    MAX_RESULT_ROWS,
    MAX_RESULT_WEIGHT_SUM,
    README_SECTIONS,
    REVIEW_ENTRYPOINTS,
    REQUIRED_DIRECTORIES,
    REQUIRED_FILES,
    RESULT_PATH,
    RESULT_STOCK_ID_PATTERN,
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _missing_paths(app_dir: Path, paths: Iterable[Path], want_dir: bool) -> list[str]:
    missing = []
    for relative in paths:
        path = app_dir / relative
        if want_dir and not path.is_dir():
            missing.append(relative.as_posix())
        if not want_dir and not path.is_file():
            missing.append(relative.as_posix())
    return missing


def _missing_readme_sections(app_dir: Path) -> list[str]:
    text = _read_text(app_dir / "readme.md").lower()
    missing = []
    for section, aliases in README_SECTIONS.items():
        if not any(alias.lower() in text for alias in aliases):
            missing.append(section)
    return missing


def _forbidden_entrypoint_issues(app_dir: Path) -> list[str]:
    issues = []
    for relative in REVIEW_ENTRYPOINTS:
        text = _read_text(app_dir / relative)
        for name, pattern in FORBIDDEN_REVIEW_PATTERNS.items():
            if pattern.search(text):
                issues.append(f"{relative.as_posix()} contains forbidden review action: {name}")
    return issues


def _has_official_output_path(app_dir: Path) -> bool:
    script_text = _read_text(app_dir / "test.sh").replace("\\", "/")
    test_py_text = _read_text(app_dir / "code/src/test.py").replace("\\", "/")
    return "output/result.csv" in script_text and "output/result.csv" in test_py_text


def _validate_result_csv(app_dir: Path) -> tuple[bool, list[str]]:
    path = app_dir / RESULT_PATH
    if not path.exists():
        return True, []
    issues: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["stock_id", "weight"]:
                return False, ["output/result.csv columns must be exactly stock_id,weight"]
            rows = list(reader)
    except UnicodeDecodeError as exc:
        return False, [f"output/result.csv must be UTF-8 encoded: {exc}"]
    except csv.Error as exc:
        return False, [f"output/result.csv is invalid CSV: {exc}"]

    if not rows:
        issues.append("output/result.csv must contain at least one stock")
    if len(rows) > MAX_RESULT_ROWS:
        issues.append(f"output/result.csv has more than {MAX_RESULT_ROWS} rows")

    seen: set[str] = set()
    weights: list[float] = []
    for line_number, row in enumerate(rows, start=2):
        stock_id = str(row.get("stock_id", "")).strip()
        weight_text = str(row.get("weight", "")).strip()
        if not RESULT_STOCK_ID_PATTERN.fullmatch(stock_id):
            issues.append(f"line {line_number}: stock_id must be a 6-digit A-share code")
        elif stock_id in seen:
            issues.append(f"line {line_number}: duplicate stock_id {stock_id}")
        else:
            seen.add(stock_id)

        try:
            weight = float(weight_text)
        except ValueError:
            issues.append(f"line {line_number}: weight is not numeric")
            continue
        if not math.isfinite(weight):
            issues.append(f"line {line_number}: weight is not finite")
            continue
        if weight < 0:
            issues.append(f"line {line_number}: weight is negative")
            continue
        weights.append(weight)

    total_weight = sum(weights)
    if total_weight > MAX_RESULT_WEIGHT_SUM + 1e-10:
        issues.append(f"result weight sum exceeds 1.0: {total_weight:.12g}")
    return not issues, issues


def _total_file_bytes(app_dir: Path) -> int:
    return sum(path.stat().st_size for path in app_dir.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_source_files(app_dir: Path) -> list[Path]:
    roots = [
        app_dir / "code",
        app_dir / "init.sh",
        app_dir / "train.sh",
        app_dir / "test.sh",
        app_dir / "readme.md",
        app_dir / "README.md",
        app_dir / "Dockerfile",
        app_dir / ".dockerignore",
        app_dir / "pyproject.toml",
        app_dir / "uv.lock",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for path in root.rglob("*") if path.is_file()]
        for path in candidates:
            relative = path.relative_to(app_dir)
            if relative == MANIFEST_PATH:
                continue
            if path.suffix in MANIFEST_EXTENSIONS:
                files.append(path)
    return sorted(set(files), key=lambda item: item.relative_to(app_dir).as_posix())


def build_code_manifest(app_dir: str | Path) -> dict[str, object]:
    app_path = Path(app_dir)
    entries = []
    total_bytes = 0
    for path in _manifest_source_files(app_path):
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(app_path).as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "description": "BDC2026 clean in-place submission source manifest",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }


def write_code_manifest(app_dir: str | Path) -> dict[str, object]:
    app_path = Path(app_dir)
    manifest = build_code_manifest(app_path)
    path = app_path / MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _validate_manifest(app_dir: Path) -> tuple[bool, int, list[str]]:
    path = app_dir / MANIFEST_PATH
    if not path.is_file():
        return False, 0, [f"missing source manifest: {MANIFEST_PATH.as_posix()}"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, 0, [f"invalid source manifest JSON: {exc}"]

    issues = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        issues.append("source manifest schema_version is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        issues.append("source manifest files must be a non-empty list")
        return False, 0, issues

    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            issues.append("source manifest entry is not an object")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            issues.append("source manifest entry missing path")
            continue
        if relative in seen:
            issues.append(f"source manifest duplicate path: {relative}")
            continue
        seen.add(relative)
        source_path = app_dir / relative
        if not source_path.is_file():
            issues.append(f"source manifest file missing: {relative}")
            continue
        if entry.get("bytes") != source_path.stat().st_size:
            issues.append(f"source manifest byte count stale: {relative}")
        if entry.get("sha256") != _sha256(source_path):
            issues.append(f"source manifest checksum stale: {relative}")

    current = {path.relative_to(app_dir).as_posix() for path in _manifest_source_files(app_dir)}
    for relative in sorted(current - seen):
        issues.append(f"source manifest missing current file: {relative}")
    return not issues, len(files), issues


def validate_submission_layout(
    app_dir: str | Path,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, object]:
    app_path = Path(app_dir)
    issues: list[str] = []

    missing_dirs = _missing_paths(app_path, REQUIRED_DIRECTORIES, want_dir=True)
    missing_files = _missing_paths(app_path, REQUIRED_FILES, want_dir=False)
    missing_sections = _missing_readme_sections(app_path)
    forbidden_issues = _forbidden_entrypoint_issues(app_path)
    result_path_ok = _has_official_output_path(app_path)
    result_ok, result_issues = _validate_result_csv(app_path)
    total_bytes = _total_file_bytes(app_path)
    size_ok = total_bytes <= max_total_bytes
    manifest_ok, manifest_file_count, manifest_issues = _validate_manifest(app_path)

    issues.extend(f"missing required directory: {item}" for item in missing_dirs)
    issues.extend(f"missing required file: {item}" for item in missing_files)
    issues.extend(f"readme missing section: {item}" for item in missing_sections)
    issues.extend(forbidden_issues)
    if not result_path_ok:
        issues.append("test entrypoint must write output/result.csv by default")
    issues.extend(result_issues)
    if not size_ok:
        issues.append(f"submission size exceeds limit: {total_bytes} > {max_total_bytes}")
    issues.extend(manifest_issues)

    checks = {
        "required_paths": not missing_dirs and not missing_files,
        "readme_sections": not missing_sections,
        "offline_review_scripts": not forbidden_issues,
        "result_output_path": result_path_ok,
        "result_csv_format": result_ok,
        "submission_size_budget": size_ok,
        "source_manifest": manifest_ok,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "total_bytes": total_bytes,
        "max_total_bytes": max_total_bytes,
        "manifest_file_count": manifest_file_count,
        "issues": issues,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the BDC2026 submission app layout.")
    parser.add_argument("--app-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument("--write-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_manifest:
        write_code_manifest(args.app_dir)
    report = validate_submission_layout(args.app_dir, args.max_total_bytes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
