from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest


H_ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    (H_ROOT / "code" / "src" / "loss.py").resolve(),
    (H_ROOT / "code" / "src" / "state_cache.py").resolve(),
)
TARGET_NAMES = {str(path): path.name for path in TARGETS}
EXECUTED: dict[str, set[int]] = {str(path): set() for path in TARGETS}


def local_tracer(frame, event, arg):
    path = str(Path(frame.f_code.co_filename).resolve())
    if event == "line" and path in EXECUTED:
        EXECUTED[path].add(frame.f_lineno)
    return local_tracer


def global_tracer(frame, event, arg):
    path = str(Path(frame.f_code.co_filename).resolve())
    if path in EXECUTED:
        return local_tracer
    return None


def statement_lines(path: Path) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: set[int] = set()
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        for node in ast.walk(function):
            if isinstance(node, ast.stmt) and not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                lines.add(node.lineno)
    return lines


def main() -> None:
    suite = unittest.defaultTestLoader.discover(str(H_ROOT / "tests"))
    sys.settrace(global_tracer)
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    sys.settrace(None)
    reports: dict[str, object] = {}
    for path in TARGETS:
        statements = statement_lines(path)
        hit = statements & EXECUTED[str(path)]
        reports[path.name] = {
            "statement_lines": len(statements),
            "executed_statement_lines": len(hit),
            "coverage_pct": round(100.0 * len(hit) / len(statements), 2),
            "missing_statement_lines": sorted(statements - hit),
        }
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "method": "stdlib sys.settrace plus AST statement lines",
        "tests_successful": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "minimum_required_pct": 80.0,
        "reports": reports,
        "passed": result.wasSuccessful()
        and all(
            float(report["coverage_pct"]) >= 80.0
            for report in reports.values()
        ),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
