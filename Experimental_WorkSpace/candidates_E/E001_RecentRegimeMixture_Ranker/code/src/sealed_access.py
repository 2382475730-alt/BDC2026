from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from featurework import DATE_COL


HOLDOUT_NUMERIC_START = pd.Timestamp("2026-04-03")
REQUIRED_LOCK_KEYS = {"config_id", "config_sha256", "code_sha256", "seed", "policy", "locked_at"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_finalist_lock(path: str | Path) -> dict[str, Any]:
    lock_path = Path(path)
    if not lock_path.exists():
        raise PermissionError("Recent Sealed Holdout remains sealed: finalist lock missing")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_LOCK_KEYS - set(payload))
    if missing or payload.get("status") != "FINALIST_LOCKED_BEFORE_HOLDOUT_ACCESS":
        raise PermissionError(f"invalid finalist lock; missing={missing}")
    return payload


def read_recent_holdout(
    source_csv: str | Path,
    finalist_lock: str | Path,
    access_receipt: str | Path,
) -> pd.DataFrame:
    """One-shot numeric access, irreversibly receipted before returning rows."""
    validate_finalist_lock(finalist_lock)
    receipt_path = Path(access_receipt)
    if receipt_path.exists():
        raise PermissionError("Recent Sealed Holdout unseal limit already consumed")
    receipt = {
        "status": "UNSEAL_CONSUMED_NO_FURTHER_MODEL_CHANGES_ALLOWED",
        "source_sha256": sha256_file(source_csv),
        "finalist_lock_sha256": sha256_file(finalist_lock),
        "unseal_count": 1,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    frame = pd.read_csv(source_csv)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL]).dt.normalize()
    return frame[frame[DATE_COL] >= HOLDOUT_NUMERIC_START].copy()


def assert_sealed(source_csv: str | Path, finalist_lock: str | Path, access_receipt: str | Path) -> bool:
    del source_csv
    if Path(access_receipt).exists():
        raise AssertionError("holdout already unsealed")
    try:
        validate_finalist_lock(finalist_lock)
    except PermissionError:
        return True
    raise AssertionError("a valid finalist lock exists; sealed state is no longer expected")
