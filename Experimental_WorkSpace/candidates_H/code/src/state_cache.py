from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def cache_key(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(cache_root: str | Path, key: str) -> tuple[Path, Path]:
    root = Path(cache_root)
    return root / f"{key}.pkl", root / f"{key}.manifest.json"


def save_training_state_cache(
    cache_root: str | Path,
    key: str,
    identity: dict[str, Any],
    state: Any,
) -> dict[str, Any]:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    blob_path, manifest_path = _paths(root, key)
    temporary_blob = blob_path.with_suffix(".pkl.tmp")
    with temporary_blob.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_blob.replace(blob_path)
    manifest = {
        "schema_version": 1,
        "marker": "TEST",
        "cache_key": key,
        "identity": identity,
        "blob_path": blob_path.name,
        "blob_bytes": blob_path.stat().st_size,
        "blob_sha256": file_sha256(blob_path),
        "trusted_local_artifact": True,
        "scaler_fit_scope": identity.get("scaler_fit_scope"),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def load_training_state_cache(
    cache_root: str | Path,
    key: str,
    identity: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    blob_path, manifest_path = _paths(cache_root, key)
    if not blob_path.exists() or not manifest_path.exists():
        return None, {
            "status": "MISS",
            "verified": False,
            "cache_key": key,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_key") != key or manifest.get("identity") != identity:
        return None, {
            "status": "IDENTITY_MISMATCH",
            "verified": False,
            "cache_key": key,
        }
    actual_sha256 = file_sha256(blob_path)
    if actual_sha256 != manifest.get("blob_sha256"):
        return None, {
            "status": "HASH_MISMATCH",
            "verified": False,
            "cache_key": key,
            "blob_sha256": actual_sha256,
        }
    with blob_path.open("rb") as handle:
        state = pickle.load(handle)
    audit = dict(manifest)
    audit.update({"status": "HIT", "verified": True})
    return state, audit
