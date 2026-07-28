from __future__ import annotations

import re
from pathlib import Path


REQUIRED_DIRECTORIES = [
    Path("code/src"),
    Path("data"),
    Path("model"),
    Path("output"),
    Path("temp"),
]

REQUIRED_FILES = [
    Path("code/src/featurework.py"),
    Path("code/src/score_self.py"),
    Path("code/src/train.py"),
    Path("code/src/test.py"),
    Path("code/src/validate_submission.py"),
    Path("code/src/submission_spec.py"),
    Path("data/train.csv"),
    Path("data/test.csv"),
    Path("init.sh"),
    Path("train.sh"),
    Path("test.sh"),
    Path("readme.md"),
]

README_SECTIONS = {
    "environment": ["## environment", "## \u73af\u5883\u914d\u7f6e"],
    "data": ["## data", "## \u6570\u636e"],
    "pretrained models": ["## pretrained models", "## \u9884\u8bad\u7ec3\u6a21\u578b"],
    "algorithm": ["## algorithm", "## \u7b97\u6cd5"],
    "training flow": ["## training flow", "## \u8bad\u7ec3\u6d41\u7a0b"],
    "inference flow": ["## inference flow", "## \u63a8\u7406\u6d41\u7a0b"],
    "other notes": ["## other notes", "## \u5176\u4ed6\u6ce8\u610f\u4e8b\u9879"],
}

REVIEW_ENTRYPOINTS = [
    Path("train.sh"),
    Path("test.sh"),
    Path("code/src/train.py"),
    Path("code/src/test.py"),
]

FORBIDDEN_REVIEW_PATTERNS = {
    "network download": re.compile(r"\b(wget|curl|baostock|akshare|get_stock_data)\b", re.IGNORECASE),
    "docker compose": re.compile(r"\bdocker\s+compose\b", re.IGNORECASE),
    "data run helper": re.compile(r"\bdata/run\.sh\b", re.IGNORECASE),
}

MANIFEST_PATH = Path("code/submission_manifest.json")
MANIFEST_SCHEMA_VERSION = 1
RESULT_PATH = Path("output/result.csv")
RESULT_STOCK_ID_PATTERN = re.compile(r"^\d{6}$")
MAX_RESULT_ROWS = 5
MAX_RESULT_WEIGHT_SUM = 1.0
DEFAULT_MAX_TOTAL_BYTES = 10_000_000_000

MANIFEST_EXTENSIONS = {
    "",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
