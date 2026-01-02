from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import CACHE_DIR


def submissions_dir() -> Path:
    out_dir = CACHE_DIR / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def method_submissions_dir(method: str) -> Path:
    out_dir = submissions_dir() / str(method)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def append_metrics_csv(
    *,
    csv_path: Path,
    row: dict[str, Any],
    base_columns: list[str] | None = None,
) -> Path:
    """Append a single run row to a metrics CSV.

    - Writes header only if file does not exist.
    - Keeps a stable prefix of columns if `base_columns` is provided.
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    base_columns = list(base_columns or [])
    row = dict(row)

    # Ensure base columns exist in the row (stable schema for aggregation).
    for col in base_columns:
        row.setdefault(col, None)

    extra_cols = sorted([c for c in row.keys() if c not in base_columns])
    columns = base_columns + extra_cols

    df = pd.DataFrame([{c: row.get(c, None) for c in columns}])

    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    return csv_path
