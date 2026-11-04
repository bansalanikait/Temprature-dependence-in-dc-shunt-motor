"""CSV and JSON export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def export_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    frame.to_csv(path, index=False)


def export_json(data: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
