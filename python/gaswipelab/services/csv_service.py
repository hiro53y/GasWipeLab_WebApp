"""csv_service.py — Webアプリ版（StringIO対応）"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from gaswipelab.models.gas_properties import normalize_gas_type

REQUIRED_ACTUAL_COLUMNS = [
    "plenum_pressure_kpa",
    "nozzle_gap_mm",
    "nozzle_strip_distance_mm",
    "line_speed_mpm",
    "strip_width_mm",
    "bath_temp_c",
    "measured_cw_one_side_gm2",
]
NUMERIC_REQUIRED_COLUMNS = REQUIRED_ACTUAL_COLUMNS

# gas_type 列はオプション（ない場合は "air" で補完）
GAS_TYPE_COLUMN = "gas_type"


@dataclass
class CsvLoadReport:
    rows_loaded: int = 0
    rows_dropped: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)


def read_actual_results_csv_with_report(
    path: "str | Path | io.StringIO",
    is_text: bool = False,
) -> "tuple[pd.DataFrame, CsvLoadReport]":
    """CSV またはテキストを読み込む（StringIO対応）。"""
    if isinstance(path, (io.StringIO, io.BytesIO)):
        df = pd.read_csv(path)
    else:
        last_error = None
        df = None
        for encoding in ("utf-8-sig", "utf-8", "cp932"):
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        if df is None:
            raise ValueError(f"CSVの文字コードを判定できませんでした: {last_error}")

    # gas_type がない場合は air で補完
    if GAS_TYPE_COLUMN not in df.columns:
        df[GAS_TYPE_COLUMN] = "air"

    missing = [c for c in REQUIRED_ACTUAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要列がありません: {', '.join(missing)}")

    for column in NUMERIC_REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # gas_type 正規化
    normalized = []
    for value in df[GAS_TYPE_COLUMN]:
        try:
            normalized.append(normalize_gas_type(str(value)))
        except ValueError:
            normalized.append("air")
    df[GAS_TYPE_COLUMN] = normalized

    original_count = len(df)
    dropped_reasons: dict[str, int] = {}
    for column in NUMERIC_REQUIRED_COLUMNS:
        count = int(df[column].isna().sum())
        if count > 0:
            dropped_reasons[column] = count

    cleaned = df.dropna(subset=NUMERIC_REQUIRED_COLUMNS).reset_index(drop=True)
    report = CsvLoadReport(
        rows_loaded=len(cleaned),
        rows_dropped=original_count - len(cleaned),
        dropped_reasons=dropped_reasons,
    )
    return cleaned, report
