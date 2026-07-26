"""web_api.py — ブラウザ（Pyodide）から呼ぶ JSON API 層。

UI 側は JSON 文字列を渡して JSON 文字列を受け取るだけにし、
Python コードの文字列組み立てを index.html から排除する。
物理モデルの計算はすべて既存の AnalysisService / CalibrationService に委譲する。
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from gaswipelab.services.analysis_service import AnalysisService
from gaswipelab.services.calibration_service import CalibrationService
from gaswipelab.services.csv_service import (
    REQUIRED_ACTUAL_COLUMNS,
    read_actual_results_csv_with_report,
)
from gaswipelab.services.design_service import (
    BATH_MAX_C,
    BATH_MIN_C,
    DISTANCE_MAX_MM,
    DISTANCE_MIN_MM,
    GAP_MAX_MM,
    GAP_MIN_MM,
    PRESSURE_MAX_KPA,
    PRESSURE_MIN_KPA,
    SPEED_MAX_MPM,
    SPEED_MIN_MPM,
    DesignService,
)
from gaswipelab.services.settings_service import (
    load_base_model_coefficients,
    load_calibration_coefficients,
    load_default_conditions,
    save_calibration_coefficients,
)

_analysis = AnalysisService()
_calibration = CalibrationService(_analysis)
_design = DesignService(_analysis)

# 校正タブの状態（CSV読み込み → 校正実行 → 保存）
_state: dict[str, Any] = {"df": None, "result": None, "before": None}

# JSON に載せない（UI が使わない）重いキー
_SKIP_KEYS = {"gas_state", "pressure_gradient_pa_m"}


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.bool_)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return str(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_default, ensure_ascii=False)


def _ok(payload: dict[str, Any]) -> str:
    return _dumps({"ok": True, **payload})


def _error(exc: Exception) -> str:
    return _dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _clean_result(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in result.items():
        if key in _SKIP_KEYS:
            continue
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        elif isinstance(value, (bool, int, float, str, list, dict)) or value is None:
            out[key] = value
        else:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = str(value)
    return out


def _thin(values: list[float], step: int) -> list[float]:
    """表示用にプロファイル配列を間引く（描画負荷とJSON量の削減）。"""
    return [round(float(v), 5) for v in values[::step]]


# ==================================================================
# 起動情報
# ==================================================================
def bootstrap() -> str:
    """既定条件・入力レンジ・校正状態をまとめて返す。"""
    try:
        defaults = load_default_conditions()
        return _ok(
            {
                "default_condition": defaults["default_condition"],
                "validation_ranges": defaults["validation_ranges"],
                "design_ranges": {
                    "plenum_pressure_kpa": [PRESSURE_MIN_KPA, PRESSURE_MAX_KPA],
                    "nozzle_strip_distance_mm": [DISTANCE_MIN_MM, DISTANCE_MAX_MM],
                    "line_speed_mpm": [SPEED_MIN_MPM, SPEED_MAX_MPM],
                    "nozzle_gap_mm": [GAP_MIN_MM, GAP_MAX_MM],
                    "bath_temp_c": [BATH_MIN_C, BATH_MAX_C],
                },
                "csv_columns": list(REQUIRED_ACTUAL_COLUMNS),
                "calibration": calibration_snapshot(),
            }
        )
    except Exception as exc:  # pragma: no cover - UI へエラーを返すため
        return _error(exc)


def calibration_snapshot() -> dict[str, Any]:
    base = load_base_model_coefficients().get("calibration", {})
    current = load_calibration_coefficients()
    applied = any(
        abs(float(current.get(key, value)) - float(value)) > 1.0e-9 for key, value in base.items()
    )
    return {"applied": applied, "coefficients": {k: float(v) for k, v in current.items()}, "base": dict(base)}


def calibration_state() -> str:
    try:
        return _ok({"calibration": calibration_snapshot()})
    except Exception as exc:
        return _error(exc)


# ==================================================================
# 単条件解析
# ==================================================================
def analyze(payload_json: str) -> str:
    """1条件の解析。UI 表示に必要な分布は間引いて返す。"""
    try:
        condition = json.loads(payload_json)
        condition.setdefault("project_name", "WebSession")
        result = _analysis.analyze(condition)
        cleaned = _clean_result(result)
        for key in ("x_mm", "pressure_kpa", "shear_pa", "film_profile_um"):
            cleaned[key] = _thin(cleaned[key], 4)
        return _ok({"result": cleaned})
    except Exception as exc:
        return _error(exc)


# ==================================================================
# 条件設計（逆算）
# ==================================================================
def design(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
        target = float(payload.pop("target_cw_one_side_gm2", 60.0))
        return _ok({"design": _design.design(payload, target)})
    except Exception as exc:
        return _error(exc)


def quick_design(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
        return _ok(
            {
                "quick": _design.quick_design(
                    gas_type=str(payload.get("gas_type", "air")),
                    strip_width_mm=float(payload.get("strip_width_mm", 1200.0)),
                    target_gm2=float(payload.get("target_cw_one_side_gm2", 60.0)),
                    bath_temp_c=float(payload.get("bath_temp_c", 460.0)),
                )
            }
        )
    except Exception as exc:
        return _error(exc)


def response_curves(payload_json: str) -> str:
    try:
        condition = json.loads(payload_json)
        return _ok({"curves": _design.response_curves(condition)})
    except Exception as exc:
        return _error(exc)


def coating_map(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
        points = int(payload.pop("points", 21))
        return _ok({"map": _design.coating_map(payload, points=points)})
    except Exception as exc:
        return _error(exc)


# ==================================================================
# 実績校正
# ==================================================================
def load_csv(text: str) -> str:
    try:
        frame, report = read_actual_results_csv_with_report(io.StringIO(text))
        _state["df"] = frame
        _state["result"] = None
        preview = frame.head(5).to_dict(orient="records")
        summary = {
            column: {
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
                "unique": int(frame[column].nunique()),
            }
            for column in REQUIRED_ACTUAL_COLUMNS
            if column in frame.columns
        }
        return _ok(
            {
                "rows": int(report.rows_loaded),
                "dropped": int(report.rows_dropped),
                "columns": list(frame.columns),
                "preview": preview,
                "summary": summary,
            }
        )
    except Exception as exc:
        return _error(exc)


def calibrate() -> str:
    try:
        frame = _state["df"]
        if frame is None:
            raise ValueError("先にCSVを読み込んでください。")
        before = dict(load_calibration_coefficients())
        predicted_before = _calibration.predict_dataframe(frame)
        result = _calibration.calibrate(frame)
        # CalibrationService.calibrate() は推定値をそのまま保存するため、
        # 利用者が「保存」を押すまで適用しないよう、いったん元の係数へ戻す。
        save_calibration_coefficients(before)
        _state["result"] = result
        _state["before"] = before
        actual = frame["measured_cw_one_side_gm2"].to_numpy(dtype=float)
        return _ok(
            {
                "rows": int(len(frame)),
                "pending": True,
                "metrics_before": {
                    "mae": float(result.metrics_before.mae),
                    "mape": float(result.metrics_before.mape),
                    "rmse": float(result.metrics_before.rmse),
                    "r2": float(result.metrics_before.r2),
                },
                "metrics_after": {
                    "mae": float(result.metrics_after.mae),
                    "mape": float(result.metrics_after.mape),
                    "rmse": float(result.metrics_after.rmse),
                    "r2": float(result.metrics_after.r2),
                },
                "actual": [float(v) for v in actual],
                "predicted_before": [float(v) for v in predicted_before],
                "predicted_after": [float(v) for v in result.predicted_after],
                "residual": [float(v) for v in result.residual_after],
                "coefficient_rows": _calibration.coefficient_rows(result.coefficients, before=before),
                "coefficients": {k: float(v) for k, v in result.coefficients.items()},
                "calibration": calibration_snapshot(),
            }
        )
    except Exception as exc:
        return _error(exc)


def save_calibration() -> str:
    try:
        result = _state["result"]
        if result is None:
            raise ValueError("先に校正を実行してください。")
        save_calibration_coefficients({k: float(v) for k, v in result.coefficients.items()})
        return _ok({"calibration": calibration_snapshot()})
    except Exception as exc:
        return _error(exc)


def reset_calibration() -> str:
    try:
        base = CalibrationService.reset_to_base()
        _state["result"] = None
        return _ok({"calibration": calibration_snapshot(), "coefficients": {k: float(v) for k, v in base.items()}})
    except Exception as exc:
        return _error(exc)
