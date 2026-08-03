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
from gaswipelab.hmi import diagnostics as hmi_diagnostics
from gaswipelab.hmi import screen as hmi_screen_module
from gaswipelab.ml.ood import DEFAULT_PRODUCT_SIZE, TARGET_RANGE_GM2, RangeChecker
from gaswipelab.ml.predictor import (
    CODE_CHANGE_ERROR_FACTOR,
    ERROR_REFERENCE,
    MODEL_SKILL,
    GasWipingPredictor,
    OutOfScopeError,
)
from gaswipelab.services.machine_design_service import (
    CODE_CHANGED_KEY,
    FIXED_NOTE,
    INTERPOLATION_NOTE,
    LEVERS,
    MachineDesignService,
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

#: 画面の選択肢から外すめっき付着量記号。
#: 305 は予測自体を拒否する（ウォーマーコイル）。122 は運用対象外のため一覧から隠す。
HIDDEN_COATING_CODES = frozenset({"305", "122"})

# --- 実績モデル（v4.0）。モデルJSONが重いので初回アクセス時に読み込む ---
_MODEL_DIRS = ("/models", "models", "deliverables/GasWipeLab_WebApp/models")
_machine_state: dict[str, Any] = {"predictor": None, "checker": None, "service": None, "reference": None}


def _model_dir() -> str:
    from pathlib import Path

    for candidate in _MODEL_DIRS:
        if (Path(candidate) / "manifest.json").exists():
            return candidate
    raise FileNotFoundError("実績モデル（models/manifest.json）が見つかりません。")


def _machine() -> MachineDesignService:
    if _machine_state["service"] is None:
        directory = _model_dir()
        reference = json.loads((__import__("pathlib").Path(directory) / "reference.json").read_text(encoding="utf-8"))
        predictor = GasWipingPredictor(directory)
        checker = RangeChecker(reference)
        _machine_state.update({
            "predictor": predictor,
            "checker": checker,
            "reference": reference,
            "service": MachineDesignService(predictor, checker),
        })
    return _machine_state["service"]

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
# 実績モデル（実機の操業実績データ由来）
# ==================================================================
def ml_bootstrap() -> str:
    """ライン・付着量記号・カテゴリ選択肢・レンジ・誤差参考値をまとめて返す。"""
    try:
        service = _machine()
        reference = _machine_state["reference"]
        checker = _machine_state["checker"]
        lines = {}
        for line, info in reference["lines"].items():
            lines[line] = {
                # 公開用に件数を落とした reference.json（build_reference_data.py --redact）でも動くようにする
                "n": info.get("n"),
                "features": info["features"],
                "categories": info["categories"],
                "flow": info["flow"],
                "ui_ranges": checker.ui_ranges(line),
                # 実データから求めた偏効果。画面で「何がどれだけ効くか」を示すのに使う。
                "trends": checker.trends(line),
                "codes": {
                    code: {"stats": entry["stats"]}
                    for code, entry in info["codes"].items()
                    if code not in HIDDEN_COATING_CODES
                },
            }
        return _ok({
            "model_version": service.predictor.model_version,
            "target_range_gm2": list(TARGET_RANGE_GM2),
            "lines": lines,
            "levers": [{"key": x.key, "label": x.label, "unit": x.unit, "driver": x.driver} for x in LEVERS],
            "error_reference": ERROR_REFERENCE,
            # 「記号別中央値を出すだけ」に対する上乗せ分。GLではほぼ無いことを画面に出す。
            "model_skill": MODEL_SKILL,
            # 直前コイルから記号が変わったときの誤差倍率（実測）
            "code_change_error_factor": CODE_CHANGE_ERROR_FACTOR,
            "code_changed_key": CODE_CHANGED_KEY,
            "fixed_note": FIXED_NOTE,
            "note": INTERPOLATION_NOTE,
        })
    except Exception as exc:
        return _error(exc)


def ml_defaults(payload_json: str) -> str:
    """付着量記号を選んだときの初期条件（その記号の実績中央値）。"""
    try:
        payload = json.loads(payload_json)
        line = str(payload.get("line", "GI")).strip()
        code = str(payload.get("coating_code", "")).strip()
        service = _machine()
        checker = _machine_state["checker"]
        reference = _machine_state["reference"]
        entry = reference["lines"].get(line, {}).get("codes", {}).get(code)
        categories = {}
        if entry:
            for name, options in entry["categories"].items():
                categories[name] = options[0][0] if options else "__MISSING__"
        condition = dict(checker.defaults(line, code))
        condition.update(categories)
        condition["line"] = line
        condition["coating_code"] = code
        stats = checker.code_stats(line, code) or {}
        speed_hint = checker.speed_for_size(
            line, DEFAULT_PRODUCT_SIZE["製品板厚_mm"], DEFAULT_PRODUCT_SIZE["製品板幅_mm"])
        return _ok({
            "condition": condition,
            "stats": stats,
            "target_ch_gm2": stats.get("CH_median"),
            "features": (entry or {}).get("features", {}),
            "speed_hint": speed_hint,
        })
    except Exception as exc:
        return _error(exc)


def ml_speed_hint(payload_json: str) -> str:
    """製品板厚・製品板幅から、実績に基づく通板速度の初期値を返す。"""
    try:
        payload = json.loads(payload_json)
        line = str(payload.get("line", "GI")).strip()
        thickness_mm = float(payload["製品板厚_mm"])
        width_mm = float(payload["製品板幅_mm"])
        _machine()
        checker = _machine_state["checker"]
        return _ok({"speed_hint": checker.speed_for_size(line, thickness_mm, width_mm)})
    except Exception as exc:
        return _error(exc)


def ml_predict(payload_json: str) -> str:
    try:
        condition = json.loads(payload_json)
        return _ok({"prediction": _machine().predict(condition)})
    except OutOfScopeError as exc:
        return _dumps({"ok": False, "out_of_scope": True, "error": str(exc)})
    except Exception as exc:
        return _error(exc)


def ml_design(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
        target = float(payload.pop("target_ch_gm2", 0.0))
        return _ok({"design": _machine().design(payload, target)})
    except OutOfScopeError as exc:
        return _dumps({"ok": False, "out_of_scope": True, "error": str(exc)})
    except Exception as exc:
        return _error(exc)


def ml_compare(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
        return _ok({"comparison": _machine().compare(payload["before"], payload["after"])})
    except OutOfScopeError as exc:
        return _dumps({"ok": False, "out_of_scope": True, "error": str(exc)})
    except Exception as exc:
        return _error(exc)


def hmi_screen(payload_json: str) -> str:
    """実機YG装置監視画面の再現データ。

    予測は既存の実績モデルをそのまま呼ぶだけで、モデルや前処理には手を入れていない。
    記録が無い項目は推測で埋めず、未取得として返す。
    """
    try:
        payload = json.loads(payload_json)
        condition = payload.get("condition", payload)
        modes = payload.get("control_modes")
        service = _machine()
        prediction = service.predict(condition)
        screen = hmi_screen_module.build_screen(
            condition, prediction, modes, model_version=service.predictor.model_version)
        return _ok({
            "screen": screen,
            "diagnostics": hmi_diagnostics.compute(screen),
            "coating_consistency": hmi_diagnostics.coating_consistency(
                screen["coating"]["device_total"]["value"],
                screen["coating"]["front"]["value"],
                screen["coating"]["back"]["value"],
            ),
            "prediction": prediction,
        })
    except OutOfScopeError as exc:
        return _dumps({"ok": False, "out_of_scope": True, "error": str(exc)})
    except Exception as exc:
        return _error(exc)


def ml_curves(payload_json: str) -> str:
    """3本のレバーそれぞれについてCH応答カーブを返す。"""
    try:
        condition = json.loads(payload_json)
        service = _machine()
        return _ok({"curves": {x.key: service.response_curve(condition, x.key) for x in LEVERS}})
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
