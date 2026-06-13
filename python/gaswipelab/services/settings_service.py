"""settings_service.py — Webアプリ用（ファイルI/OなしでConfigをインメモリ提供）

YAMLファイルの代わりにPython dictで設定値を直接保持。
校正係数はlocalStorage経由でブラウザに永続化する。
"""
from __future__ import annotations

import json
from typing import Any

# ============================================================
# インラインConfig（configs/*.yaml の内容を Python dict で保持）
# ============================================================

_MODEL_COEFFICIENTS: dict[str, Any] = {
    "nozzle": {"discharge_coefficient": 0.95},
    "jet": {
        "near_pm_quad": -0.0056, "near_pm_lin": 0.0268, "near_pm_const": 1.0108,
        "pm_ratio_cap": 1.10,
        "near_bp_lin": 0.0453, "near_bp_const": 0.7921,
        "near_tau_lin": -0.0001, "near_tau_const": 0.0035,
        "near_bs_lin": 0.0443, "near_bs_const": 1.1687,
        "far_bp_quad": 0.0019, "far_bp_lin": 0.0551, "far_bp_const": 0.4035,
    },
    "film": {
        "reference_strip_width_mm": 1200.0,
        "edge_effect_strength": 0.05,
        "unwipeable_alloy_layer_gm2": 9.0,
    },
    "splash": {
        "limit_coef_a": 0.042, "limit_coef_b": 6.0, "limit_exp_n": 1.6,
        "runback_film_thickness_mm": 0.4, "nozzle_tilt_deg": 0.0,
        "low_threshold": 0.6, "high_threshold": 1.0,
    },
    "calibration": {
        "scale_factor": 1.00, "pressure_factor": 1.00, "distance_factor": 1.00,
        "speed_factor": 1.00, "gap_factor": 1.00, "offset_gm2": 0.00,
    },
    "uncertainty": {
        "base_relative_percent": 12.0, "far_field_add_percent": 8.0,
        "very_high_standoff_add_percent": 8.0, "caution_h_over_b_max": 15.0,
        "validated_cw_max_gm2": 75.0, "high_cw_add_percent": 8.0,
        "choked_add_percent": 8.0, "high_mach_add_percent": 5.0,
        "splash_high_add_percent": 8.0, "low_distance_add_percent": 10.0,
        "low_bath_temp_c": 450.0, "bath_temperature_add_percent": 5.0,
        "confidence_high_max_percent": 18.0, "confidence_medium_max_percent": 30.0,
    },
    "metadata": {"model_version": "2.0"},
}

_MATERIAL_PROPERTIES: dict[str, Any] = {
    "zinc": {
        "density_liquid_kg_m3": 6623,
        "density_solid_kg_m3": 7140,
        "coating_density_kg_m3": 7140,
        "viscosity_pa_s": 0.00294,
        "viscosity_ref_pa_s": 0.00294,
        "viscosity_ref_temp_c": 460.0,
        "viscosity_e_over_r_k": 1312.0,
        "surface_tension_n_m": 0.78,
        "melting_point_c": 419.5,
    },
    "gas": {
        "air": {
            "display_name_jp": "空気",
            "r_specific_j_kgk": 287.05,
            "gamma": 1.4,
            "viscosity_pa_s_25c": 1.85e-5,
        },
        "nitrogen": {
            "display_name_jp": "窒素",
            "r_specific_j_kgk": 296.8,
            "gamma": 1.4,
            "viscosity_pa_s_25c": 1.76e-5,
        },
    },
    "constants": {
        "gravity_m_s2": 9.80665,
        "atmospheric_pressure_pa": 101325,
        "default_gas_temperature_c": 25,
    },
}

_DEFAULT_CONDITIONS: dict[str, Any] = {
    "default_condition": {
        "project_name": "Sample_Project_001",
        "plenum_pressure_kpa": 30.0,
        "nozzle_gap_mm": 1.0,
        "nozzle_strip_distance_mm": 10.0,
        "gas_type": "air",
        "line_speed_mpm": 120.0,
        "strip_width_mm": 1200,
        "target_cw_one_side_gm2": 60.0,
        "bath_temp_c": 460.0,
    },
    "validation_ranges": {
        "plenum_pressure_kpa": {"min": 1.0, "max": 100.0},
        "nozzle_gap_mm": {"min": 0.2, "max": 3.0},
        "nozzle_strip_distance_mm": {"min": 1.0, "max": 50.0},
        "line_speed_mpm": {"min": 10.0, "max": 300.0},
        "strip_width_mm": {"min": 300, "max": 2000},
        "target_cw_one_side_gm2": {"min": 10.0, "max": 300.0},
        "bath_temp_c": {"min": 420.0, "max": 500.0},
    },
}

# ランタイムオーバーライド（設定ダイアログ・校正で上書きされる値を保持）
_runtime_calibration: dict[str, Any] = {}
_runtime_model_override: dict[str, Any] = {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ============================================================
# Public API（analysis_service.py が呼ぶ関数群）
# ============================================================

def load_material_properties() -> dict[str, Any]:
    return _MATERIAL_PROPERTIES


def load_model_coefficients() -> dict[str, Any]:
    base = load_base_model_coefficients()
    if _runtime_model_override:
        return _deep_merge(base, _runtime_model_override)
    return base


def load_base_model_coefficients() -> dict[str, Any]:
    return _MODEL_COEFFICIENTS


def load_default_conditions() -> dict[str, Any]:
    return _DEFAULT_CONDITIONS


def load_calibration_coefficients() -> dict[str, Any]:
    """校正係数を返す。ランタイム値 > localStorage > モデルデフォルト の優先順。"""
    if _runtime_calibration:
        return _runtime_calibration

    # Pyodide環境ではJSのlocalStorageにアクセスできる
    try:
        from js import localStorage  # type: ignore[import]
        raw = localStorage.getItem("gaswipelab_calibration")
        if raw:
            return json.loads(raw)
    except Exception:
        pass

    return dict(_MODEL_COEFFICIENTS["calibration"])


def save_calibration_coefficients(coefficients: dict[str, Any]) -> None:
    global _runtime_calibration
    _runtime_calibration = dict(coefficients)
    try:
        from js import localStorage  # type: ignore[import]
        localStorage.setItem("gaswipelab_calibration", json.dumps(coefficients))
    except Exception:
        pass


def save_model_coefficients_override(coefficients: dict[str, Any]) -> None:
    global _runtime_model_override
    _runtime_model_override = dict(coefficients)
    try:
        from js import localStorage  # type: ignore[import]
        localStorage.setItem("gaswipelab_model_override", json.dumps(coefficients))
    except Exception:
        pass


def load_model_coefficients_override() -> dict[str, Any]:
    try:
        from js import localStorage  # type: ignore[import]
        raw = localStorage.getItem("gaswipelab_model_override")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {}


# coefficients_override_path はファイルシステムAPIだがWebでは不要。互換のためダミーを返す。
def calibration_path():
    from pathlib import Path
    return Path("/user_data/calibration_coefficients.json")


def coefficients_override_path():
    from pathlib import Path
    return Path("/user_data/model_coefficients_override.yaml")
