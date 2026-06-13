from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from scipy.optimize import least_squares
except ModuleNotFoundError:  # pragma: no cover - フォールバック（scipy 未導入環境）
    least_squares = None


# 校正対象の6係数。順序を固定して least_squares のベクトルと対応付ける。
CALIBRATION_KEYS: tuple[str, ...] = (
    "scale_factor",
    "pressure_factor",
    "distance_factor",
    "speed_factor",
    "gap_factor",
    "offset_gm2",
)

# 各係数の探索範囲。物理的に妥当な区間を設定し、最小二乗解の暴走を防ぐ。
_BOUNDS_LOWER: dict[str, float] = {
    "scale_factor": 0.10,
    "pressure_factor": 0.10,
    "distance_factor": 0.10,
    "speed_factor": 0.10,
    "gap_factor": 0.10,
    "offset_gm2": -200.0,
}
_BOUNDS_UPPER: dict[str, float] = {
    "scale_factor": 10.0,
    "pressure_factor": 10.0,
    "distance_factor": 10.0,
    "speed_factor": 10.0,
    "gap_factor": 10.0,
    "offset_gm2": 200.0,
}


@dataclass(frozen=True)
class CalibrationMetrics:
    mae: float
    mape: float
    rmse: float
    r2: float
    bias: float


@dataclass(frozen=True)
class CalibrationResult:
    coefficients: dict[str, float]
    metrics_before: CalibrationMetrics
    metrics_after: CalibrationMetrics
    predicted_before: np.ndarray
    predicted_after: np.ndarray
    residual_after: np.ndarray
    # Phase 2: 起点（before）として実際に使った係数。
    coefficients_before: dict[str, float]
    # v2.0: 係数が探索境界に張り付いた場合の警告（非同定性・外挿の検出）。
    boundary_warnings: tuple[str, ...] = ()


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> CalibrationMetrics:
    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    mape = float(np.mean(np.abs(residual) / np.maximum(np.abs(actual), 1.0e-9)) * 100.0)
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    bias = float(np.mean(residual))
    return CalibrationMetrics(mae=mae, mape=mape, rmse=rmse, r2=float(r2), bias=bias)


def _ensure_full_coefficients(coefficients: dict[str, float]) -> dict[str, float]:
    """6係数すべてを含む dict にデフォルト値で補完する。"""
    defaults = {
        "scale_factor": 1.0,
        "pressure_factor": 1.0,
        "distance_factor": 1.0,
        "speed_factor": 1.0,
        "gap_factor": 1.0,
        "offset_gm2": 0.0,
    }
    result = dict(defaults)
    for key, value in coefficients.items():
        result[key] = float(value)
    return result


def _vector_to_coefficients(vector: np.ndarray) -> dict[str, float]:
    return {key: float(vector[i]) for i, key in enumerate(CALIBRATION_KEYS)}


def _coefficients_to_vector(coefficients: dict[str, float]) -> np.ndarray:
    return np.array([coefficients[key] for key in CALIBRATION_KEYS], dtype=float)


def fit_global_scale(actual: np.ndarray, predicted: np.ndarray, base_coefficients: dict) -> CalibrationResult:
    """[後方互換] scale_factor と offset_gm2 のみを線形回帰で推定する。

    Phase 2 以降は fit_multi_factor の使用を推奨する。テスト・既存呼出し向けに残置。
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    before_full = _ensure_full_coefficients(base_coefficients)
    before = calculate_metrics(actual, predicted)

    denominator = float(np.dot(predicted, predicted))
    scale = float(np.dot(predicted, actual) / denominator) if denominator > 0 else 1.0
    scaled = predicted * scale
    offset = float(np.mean(actual - scaled))
    adjusted = scaled + offset
    after = calculate_metrics(actual, adjusted)

    coefficients = dict(before_full)
    coefficients["scale_factor"] = scale
    coefficients["offset_gm2"] = offset
    return CalibrationResult(
        coefficients=coefficients,
        metrics_before=before,
        metrics_after=after,
        predicted_before=predicted,
        predicted_after=adjusted,
        residual_after=adjusted - actual,
        coefficients_before=before_full,
    )


def _detect_boundary_hits(coefficients: dict[str, float]) -> tuple[str, ...]:
    """係数が探索境界の近傍（2%以内）に張り付いていないか検査する。

    境界到達は、データの情報量不足・モデルとの系統的不一致・外挿のいずれかを
    示す。物理的に無意味な係数で運用しないよう、UI で警告表示する。
    """
    warnings: list[str] = []
    for key in CALIBRATION_KEYS:
        value = float(coefficients[key])
        lower, upper = _BOUNDS_LOWER[key], _BOUNDS_UPPER[key]
        span = upper - lower
        if value <= lower + 0.02 * span or value >= upper - 0.02 * span:
            warnings.append(
                f"係数 {key} が探索範囲の境界（{value:.3g}）に到達しました。"
                "校正データの条件範囲・件数を確認してください。"
            )
    return tuple(warnings)


def fit_multi_factor(
    actual: np.ndarray,
    predict_func: Callable[[dict[str, float]], np.ndarray],
    initial_coefficients: dict[str, float],
) -> CalibrationResult:
    """6係数を最小二乗で同時推定する（Phase 2 で新設）。

    Parameters
    ----------
    actual : ndarray
        実測値（片面目付 g/m²）。
    predict_func : callable(calibration_dict) -> ndarray
        係数 dict を受け取り予測値配列を返す関数。
        N サンプルそれぞれを analyze して得られる cw_one_side_gm2 を返すラッパーを想定。
    initial_coefficients : dict
        最適化の起点。通常は現在保存されている校正値か、未保存ならベース値（全1, offset=0）。

    Notes
    -----
    - 非線形性（pressure_factor は wiping_strength の分母にあるなど）に対応するため
      scipy.optimize.least_squares（Trust Region Reflective 法）を用いる。
    - scipy が利用できない環境では fit_global_scale にフォールバック。
    """
    actual = np.asarray(actual, dtype=float)
    initial_full = _ensure_full_coefficients(initial_coefficients)
    predicted_before = predict_func(initial_full)
    metrics_before = calculate_metrics(actual, predicted_before)

    if least_squares is None:
        # scipy が無い環境では旧来の線形フィットにフォールバック。
        legacy = fit_global_scale(actual, predicted_before, initial_full)
        return CalibrationResult(
            coefficients=legacy.coefficients,
            metrics_before=metrics_before,
            metrics_after=legacy.metrics_after,
            predicted_before=predicted_before,
            predicted_after=legacy.predicted_after,
            residual_after=legacy.residual_after,
            coefficients_before=initial_full,
        )

    x0 = _coefficients_to_vector(initial_full)
    lower = np.array([_BOUNDS_LOWER[k] for k in CALIBRATION_KEYS], dtype=float)
    upper = np.array([_BOUNDS_UPPER[k] for k in CALIBRATION_KEYS], dtype=float)
    # 初期値が境界内に収まるよう保護。
    x0 = np.clip(x0, lower + 1.0e-6, upper - 1.0e-6)

    def residual_vector(vec: np.ndarray) -> np.ndarray:
        cal = _vector_to_coefficients(vec)
        pred = predict_func(cal)
        return np.asarray(pred, dtype=float) - actual

    try:
        result = least_squares(
            residual_vector,
            x0=x0,
            bounds=(lower, upper),
            method="trf",
            xtol=1.0e-6,
            ftol=1.0e-6,
            max_nfev=200,
        )
        coefficients = _vector_to_coefficients(result.x)
    except Exception:
        # 最適化が失敗した場合は scale/offset のみのフィットに退避。
        return fit_global_scale(actual, predicted_before, initial_full)

    predicted_after = predict_func(coefficients)
    metrics_after = calculate_metrics(actual, predicted_after)

    # 校正後の方が悪化した場合は起点を採用する（過学習・最適化失敗の保険）。
    if metrics_after.rmse > metrics_before.rmse * 1.001:
        coefficients = dict(initial_full)
        predicted_after = predicted_before
        metrics_after = metrics_before

    return CalibrationResult(
        coefficients=coefficients,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        predicted_before=predicted_before,
        predicted_after=predicted_after,
        residual_after=predicted_after - actual,
        coefficients_before=initial_full,
        boundary_warnings=_detect_boundary_hits(coefficients),
    )
