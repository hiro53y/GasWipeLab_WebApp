from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gaswipelab.models.calibration_model import (
    CALIBRATION_KEYS,
    CalibrationMetrics,
    CalibrationResult,
    calculate_metrics,
    fit_multi_factor,
)
from gaswipelab.services.analysis_service import AnalysisService
from gaswipelab.services.settings_service import (
    load_calibration_coefficients,
    load_model_coefficients,
    save_calibration_coefficients,
)


class CalibrationService:
    def __init__(self, analysis_service: AnalysisService | None = None) -> None:
        self.analysis_service = analysis_service or AnalysisService()

    def predict_dataframe(self, df: pd.DataFrame, calibration: dict[str, float] | None = None) -> np.ndarray:
        predictions: list[float] = []
        for _, row in df.iterrows():
            result = self.analysis_service.analyze_for_csv_row(row.to_dict(), calibration=calibration)
            predictions.append(float(result["cw_one_side_gm2"]))
        return np.asarray(predictions, dtype=float)

    def calibrate(self, df: pd.DataFrame) -> CalibrationResult:
        """6係数同時推定で校正を実施。

        Phase 2 (A-3): 起点は「保存済みの校正値」とし、追加調整型のイテレーションが可能。
        保存値が無ければ YAML のベース値（全1.0, offset=0）。
        """
        self.validate_calibration_dataset(df)
        initial_coefficients = load_calibration_coefficients()
        actual = df["measured_cw_one_side_gm2"].to_numpy(dtype=float)

        # least_squares から渡される候補係数で予測する callable。
        def predict_func(cal: dict[str, float]) -> np.ndarray:
            return self.predict_dataframe(df, calibration=cal)

        result = fit_multi_factor(actual, predict_func, initial_coefficients)
        save_calibration_coefficients(result.coefficients)
        return result

    @staticmethod
    def validate_calibration_dataset(df: pd.DataFrame) -> None:
        """校正CSVの最低条件を検査する。

        6係数同時推定は自由度が高いため、少数・変動不足データで保存すると過学習になる。
        """
        min_rows = 8
        if len(df) < min_rows:
            raise ValueError(f"校正には最低{min_rows}件の有効データが必要です（現在 {len(df)}件）。")
        varying_columns = [
            "line_speed_mpm",
            "plenum_pressure_kpa",
            "nozzle_gap_mm",
            "nozzle_strip_distance_mm",
            "measured_cw_one_side_gm2",
        ]
        low_variance = [col for col in varying_columns if col in df.columns and df[col].nunique(dropna=True) < 3]
        if low_variance:
            raise ValueError("校正データの条件分散が不足しています: " + ", ".join(low_variance))

    def cross_validate_leave_one_out(self, df: pd.DataFrame) -> CalibrationMetrics:
        """LOOCVのRMSE等を返す。UI警告と将来の根拠管理向け。"""
        actual = df["measured_cw_one_side_gm2"].to_numpy(dtype=float)
        if len(df) < 3:
            predicted = self.predict_dataframe(df)
            return calculate_metrics(actual, predicted)
        preds: list[float] = []
        for i in range(len(df)):
            train = df.drop(df.index[i]).reset_index(drop=True)
            test = df.iloc[[i]].reset_index(drop=True)
            initial = load_calibration_coefficients()

            def predict_func(cal: dict[str, float]) -> np.ndarray:
                return self.predict_dataframe(train, calibration=cal)

            fitted = fit_multi_factor(train["measured_cw_one_side_gm2"].to_numpy(dtype=float), predict_func, initial)
            preds.append(float(self.predict_dataframe(test, calibration=fitted.coefficients)[0]))
        return calculate_metrics(actual, np.asarray(preds, dtype=float))

    @staticmethod
    def coefficient_rows(
        coefficients: dict[str, float],
        before: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """校正係数の比較行を返す。

        before が None の場合は「保存済み校正値（=今回の起点）」を使う。
        Phase 2 (A-3) でデフォルトの参照先を YAML 基準値から保存値へ変更。
        """
        if before is None:
            before = load_calibration_coefficients()
        labels = {
            "scale_factor": "全体スケール係数",
            "pressure_factor": "圧力係数",
            "distance_factor": "距離係数",
            "speed_factor": "速度係数",
            "gap_factor": "ギャップ係数",
            "offset_gm2": "オフセット",
        }
        rows: list[dict[str, Any]] = []
        for key in CALIBRATION_KEYS:
            start = float(before.get(key, 1.0 if key != "offset_gm2" else 0.0))
            after = float(coefficients.get(key, start))
            delta = after - start
            rows.append(
                {
                    "項目": labels[key],
                    "記号": key,
                    "校正前": start,
                    "校正後": after,
                    "変化": delta,
                }
            )
        return rows

    @staticmethod
    def reset_to_base() -> dict[str, float]:
        """校正値を YAML のベース値（全1.0, offset=0）にリセットして保存する。"""
        base = load_model_coefficients().get("calibration", {})
        save_calibration_coefficients(base)
        return base
