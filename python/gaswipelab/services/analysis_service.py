"""解析パイプライン — v2.0

計算の流れ:
    1. ガス状態（理想気体・等エントロピ圧縮性ノズル流れ）
    2. 衝突噴流の壁面圧力・せん断分布（Elsaadawy 2007 / Ellen-Tu 相関）
    3. 薄膜方程式の数値解（最大流束原理）→ 最終液膜厚さ h_f
    4. 目付 = rho_L(液体) × h_f（質量保存。凝固で質量は変わらない）
    5. スプラッシュ限界（Gosset-Buchlin We/We* 判定）
    6. 不確かさ・モデル信頼度評価

校正係数の適用（物理量レベルで適用する）:
    pressure_factor → プレナムゲージ圧, gap_factor → スロットギャップ,
    distance_factor → ノズル-鋼板距離, speed_factor → 通板速度,
    scale_factor → 最終膜厚, offset_gm2 → 目付オフセット
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - 制限環境向けフォールバック
    BaseModel = object

    def Field(*, ge: float | None = None, le: float | None = None):  # type: ignore
        return None

from gaswipelab.models.film_model import edge_effect_factor_from_width, solve_film
from gaswipelab.models.gas_properties import calculate_gas_state, normalize_gas_type
from gaswipelab.models.jet_impingement import calculate_jet_field
from gaswipelab.models.nozzle_model import nozzle_exit_flow
from gaswipelab.models.splash_risk import calculate_splash_risk
from gaswipelab.models.units import kpa_to_pa, m_to_um, pa_to_kpa
from gaswipelab.models.zinc_properties import zinc_viscosity_pa_s
from gaswipelab.services.settings_service import (
    load_calibration_coefficients,
    load_default_conditions,
    load_material_properties,
    load_model_coefficients,
)


if hasattr(BaseModel, "model_validate"):

    class AnalysisCondition(BaseModel):
        project_name: str = "Sample_Project_001"
        plenum_pressure_kpa: float = Field(ge=1.0, le=100.0)
        nozzle_gap_mm: float = Field(ge=0.2, le=3.0)
        nozzle_strip_distance_mm: float = Field(ge=1.0, le=50.0)
        gas_type: str = "air"
        line_speed_mpm: float = Field(ge=10.0, le=300.0)
        strip_width_mm: float = Field(ge=300.0, le=2000.0)
        target_cw_one_side_gm2: float = Field(ge=10.0, le=300.0)
        bath_temp_c: float = Field(ge=420.0, le=500.0)

else:

    class AnalysisCondition:
        _ranges = {
            "plenum_pressure_kpa": (1.0, 100.0),
            "nozzle_gap_mm": (0.2, 3.0),
            "nozzle_strip_distance_mm": (1.0, 50.0),
            "line_speed_mpm": (10.0, 300.0),
            "strip_width_mm": (300.0, 2000.0),
            "target_cw_one_side_gm2": (10.0, 300.0),
            "bath_temp_c": (420.0, 500.0),
        }
        _allowed_gas_types = {"air", "nitrogen"}

        def __init__(
            self,
            project_name: str = "Sample_Project_001",
            plenum_pressure_kpa: float = 30.0,
            nozzle_gap_mm: float = 1.0,
            nozzle_strip_distance_mm: float = 10.0,
            gas_type: str = "air",
            line_speed_mpm: float = 120.0,
            strip_width_mm: float = 1200.0,
            target_cw_one_side_gm2: float = 60.0,
            bath_temp_c: float = 460.0,
        ) -> None:
            normalized_gas = normalize_gas_type(gas_type)
            if normalized_gas not in self._allowed_gas_types:
                raise ValueError(f"gas_type must be one of {sorted(self._allowed_gas_types)}; got '{gas_type}'")
            values = {
                "project_name": project_name,
                "plenum_pressure_kpa": float(plenum_pressure_kpa),
                "nozzle_gap_mm": float(nozzle_gap_mm),
                "nozzle_strip_distance_mm": float(nozzle_strip_distance_mm),
                "gas_type": normalized_gas,
                "line_speed_mpm": float(line_speed_mpm),
                "strip_width_mm": float(strip_width_mm),
                "target_cw_one_side_gm2": float(target_cw_one_side_gm2),
                "bath_temp_c": float(bath_temp_c),
            }
            errors = []
            for key, (minimum, maximum) in self._ranges.items():
                if values[key] < minimum or values[key] > maximum:
                    errors.append(f"{key} must be between {minimum} and {maximum}")
            if errors:
                raise ValueError("; ".join(errors))
            self.__dict__.update(values)

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)


@dataclass
class AnalysisService:
    material_config: dict[str, Any] | None = None
    coefficients: dict[str, Any] | None = None
    default_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.material_config = self.material_config or load_material_properties()
        self.coefficients = self.coefficients or load_model_coefficients()
        self.default_config = self.default_config or load_default_conditions()

    def default_condition(self) -> AnalysisCondition:
        return AnalysisCondition(**self.default_config["default_condition"])

    def analyze(
        self,
        condition: AnalysisCondition | dict,
        calibration: dict | None = None,
        include_profile: bool = True,
    ) -> dict[str, Any]:
        cond = condition if isinstance(condition, AnalysisCondition) else AnalysisCondition(**condition)
        cal = calibration if calibration is not None else load_calibration_coefficients()

        # --- 校正係数を物理入力へ適用 ---
        p_plenum_kpa = cond.plenum_pressure_kpa * float(cal.get("pressure_factor", 1.0))
        gap_mm = cond.nozzle_gap_mm * float(cal.get("gap_factor", 1.0))
        distance_mm = cond.nozzle_strip_distance_mm * float(cal.get("distance_factor", 1.0))
        speed_mpm = cond.line_speed_mpm * float(cal.get("speed_factor", 1.0))
        scale_factor = float(cal.get("scale_factor", 1.0))
        offset_gm2 = float(cal.get("offset_gm2", 0.0))

        # --- 1. ガス状態・ノズル出口流れ ---
        gas_temp_c = self.material_config["constants"]["default_gas_temperature_c"]
        gas_state = calculate_gas_state(cond.gas_type, p_plenum_kpa, gas_temp_c, self.material_config)
        flow = nozzle_exit_flow(p_plenum_kpa, gas_state, self.coefficients)

        # --- 2. 壁面圧力・せん断分布（プレナムゲージ圧スケール） ---
        jet = calculate_jet_field(
            kpa_to_pa(p_plenum_kpa),
            gap_mm,
            distance_mm,
            self.coefficients,
        )

        # --- 3. 薄膜方程式（最大流束原理） ---
        zinc = self.material_config["zinc"]
        gravity = float(self.material_config["constants"]["gravity_m_s2"])
        rho_l = float(zinc["density_liquid_kg_m3"])
        mu_l = zinc_viscosity_pa_s(cond.bath_temp_c, self.material_config)
        edge_factor = edge_effect_factor_from_width(cond.strip_width_mm, self.coefficients)
        film = solve_film(
            jet.x_m,
            jet.pressure_gradient_pa_m,
            jet.shear_pa,
            speed_mpm,
            rho_l,
            mu_l,
            gravity,
            edge_effect_factor=edge_factor,
            compute_profile=include_profile,
        )
        h_f = film.film_thickness_m * scale_factor

        # --- 4. 目付（質量保存: 液膜質量 = rho_L × h_f） ---
        film_cfg = self.coefficients.get("film", {})
        alloy_floor_gm2 = float(film_cfg.get("unwipeable_alloy_layer_gm2", 9.0))
        cw_raw = rho_l * h_f * 1000.0  # [g/m^2] (h in m, rho in kg/m^3)
        cw_one = max(cw_raw + offset_gm2, alloy_floor_gm2)
        alloy_floor_active = cw_raw + offset_gm2 < alloy_floor_gm2
        cw_both = 2.0 * cw_one
        t_um = m_to_um(h_f)

        # --- 5. スプラッシュ限界 ---
        splash = calculate_splash_risk(
            flow.exit_density_kg_m3,
            flow.exit_velocity_m_s,
            gap_mm,
            distance_mm,
            p_plenum_kpa,
            speed_mpm,
            h_f,
            self.material_config,
            self.coefficients,
            bath_temp_c=cond.bath_temp_c,
        )

        # --- 目標との比較 ---
        target_cw = float(cond.target_cw_one_side_gm2)
        achievement_ratio = (cw_one / target_cw) if target_cw > 0.0 else 0.0
        gap_to_target_gm2 = cw_one - target_cw
        target_status = self._evaluate_target_status(achievement_ratio)

        # --- 警告 ---
        warnings = [
            "文献検証済み相関に基づく1次元物理モデルの推定です。",
            "実機条件に対しては必ず実績校正・検証を行ってください。",
        ]
        if flow.mach_warning:
            warnings.insert(0, f"ノズル出口Mach数 {flow.mach:.2f}（>0.3）。圧縮性を考慮した等エントロピ流で評価しています。")
        if flow.choked:
            warnings.insert(0, "ノズル出口がチョーク（臨界流）条件です。噴射圧力を上げても出口速度は増加しません。")
        if splash.level in {"中", "高"}:
            warnings.append(
                f"スプラッシュ注意レベル：{splash.level}（We/We* = {splash.score:.2f}、1.0以上で文献の発生域）。"
            )
        if alloy_floor_active:
            warnings.append(
                f"推定値が払拭不能なFe-Zn合金層相当（約{alloy_floor_gm2:.0f} g/m²）を下回ったため、下限値を表示しています。"
            )
        if target_status == "未達":
            warnings.append(f"推定目付が目標を下回っています（達成率 {achievement_ratio * 100:.1f}%）。")
        elif target_status == "過剰":
            warnings.append(f"推定目付が目標を上回っています（達成率 {achievement_ratio * 100:.1f}%）。")
        melting_point = float(zinc.get("melting_point_c", 419.5))
        if cond.bath_temp_c < melting_point:
            warnings.append(f"浴温が亜鉛融点（{melting_point:.1f}℃）を下回っています。入力値を確認してください。")
        if cond.nozzle_strip_distance_mm < 3.0:
            warnings.append("ノズル-鋼板距離が3mm未満です。スプラッシュ・接触リスクが高く、相関の検証範囲外です。")

        validity = self._evaluate_model_validity(cond, flow, jet, splash, cw_one)
        warnings.extend(validity["notes"])

        film_profile_um = m_to_um(film.film_profile_m * scale_factor)

        return {
            "condition": cond.model_dump(),
            "gas_state": gas_state,
            "exit_velocity_m_s": flow.exit_velocity_m_s,
            "mach": flow.mach,
            "mach_warning": flow.mach_warning,
            "dynamic_pressure_kpa": pa_to_kpa(flow.dynamic_pressure_pa),
            "nozzle_flow_model": flow.model_name,
            "nozzle_choked": flow.choked,
            "nozzle_pressure_ratio": flow.pressure_ratio,
            "exit_density_kg_m3": flow.exit_density_kg_m3,
            "x_mm": jet.x_m * 1000.0,
            "pressure_kpa": jet.pressure_pa / 1000.0,
            "pressure_gradient_pa_m": jet.pressure_gradient_pa_m,
            "peak_pressure_kpa": jet.peak_pressure_pa / 1000.0,
            "standoff_ratio": jet.standoff_ratio,
            "jet_regime": jet.regime,
            "pressure_standoff_attenuation": jet.peak_pressure_pa / max(kpa_to_pa(p_plenum_kpa), 1.0e-9),
            "shear_pa": jet.shear_pa,
            "peak_shear_pa": jet.peak_shear_pa,
            "shear_standoff_attenuation": jet.peak_shear_pa / max(kpa_to_pa(p_plenum_kpa), 1.0e-9),
            "film_profile_um": film_profile_um,
            "film_thickness_um": t_um,
            "cw_one_side_gm2": cw_one,
            "cw_both_sides_gm2": cw_both,
            "base_film_um": m_to_um(film.base_thickness_m),
            "wiping_strength": film.wiping_strength,
            "dimensionless_flux": film.dimensionless_flux,
            "metering_position_mm": film.metering_position_m * 1000.0,
            "g_max": film.g_max,
            "s_at_metering": film.s_at_metering,
            "edge_effect_factor": film.edge_effect_factor,
            "bath_viscosity_pa_s": film.bath_viscosity_pa_s,
            "alloy_floor_active": alloy_floor_active,
            "target_cw_one_side_gm2": target_cw,
            "achievement_ratio": achievement_ratio,
            "gap_to_target_gm2": gap_to_target_gm2,
            "target_status": target_status,
            "splash_level": splash.level,
            "splash_score": splash.score,
            "weber_number": splash.weber_number,
            "critical_weber": splash.critical_weber,
            "wall_jet_velocity_m_s": splash.wall_jet_velocity_m_s,
            "liquid_reynolds": splash.liquid_reynolds,
            "model_confidence": validity["confidence"],
            "uncertainty_relative_percent": validity["uncertainty_percent"],
            "cw_one_side_low_gm2": cw_one * (1.0 - validity["uncertainty_percent"] / 100.0),
            "cw_one_side_high_gm2": cw_one * (1.0 + validity["uncertainty_percent"] / 100.0),
            "model_validity_notes": validity["notes"],
            "warnings": warnings,
        }

    @staticmethod
    def _evaluate_target_status(achievement_ratio: float) -> str:
        """達成率を「未達 / 適正 / 過剰」に区分する。閾値は±10%。"""
        if achievement_ratio < 0.90:
            return "未達"
        if achievement_ratio > 1.10:
            return "過剰"
        return "適正"

    def _evaluate_model_validity(self, cond: AnalysisCondition, flow, jet, splash, cw_one: float) -> dict[str, Any]:
        """モデル適用性と概算不確かさを評価する。

        基準: Elsaadawy 2007 は実機CGL（目付≦75 g/m²・近接場）でコイル平均偏差≦8%。
        本実装は分布相関＋数値解で同系統のため、検証域内の基準不確かさを12%とし、
        外挿要因ごとに加算する。精度保証ではなく操業判断の補助情報。
        """
        cfg = self.coefficients.get("uncertainty", {})
        uncertainty = float(cfg.get("base_relative_percent", 12.0))
        notes: list[str] = []

        if jet.standoff_ratio > 8.0:
            uncertainty += float(cfg.get("far_field_add_percent", 8.0))
            notes.append(
                f"Z/d = {jet.standoff_ratio:.1f}（>8）のため遠方場相関（Ellen-Tu系）を使用しています。近接場より検証データが少ない領域です。"
            )
        if jet.standoff_ratio > float(cfg.get("caution_h_over_b_max", 15.0)):
            uncertainty += float(cfg.get("very_high_standoff_add_percent", 8.0))
            notes.append("ノズル-鋼板距離が大きく、外挿比率が高い条件です。")
        if cw_one > float(cfg.get("validated_cw_max_gm2", 75.0)):
            uncertainty += float(cfg.get("high_cw_add_percent", 8.0))
            notes.append("推定目付が75 g/m²超です。文献モデルの検証精度が低下する領域です（Elsaadawy 2007）。")
        if flow.choked:
            uncertainty += float(cfg.get("choked_add_percent", 8.0))
            notes.append("チョーク条件のため、衝突圧スケールの線形性が崩れる可能性があります。")
        elif flow.mach > 0.8:
            uncertainty += float(cfg.get("high_mach_add_percent", 5.0))
            notes.append("Mach数が0.8超です。衝突圧相関は非圧縮CFDベースのため誤差が増えます。")
        if splash.score >= 1.0:
            uncertainty += float(cfg.get("splash_high_add_percent", 8.0))
            notes.append("スプラッシュ発生域のため、液滴飛散・ノズル付着により実目付が乱れる可能性があります。")
        if cond.nozzle_strip_distance_mm < 3.0:
            uncertainty += float(cfg.get("low_distance_add_percent", 10.0))
        if cond.bath_temp_c < float(cfg.get("low_bath_temp_c", 450.0)):
            uncertainty += float(cfg.get("bath_temperature_add_percent", 5.0))
            notes.append(
                "浴温450℃未満では鋼帯近傍の微視的凝固により実効粘度が上昇し、実目付が推定より厚くなる傾向があります（JFE 2023）。"
            )

        high_max = float(cfg.get("confidence_high_max_percent", 18.0))
        medium_max = float(cfg.get("confidence_medium_max_percent", 30.0))
        if uncertainty <= high_max:
            confidence = "高"
        elif uncertainty <= medium_max:
            confidence = "中"
        else:
            confidence = "低"
        return {
            "confidence": confidence,
            "uncertainty_percent": min(80.0, max(5.0, uncertainty)),
            "notes": notes,
        }

    def analyze_for_csv_row(self, row: dict[str, Any], calibration: dict | None = None) -> dict[str, Any]:
        def _required_float(key: str) -> float:
            if key not in row or row[key] is None or (isinstance(row[key], float) and np.isnan(row[key])):
                raise ValueError(f"CSV行に必須数値列 '{key}' が欠損しています。csv_service の dropna 設定を確認してください。")
            return float(row[key])

        condition = {
            "project_name": "CSV_Calibration",
            "plenum_pressure_kpa": _required_float("plenum_pressure_kpa"),
            "nozzle_gap_mm": _required_float("nozzle_gap_mm"),
            "nozzle_strip_distance_mm": _required_float("nozzle_strip_distance_mm"),
            "gas_type": normalize_gas_type(str(row.get("gas_type", "nitrogen"))),
            "line_speed_mpm": _required_float("line_speed_mpm"),
            "strip_width_mm": _required_float("strip_width_mm"),
            "target_cw_one_side_gm2": float(row.get("measured_cw_one_side_gm2", 60.0)),
            "bath_temp_c": _required_float("bath_temp_c"),
        }
        return self.analyze(condition, calibration=calibration, include_profile=False)

    def sweep(
        self,
        base_condition: AnalysisCondition,
        pair: str,
        metric: str,
        x_points: int = 31,
        y_points: int = 31,
        progress_callback=None,
        is_cancelled=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
        """パラメトリックスイープを実行する。"""
        base = base_condition.model_dump()
        # 旧名称（プレナム圧/ノズルギャップ）と新名称（噴射圧力/ノズルすき間）の両方を受理する。
        if "通板速度" in pair:
            x_name, y_name = "line_speed_mpm", "plenum_pressure_kpa"
            x_values = np.linspace(30.0, 300.0, x_points)
            y_values = np.linspace(5.0, 80.0, y_points)
            x_label, y_label = "通板速度 [m/min]", "噴射圧力 [kPa]"
        elif "距離" in pair:
            x_name, y_name = "nozzle_strip_distance_mm", "plenum_pressure_kpa"
            x_values = np.linspace(2.0, 30.0, x_points)
            y_values = np.linspace(5.0, 80.0, y_points)
            x_label, y_label = "ノズル-鋼板距離 [mm]", "噴射圧力 [kPa]"
        else:
            x_name, y_name = "nozzle_gap_mm", "plenum_pressure_kpa"
            x_values = np.linspace(0.2, 3.0, x_points)
            y_values = np.linspace(5.0, 80.0, y_points)
            x_label, y_label = "ノズルすき間 [mm]", "噴射圧力 [kPa]"

        values = np.zeros((len(y_values), len(x_values)))
        total = len(y_values) * len(x_values)
        sweep_calibration = load_calibration_coefficients()
        done = 0
        for yi, y_value in enumerate(y_values):
            for xi, x_value in enumerate(x_values):
                if is_cancelled is not None and is_cancelled():
                    return x_values, y_values, values, x_label, y_label
                cond = dict(base)
                cond[x_name] = float(x_value)
                cond[y_name] = float(y_value)
                result = self.analyze(cond, calibration=sweep_calibration, include_profile=False)
                if metric == "推定膜厚 [µm]":
                    values[yi, xi] = result["film_thickness_um"]
                elif metric == "スプラッシュ注意レベル":
                    values[yi, xi] = {"低": 1.0, "中": 2.0, "高": 3.0}[result["splash_level"]]
                else:
                    values[yi, xi] = result["cw_one_side_gm2"]
                done += 1
                if progress_callback is not None and xi == len(x_values) - 1:
                    progress_callback(done, total)
        if progress_callback is not None:
            progress_callback(total, total)
        return x_values, y_values, values, x_label, y_label

    def reverse_solve(
        self,
        target_cw_gm2: float,
        line_speed_mpm: float,
        strip_width_mm: float,
        bath_temp_c: float,
        gas_type: str = "air",
        fixed_pressure_kpa: float | None = None,
        fixed_gap_mm: float | None = None,
        fixed_distance_mm: float | None = None,
        calibration: dict | None = None,
        n_grid: int = 12,
        n_candidates: int = 3,
    ) -> dict[str, Any]:
        """目標片面目付からワイピング条件を逆算する（グリッドサーチ＋精密化）。

        Parameters
        ----------
        target_cw_gm2 : 目標片面目付 [g/m²]
        line_speed_mpm / strip_width_mm / bath_temp_c : 固定する操業条件
        fixed_* : None で探索変数、数値で固定値
        n_grid   : 各次元のグリッド数（自由変数 k 本 → n_grid^k 評価）
        n_candidates : 返す候補数

        Returns
        -------
        dict with keys:
            "candidates" : list[dict]  — 上位候補（pressure/gap/distance/predicted_cw/splash等）
            "feasible"   : bool        — 最良解が目標の5%以内か
            "message"    : str
        """
        cal = calibration if calibration is not None else load_calibration_coefficients()
        # AnalysisCondition の valid range を外れないよう target をクランプ。
        target_cw_clamped = float(np.clip(target_cw_gm2, 10.0, 300.0))
        base_kwargs: dict[str, Any] = {
            "line_speed_mpm": float(np.clip(line_speed_mpm, 10.0, 300.0)),
            "strip_width_mm": float(np.clip(strip_width_mm, 300.0, 2000.0)),
            "bath_temp_c": float(np.clip(bath_temp_c, 420.0, 500.0)),
            "gas_type": normalize_gas_type(gas_type),
            "target_cw_one_side_gm2": target_cw_clamped,
            "project_name": "ReverseCalc",
        }

        # --- グリッド定義 ---
        p_range = (
            np.array([float(fixed_pressure_kpa)])
            if fixed_pressure_kpa is not None
            else np.logspace(np.log10(2.0), np.log10(90.0), n_grid)
        )
        g_range = (
            np.array([float(fixed_gap_mm)])
            if fixed_gap_mm is not None
            else np.linspace(0.3, 2.5, n_grid)
        )
        d_range = (
            np.array([float(fixed_distance_mm)])
            if fixed_distance_mm is not None
            else np.logspace(np.log10(3.0), np.log10(30.0), n_grid)
        )

        # --- グリッドサーチ ---
        grid_results: list[tuple[float, dict]] = []
        for p in p_range:
            for g in g_range:
                for d in d_range:
                    try:
                        cond_dict = dict(base_kwargs)
                        cond_dict["plenum_pressure_kpa"] = float(p)
                        cond_dict["nozzle_gap_mm"] = float(g)
                        cond_dict["nozzle_strip_distance_mm"] = float(d)
                        r = self.analyze(cond_dict, calibration=cal, include_profile=False)
                        err = abs(r["cw_one_side_gm2"] - target_cw_gm2)
                        grid_results.append((err, r))
                    except Exception:
                        continue

        if not grid_results:
            return {
                "candidates": [],
                "feasible": False,
                "message": "解を見つけられませんでした。入力条件の範囲を確認してください。",
            }

        grid_results.sort(key=lambda t: t[0])

        # --- scipy で精密化（利用可能な場合のみ） ---
        refined: list[tuple[float, dict]] = []
        seen: set[tuple] = set()

        for raw_err, raw_r in grid_results[:12]:
            rc = raw_r["condition"]
            p0 = rc["plenum_pressure_kpa"]
            g0 = rc["nozzle_gap_mm"]
            d0 = rc["nozzle_strip_distance_mm"]
            best_err, best_r = raw_err, raw_r

            try:
                from scipy.optimize import minimize as _minimize  # type: ignore

                p_bounds = (
                    (float(fixed_pressure_kpa), float(fixed_pressure_kpa))
                    if fixed_pressure_kpa is not None
                    else (1.0, 100.0)
                )
                g_bounds = (
                    (float(fixed_gap_mm), float(fixed_gap_mm))
                    if fixed_gap_mm is not None
                    else (0.2, 3.0)
                )
                d_bounds = (
                    (float(fixed_distance_mm), float(fixed_distance_mm))
                    if fixed_distance_mm is not None
                    else (1.0, 50.0)
                )

                def _obj(x: np.ndarray) -> float:
                    try:
                        pp = float(np.clip(x[0], *p_bounds))
                        gg = float(np.clip(x[1], *g_bounds))
                        dd = float(np.clip(x[2], *d_bounds))
                        cd: dict[str, Any] = dict(base_kwargs)
                        cd["plenum_pressure_kpa"] = pp
                        cd["nozzle_gap_mm"] = gg
                        cd["nozzle_strip_distance_mm"] = dd
                        rr = self.analyze(cd, calibration=cal, include_profile=False)
                        return float((rr["cw_one_side_gm2"] - target_cw_gm2) ** 2)
                    except Exception:
                        return 1.0e9

                res = _minimize(
                    _obj,
                    [p0, g0, d0],
                    method="L-BFGS-B",
                    bounds=[p_bounds, g_bounds, d_bounds],
                    options={"maxiter": 60, "ftol": 1.0e-8},
                )
                pp = float(np.clip(res.x[0], *p_bounds))
                gg = float(np.clip(res.x[1], *g_bounds))
                dd = float(np.clip(res.x[2], *d_bounds))
                cd_ref: dict[str, Any] = dict(base_kwargs)
                cd_ref["plenum_pressure_kpa"] = pp
                cd_ref["nozzle_gap_mm"] = gg
                cd_ref["nozzle_strip_distance_mm"] = dd
                r_ref = self.analyze(cd_ref, calibration=cal, include_profile=False)
                err_ref = abs(r_ref["cw_one_side_gm2"] - target_cw_gm2)
                if err_ref < best_err:
                    best_err, best_r = err_ref, r_ref
            except Exception:
                pass

            key = (
                round(best_r["condition"]["plenum_pressure_kpa"], 1),
                round(best_r["condition"]["nozzle_gap_mm"], 2),
                round(best_r["condition"]["nozzle_strip_distance_mm"], 1),
            )
            if key not in seen:
                seen.add(key)
                refined.append((best_err, best_r))

        refined.sort(key=lambda t: t[0])

        # --- 候補リスト生成 ---
        candidates: list[dict[str, Any]] = []
        for i, (err, r) in enumerate(refined[:n_candidates]):
            c = r["condition"]
            candidates.append(
                {
                    "rank": i + 1,
                    "plenum_pressure_kpa": round(c["plenum_pressure_kpa"], 1),
                    "nozzle_gap_mm": round(c["nozzle_gap_mm"], 2),
                    "nozzle_strip_distance_mm": round(c["nozzle_strip_distance_mm"], 1),
                    "predicted_cw_gm2": round(r["cw_one_side_gm2"], 1),
                    "cw_error_gm2": round(err, 2),
                    "splash_level": r["splash_level"],
                    "splash_score": round(r["splash_score"], 3),
                    "mach": round(r["mach"], 2),
                    "model_confidence": r["model_confidence"],
                    "uncertainty_percent": round(r["uncertainty_relative_percent"], 1),
                    "full_condition": c,
                }
            )

        best_err = candidates[0]["cw_error_gm2"] if candidates else 999.0
        feasible = best_err <= target_cw_gm2 * 0.05
        if best_err > target_cw_gm2 * 0.3:
            message = (
                f"目標目付 {target_cw_gm2:.1f} g/m² に近い条件を見つけられませんでした"
                f"（最小誤差 {best_err:.1f} g/m²）。固定条件を解除するか操業範囲を見直してください。"
            )
        elif best_err > target_cw_gm2 * 0.1:
            message = (
                f"近似解を提示します（最小誤差 {best_err:.1f} g/m²）。"
                "精度が低い場合は固定条件の見直しをお勧めします。"
            )
        else:
            message = f"目標目付 {target_cw_gm2:.1f} g/m² に対する推奨ワイピング条件を算出しました。"

        return {"candidates": candidates, "feasible": feasible, "message": message}

    def representative_conditions(self, base_condition: AnalysisCondition) -> list[dict[str, Any]]:
        base = base_condition.model_dump()
        scenarios = [
            ("低圧・低速側", {"plenum_pressure_kpa": 20.0, "line_speed_mpm": 60.0}),
            ("基準条件", {}),
            ("高速側・スプラッシュ注意", {"line_speed_mpm": min(240.0, base["line_speed_mpm"] * 2.0)}),
            ("高圧・近接側", {"plenum_pressure_kpa": 45.0, "nozzle_strip_distance_mm": 5.0}),
            ("低圧・大ギャップ側", {"plenum_pressure_kpa": 15.0, "nozzle_gap_mm": 1.5}),
        ]
        rows: list[dict[str, Any]] = []
        rep_calibration = load_calibration_coefficients()
        for index, (note, changes) in enumerate(scenarios, start=1):
            cond = dict(base)
            cond.update(changes)
            result = self.analyze(cond, calibration=rep_calibration, include_profile=False)
            rows.append(
                {
                    "No.": index,
                    "噴射圧力 [kPa]": cond["plenum_pressure_kpa"],
                    "通板速度 [m/min]": cond["line_speed_mpm"],
                    "ノズル-鋼板距離 [mm]": cond["nozzle_strip_distance_mm"],
                    "ノズルすき間 [mm]": cond["nozzle_gap_mm"],
                    "推定片面目付 [g/m²]": result["cw_one_side_gm2"],
                    "推定膜厚 [µm]": result["film_thickness_um"],
                    "スプラッシュ注意レベル": result["splash_level"],
                    "備考": note,
                }
            )
        return rows
