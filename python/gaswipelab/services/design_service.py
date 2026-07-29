"""design_service.py — 操業条件の設計（逆算）サービス

物理モデル（analysis_service 以下）には一切手を加えず、その上で
「目標片面めっき付着量を満たす操業条件」を探索する層だけを提供する。
計算は既存の AnalysisService.analyze() をそのまま呼ぶため、
モデル精度・理論・校正係数の扱いは従来と同一。

提供する2機能:

1. design()
   設備・操業側で固定したい条件（ノズルすき間・通板速度・浴温・板幅・ガス種）
   を与え、目標片面めっき付着量を満たす「噴射圧力」「ノズル距離」を求める。
   安全に達成できる解が無い場合は、
       通板速度 → ノズルすき間 → 浴温
   の優先順で「どう変えればよいか」を具体値で提案する。
   板幅・ガス種は製品仕様／設備仕様であり、変更提案の対象にしない。

2. quick_design()
   ガス種・板幅・目標片面めっき付着量だけから、実操業として妥当な条件一式
   （ノズルすき間・通板速度・浴温・噴射圧力・ノズル距離）を提案する。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from gaswipelab.models.gas_properties import normalize_gas_type
from gaswipelab.services.analysis_service import AnalysisService
from gaswipelab.services.settings_service import load_calibration_coefficients

# ------------------------------------------------------------------
# 探索範囲（いずれもモデル有効域 AnalysisCondition の内側に取った実操業レンジ）
# ------------------------------------------------------------------
PRESSURE_MIN_KPA = 5.0
PRESSURE_MAX_KPA = 80.0

DISTANCE_CANDIDATES_MM: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0)
DISTANCE_MIN_MM = DISTANCE_CANDIDATES_MM[0]
DISTANCE_MAX_MM = DISTANCE_CANDIDATES_MM[-1]
# 粗探索用の間引き距離。到達可能範囲の判定と食い違わないよう両端を必ず含める。
DISTANCE_SCAN_MM: tuple[float, ...] = (6.0, 10.0, 14.0, 18.0, 22.0)

SPEED_MIN_MPM, SPEED_MAX_MPM = 40.0, 240.0
SPEED_COARSE_STEP, SPEED_FINE_STEP = 20.0, 5.0

GAP_MIN_MM, GAP_MAX_MM = 0.5, 2.5
GAP_COARSE_STEP, GAP_FINE_STEP = 0.4, 0.1

BATH_MIN_C, BATH_MAX_C = 450.0, 475.0
BATH_COARSE_STEP, BATH_FINE_STEP = 5.0, 1.0

# 操業しやすさの推奨帯（スコアリングのみに使う。外れても不可ではない）
PREFERRED_PRESSURE_KPA = (12.0, 55.0)
PREFERRED_DISTANCE_MM = (8.0, 16.0)

# 変更提案が目指すスプラッシュ余裕。発生限界 We/We* = 1.0 に対して 15% の余裕を取る。
SPLASH_MARGIN_SCORE = 0.85

# かんたん設計で許容する噴射圧力帯。上下どちらにも調整余地を残す。
QUICK_PRESSURE_BAND_KPA = (10.0, 65.0)

# 変更提案の優先順位。板幅・ガス種は含めない（製品仕様・設備仕様のため）。
ADJUSTABLE_PARAMETERS: tuple[tuple[str, str, str, int], ...] = (
    ("line_speed_mpm", "通板速度", "m/min", 0),
    ("nozzle_gap_mm", "ノズルすき間", "mm", 2),
    ("bath_temp_c", "浴温", "℃", 1),
)

FIXED_PARAMETERS_NOTE = "板幅とガス種は製品仕様・設備仕様のため、変更提案の対象にしていません。"

_CONFIDENCE_PENALTY = {"高": 0.0, "中": 0.8, "低": 2.0}
_SPLASH_PENALTY = {"低": 0.0, "中": 1.0, "高": 4.0}


class DesignService:
    """目標めっき付着量から操業条件を逆算するサービス。"""

    def __init__(self, analysis_service: AnalysisService | None = None) -> None:
        self.analysis = analysis_service or AnalysisService()

    # ==============================================================
    # 内部: 単発評価
    # ==============================================================
    def _evaluate(self, base: dict, pressure_kpa: float, distance_mm: float, cal: dict) -> dict[str, Any]:
        cond = dict(base)
        cond["plenum_pressure_kpa"] = float(np.clip(pressure_kpa, 1.0, 100.0))
        cond["nozzle_strip_distance_mm"] = float(np.clip(distance_mm, 1.0, 50.0))
        return self.analysis.analyze(cond, calibration=cal, include_profile=False)

    def _cw(self, base: dict, pressure_kpa: float, distance_mm: float, cal: dict) -> float:
        return float(self._evaluate(base, pressure_kpa, distance_mm, cal)["cw_one_side_gm2"])

    @staticmethod
    def _normalize_base(condition: dict, target_gm2: float) -> dict[str, Any]:
        """探索用の基準条件を作る（圧力・距離は探索で上書きするため仮値）。"""
        base = {
            "project_name": "ConditionDesign",
            "gas_type": normalize_gas_type(str(condition.get("gas_type", "air"))),
            "nozzle_gap_mm": float(np.clip(float(condition.get("nozzle_gap_mm", 1.0)), 0.2, 3.0)),
            "line_speed_mpm": float(np.clip(float(condition.get("line_speed_mpm", 120.0)), 10.0, 300.0)),
            "strip_width_mm": float(np.clip(float(condition.get("strip_width_mm", 1200.0)), 300.0, 2000.0)),
            "bath_temp_c": float(np.clip(float(condition.get("bath_temp_c", 460.0)), 420.0, 500.0)),
            "target_cw_one_side_gm2": float(np.clip(target_gm2, 10.0, 300.0)),
            "plenum_pressure_kpa": 30.0,
            "nozzle_strip_distance_mm": 10.0,
        }
        return base

    # ==============================================================
    # 内部: 圧力の逆算（二分法）
    # ==============================================================
    def solve_pressure_for_target(
        self,
        base: dict,
        distance_mm: float,
        target_gm2: float,
        cal: dict,
        max_iter: int = 22,
    ) -> tuple[float, float, dict] | None:
        """ノズル距離を固定し、目標めっき付着量になる噴射圧力を二分法で求める。

        片面めっき付着量は噴射圧力に対して単調減少（圧力↑ → ワイピング強 → めっき付着量↓）。
        探索範囲内で到達できない場合は None を返す。

        Returns
        -------
        (誤差 [g/m²], 噴射圧力 [kPa], analyze() の結果) または None
        """
        tolerance = max(0.15, 0.004 * target_gm2)
        low, high = PRESSURE_MIN_KPA, PRESSURE_MAX_KPA
        cw_at_low = self._cw(base, low, distance_mm, cal)     # 最も厚い側
        cw_at_high = self._cw(base, high, distance_mm, cal)   # 最も薄い側
        if target_gm2 > cw_at_low or target_gm2 < cw_at_high:
            return None

        best: tuple[float, float, dict] | None = None
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            result = self._evaluate(base, mid, distance_mm, cal)
            error = float(result["cw_one_side_gm2"]) - target_gm2
            if best is None or abs(error) < best[0]:
                best = (abs(error), mid, result)
            if abs(error) <= tolerance:
                break
            if error > 0.0:
                low = mid   # 厚すぎる → 圧力を上げる
            else:
                high = mid  # 薄すぎる → 圧力を下げる
        return best

    # ==============================================================
    # 内部: 候補生成
    # ==============================================================
    def _candidates(
        self,
        base: dict,
        target_gm2: float,
        cal: dict,
        distances: tuple[float, ...] = DISTANCE_CANDIDATES_MM,
        with_sensitivity: bool = True,
    ) -> list[dict[str, Any]]:
        allowed_error = max(0.5, 0.02 * target_gm2)
        candidates: list[dict[str, Any]] = []
        for distance in distances:
            solved = self.solve_pressure_for_target(base, distance, target_gm2, cal)
            if solved is None:
                continue
            error, pressure, result = solved
            if error > allowed_error:
                continue
            candidates.append(
                self._build_candidate(base, pressure, distance, result, error, target_gm2, cal, with_sensitivity)
            )
        candidates.sort(key=lambda c: c["score"])
        return candidates

    def _build_candidate(
        self,
        base: dict,
        pressure: float,
        distance: float,
        result: dict,
        error: float,
        target_gm2: float,
        cal: dict,
        with_sensitivity: bool,
    ) -> dict[str, Any]:
        candidate: dict[str, Any] = {
            "plenum_pressure_kpa": round(float(pressure), 1),
            "nozzle_strip_distance_mm": round(float(distance), 1),
            "nozzle_gap_mm": round(float(base["nozzle_gap_mm"]), 2),
            "line_speed_mpm": round(float(base["line_speed_mpm"]), 0),
            "bath_temp_c": round(float(base["bath_temp_c"]), 1),
            "strip_width_mm": round(float(base["strip_width_mm"]), 0),
            "gas_type": base["gas_type"],
            "predicted_cw_gm2": round(float(result["cw_one_side_gm2"]), 1),
            "cw_low_gm2": round(float(result["cw_one_side_low_gm2"]), 1),
            "cw_high_gm2": round(float(result["cw_one_side_high_gm2"]), 1),
            "error_gm2": round(float(error), 2),
            "film_thickness_um": round(float(result["film_thickness_um"]), 2),
            "splash_level": result["splash_level"],
            "splash_score": round(float(result["splash_score"]), 3),
            "mach": round(float(result["mach"]), 3),
            "choked": bool(result["nozzle_choked"]),
            "model_confidence": result["model_confidence"],
            "uncertainty_percent": round(float(result["uncertainty_relative_percent"]), 1),
            "standoff_ratio": round(float(result["standoff_ratio"]), 1),
            "exit_velocity_m_s": round(float(result["exit_velocity_m_s"]), 0),
        }
        candidate["score"] = self._score(result, pressure, distance, error, target_gm2)
        if with_sensitivity:
            candidate["sensitivity"] = self._sensitivity(base, pressure, distance, cal)
        return candidate

    @staticmethod
    def _score(result: dict, pressure: float, distance: float, error: float, target_gm2: float) -> float:
        """操業しやすさの総合ペナルティ（小さいほど良い）。"""
        score = 0.0
        score += _SPLASH_PENALTY.get(result["splash_level"], 2.0)
        score += 3.0 * max(0.0, float(result["splash_score"]) - 0.5)
        score += _CONFIDENCE_PENALTY.get(result["model_confidence"], 1.0)
        if result["nozzle_choked"]:
            score += 5.0

        p_low, p_high = PREFERRED_PRESSURE_KPA
        if pressure < p_low:
            score += (p_low - pressure) / p_low * 2.0
        elif pressure > p_high:
            score += (pressure - p_high) / p_high * 2.5

        z_low, z_high = PREFERRED_DISTANCE_MM
        if distance < z_low:
            # 近すぎる条件は板の反り・接触・ノズル詰まりの実務リスクが高い。
            score += (z_low - distance) / z_low * 4.0
        elif distance > z_high:
            score += (distance - z_high) / z_high * 1.5

        score += 2.0 * error / max(target_gm2, 1.0)
        return float(score)

    def _sensitivity(self, base: dict, pressure: float, distance: float, cal: dict) -> dict[str, float]:
        """現場で使う「1単位動かすとめっき付着量が何 g/m² 変わるか」を中心差分で求める。"""
        d_pressure = max(0.5, 0.03 * pressure)
        d_distance = max(0.3, 0.03 * distance)
        speed = float(base["line_speed_mpm"])
        d_speed = max(2.0, 0.03 * speed)

        cw_p_plus = self._cw(base, pressure + d_pressure, distance, cal)
        cw_p_minus = self._cw(base, pressure - d_pressure, distance, cal)
        cw_z_plus = self._cw(base, pressure, distance + d_distance, cal)
        cw_z_minus = self._cw(base, pressure, distance - d_distance, cal)

        base_up = dict(base)
        base_up["line_speed_mpm"] = float(np.clip(speed + d_speed, 10.0, 300.0))
        base_down = dict(base)
        base_down["line_speed_mpm"] = float(np.clip(speed - d_speed, 10.0, 300.0))
        cw_v_plus = self._cw(base_up, pressure, distance, cal)
        cw_v_minus = self._cw(base_down, pressure, distance, cal)
        speed_span = base_up["line_speed_mpm"] - base_down["line_speed_mpm"]

        return {
            "per_kpa": round((cw_p_plus - cw_p_minus) / (2.0 * d_pressure), 3),
            "per_mm": round((cw_z_plus - cw_z_minus) / (2.0 * d_distance), 3),
            "per_10mpm": round((cw_v_plus - cw_v_minus) / max(speed_span, 1.0e-6) * 10.0, 3),
        }

    # ==============================================================
    # 内部: 到達可能範囲
    # ==============================================================
    def reachable_range(self, base: dict, cal: dict) -> dict[str, float]:
        """現在の固定条件で作れる片面めっき付着量の範囲（探索レンジ内）。"""
        thinnest = self._cw(base, PRESSURE_MAX_KPA, DISTANCE_MIN_MM, cal)
        thickest = self._cw(base, PRESSURE_MIN_KPA, DISTANCE_MAX_MM, cal)
        return {"min_gm2": round(float(thinnest), 1), "max_gm2": round(float(thickest), 1)}

    def _range_covers(self, base: dict, target_gm2: float, cal: dict) -> bool:
        thinnest = self._cw(base, PRESSURE_MAX_KPA, DISTANCE_MIN_MM, cal)
        if target_gm2 < thinnest:
            return False
        thickest = self._cw(base, PRESSURE_MIN_KPA, DISTANCE_MAX_MM, cal)
        return target_gm2 <= thickest

    @staticmethod
    def _is_safe(candidate: dict) -> bool:
        """スプラッシュ発生域（We/We* ≧ 1.0）とチョークを外れているか。"""
        return candidate["splash_level"] != "高" and not candidate["choked"]

    @staticmethod
    def _is_comfortable(candidate: dict) -> bool:
        """変更提案の到達目標。発生限界に対して余裕（15%）を持たせる。"""
        return candidate["splash_score"] <= SPLASH_MARGIN_SCORE and not candidate["choked"]

    # ==============================================================
    # 公開API 1: 条件設計
    # ==============================================================
    def design(
        self,
        condition: dict,
        target_gm2: float,
        calibration: dict | None = None,
    ) -> dict[str, Any]:
        """固定条件のもとで目標めっき付着量を満たす噴射圧力・ノズル距離を求める。

        Returns
        -------
        dict
            status      : "ok"（安全に達成可） / "risk"（達成可だがスプラッシュ高）
                          / "infeasible"（探索範囲で達成不可）
            candidates  : 推奨条件（rank順・最大3件）
            proposals   : 固定条件の変更提案（優先順: 通板速度→すき間→浴温）
            reachable   : 到達可能な片面めっき付着量の範囲
        """
        cal = calibration if calibration is not None else load_calibration_coefficients()
        target = float(np.clip(target_gm2, 10.0, 300.0))
        base = self._normalize_base(condition, target)

        candidates = self._candidates(base, target, cal)
        safe = [c for c in candidates if self._is_safe(c)]
        reachable = self.reachable_range(base, cal)

        if safe:
            status = "ok"
            shown = safe[:3]
            proposals: list[dict[str, Any]] = []
        elif candidates:
            status = "risk"
            shown = candidates[:3]
            proposals = self.suggest_changes(base, target, cal)
        else:
            status = "infeasible"
            shown = []
            proposals = self.suggest_changes(base, target, cal)

        for index, candidate in enumerate(shown, start=1):
            candidate["rank"] = index

        return {
            "status": status,
            "target_gm2": round(target, 1),
            "fixed": {
                "nozzle_gap_mm": base["nozzle_gap_mm"],
                "line_speed_mpm": base["line_speed_mpm"],
                "bath_temp_c": base["bath_temp_c"],
                "strip_width_mm": base["strip_width_mm"],
                "gas_type": base["gas_type"],
            },
            "candidates": shown,
            "proposals": proposals,
            "reachable": reachable,
            "fixed_note": FIXED_PARAMETERS_NOTE,
            "message": self._design_message(status, target, reachable, shown),
        }

    @staticmethod
    def _design_message(status: str, target: float, reachable: dict, candidates: list) -> str:
        if status == "ok":
            best = candidates[0]
            return (
                f"目標 {target:.1f} g/m² は現在の固定条件で達成できます。"
                f"噴射圧力 {best['plenum_pressure_kpa']:.1f} kPa、"
                f"ノズル距離 {best['nozzle_strip_distance_mm']:.1f} mm を推奨します。"
            )
        if status == "risk":
            return (
                f"目標 {target:.1f} g/m² は達成できますが、いずれの条件もスプラッシュ発生域"
                "（We/We* ≧ 1.0）です。下の変更提案の適用を推奨します。"
            )
        return (
            f"現在の固定条件では目標 {target:.1f} g/m² を作れません"
            f"（作れる範囲: {reachable['min_gm2']:.1f} 〜 {reachable['max_gm2']:.1f} g/m²）。"
            "下の変更提案を確認してください。"
        )

    # ==============================================================
    # 公開API 2: 固定条件の変更提案
    # ==============================================================
    def suggest_changes(self, base: dict, target_gm2: float, cal: dict) -> list[dict[str, Any]]:
        """通板速度 → ノズルすき間 → 浴温 の順に変更案を探す。

        優先度の高い項目で解決できた時点で打ち切る。解決できなかった項目は
        found=False として理由表示用に残す（板幅・ガス種は対象外）。
        """
        proposals: list[dict[str, Any]] = []
        for key, label, unit, digits in ADJUSTABLE_PARAMETERS:
            current = float(base[key])
            found = self._search_single_change(base, target_gm2, cal, key)
            if found is None:
                proposals.append(
                    {
                        "parameter": key,
                        "label": label,
                        "unit": unit,
                        "digits": digits,
                        "found": False,
                        "current": round(current, digits),
                        "note": f"{label}の調整だけでは目標を安全に達成できませんでした。",
                    }
                )
                continue
            value, candidate = found
            proposals.append(
                {
                    "parameter": key,
                    "label": label,
                    "unit": unit,
                    "digits": digits,
                    "found": True,
                    "current": round(current, digits),
                    "suggested": round(value, digits),
                    "delta": round(value - current, digits),
                    "candidate": candidate,
                    "note": self._change_note(key, label, current, value, unit, digits),
                }
            )
            break
        else:
            combined = self._search_combined_change(base, target_gm2, cal)
            if combined is not None:
                proposals.append(combined)
        return proposals

    def _search_combined_change(self, base: dict, target_gm2: float, cal: dict) -> dict[str, Any] | None:
        """単独変更で解決しない場合に、通板速度とノズルすき間の同時変更を探す。

        浴温は実操業範囲（450〜475℃）でのめっき付着量への影響が数%と小さく、
        同時変更に加えても解の有無をほとんど変えないため対象にしない。
        """
        current_speed = float(base["line_speed_mpm"])
        current_gap = float(base["nozzle_gap_mm"])
        speed_values, _ = self._trial_values("line_speed_mpm", current_speed)
        gap_values, _ = self._trial_values("nozzle_gap_mm", current_gap)

        combos: list[tuple[float, float, float]] = []
        for speed in [current_speed] + speed_values:
            for gap in [current_gap] + gap_values:
                if speed == current_speed and gap == current_gap:
                    continue
                cost = abs(speed - current_speed) / SPEED_COARSE_STEP + abs(gap - current_gap) / GAP_COARSE_STEP
                combos.append((cost, speed, gap))
        combos.sort(key=lambda t: (t[0], -t[1]))

        evaluated = 0
        for _cost, speed, gap in combos:
            trial = dict(base)
            trial["line_speed_mpm"] = speed
            trial["nozzle_gap_mm"] = gap
            if not self._range_covers(trial, target_gm2, cal):
                continue
            evaluated += 1
            if evaluated > 30:
                break
            candidates = self._candidates(
                trial, target_gm2, cal, distances=DISTANCE_SCAN_MM, with_sensitivity=False
            )
            if not any(self._is_comfortable(c) for c in candidates):
                continue
            full = self._candidates(trial, target_gm2, cal)
            comfortable = [c for c in full if self._is_comfortable(c)]
            if not comfortable:
                continue
            comfortable[0]["rank"] = 1
            return {
                "parameter": "combined",
                "label": "通板速度＋ノズルすき間",
                "unit": "",
                "digits": 2,
                "found": True,
                "current": None,
                "changes": [
                    {"label": "通板速度", "unit": "m/min", "current": round(current_speed, 0), "suggested": round(speed, 0)},
                    {"label": "ノズルすき間", "unit": "mm", "current": round(current_gap, 2), "suggested": round(gap, 2)},
                ],
                "candidate": comfortable[0],
                "note": (
                    f"1項目だけでは達成できないため、通板速度 {current_speed:.0f} → {speed:.0f} m/min と "
                    f"ノズルすき間 {current_gap:.2f} → {gap:.2f} mm の同時変更を提案します。"
                ),
            }
        return None

    @staticmethod
    def _change_note(key: str, label: str, current: float, value: float, unit: str, digits: int) -> str:
        direction = "上げる" if value > current else "下げる"
        reason = {
            "line_speed_mpm": "通板速度は持ち上げ液膜量を直接変えるため、めっき付着量調整の第一手段です。",
            "nozzle_gap_mm": "ノズルすき間は噴流の強さと広がりを変えます。機械調整が必要です。",
            "bath_temp_c": (
                "浴温は亜鉛の粘度を変えますが、実操業範囲でのめっき付着量への影響は数%と小さく、"
                "浴全体・合金化にも影響するため最後の手段です。"
            ),
        }[key]
        return (
            f"{label}を {current:.{digits}f} → {value:.{digits}f} {unit} に{direction}と、"
            f"目標を安全に達成できます。{reason}"
        )

    def _search_single_change(
        self,
        base: dict,
        target_gm2: float,
        cal: dict,
        key: str,
    ) -> tuple[float, dict] | None:
        """1項目だけを動かして安全に達成できる値を、現在値に最も近い側から探す。"""
        current = float(base[key])
        coarse_values, fine_step = self._trial_values(key, current)

        hit = None
        for value in coarse_values:
            if self._change_works(base, target_gm2, cal, key, value):
                hit = value
                break
        if hit is None:
            return None

        # 現在値と粗探索ヒットの間を細かく詰め、変更量を最小化する。
        for value in self._refine_values(current, hit, fine_step):
            if self._change_works(base, target_gm2, cal, key, value):
                hit = value
                break

        trial = dict(base)
        trial[key] = hit
        candidates = self._candidates(trial, target_gm2, cal)
        comfortable = [c for c in candidates if self._is_comfortable(c)]
        if not comfortable:
            return None
        comfortable[0]["rank"] = 1
        return hit, comfortable[0]

    def _change_works(self, base: dict, target_gm2: float, cal: dict, key: str, value: float) -> bool:
        trial = dict(base)
        trial[key] = float(value)
        if not self._range_covers(trial, target_gm2, cal):
            return False
        candidates = self._candidates(
            trial, target_gm2, cal, distances=DISTANCE_SCAN_MM, with_sensitivity=False
        )
        return any(self._is_comfortable(c) for c in candidates)

    @staticmethod
    def _trial_values(key: str, current: float) -> tuple[list[float], float]:
        """粗探索の候補値（現在値に近い順）と細探索の刻みを返す。"""
        if key == "line_speed_mpm":
            low, high, coarse, fine = SPEED_MIN_MPM, SPEED_MAX_MPM, SPEED_COARSE_STEP, SPEED_FINE_STEP
        elif key == "nozzle_gap_mm":
            low, high, coarse, fine = GAP_MIN_MM, GAP_MAX_MM, GAP_COARSE_STEP, GAP_FINE_STEP
        else:
            low, high, coarse, fine = BATH_MIN_C, BATH_MAX_C, BATH_COARSE_STEP, BATH_FINE_STEP

        count = int(round((high - low) / coarse)) + 1
        values = [round(low + i * coarse, 4) for i in range(count)]
        values = [v for v in values if abs(v - current) >= coarse * 0.5]
        # 現在値に近い順。同距離なら生産性の高い（速い）側を優先する。
        values.sort(key=lambda v: (abs(v - current), -v if key == "line_speed_mpm" else v))
        return values, fine

    @staticmethod
    def _refine_values(current: float, hit: float, step: float) -> list[float]:
        """現在値と粗探索ヒットの間（ヒット寄りを除く）を現在値に近い順で返す。"""
        if abs(hit - current) <= step:
            return []
        sign = 1.0 if hit > current else -1.0
        values: list[float] = []
        value = current + sign * step
        while (hit - value) * sign > step * 0.5:
            values.append(round(value, 4))
            value += sign * step
        return values

    # ==============================================================
    # 公開API 3: かんたん設計
    # ==============================================================
    def quick_design(
        self,
        gas_type: str,
        strip_width_mm: float,
        target_gm2: float,
        bath_temp_c: float = 460.0,
        gap_options: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5),
        calibration: dict | None = None,
    ) -> dict[str, Any]:
        """ガス種・板幅・目標めっき付着量だけから、実操業として妥当な条件一式を提案する。

        考え方: 標準浴温（既定 460℃）を前提に、ノズルすき間ごとに
        「目標を安全に達成できる最大通板速度」を求め、
        生産性（通板速度）と安定性（スプラッシュ余裕・圧力余裕）で順位付けする。

        まず厳しい基準（スプラッシュ低・モデル適用域内・圧力に上下の余裕あり）で探し、
        見つからない場合のみ基準を緩めて再探索し、その旨を message に明示する。
        """
        cal = calibration if calibration is not None else load_calibration_coefficients()
        target = float(np.clip(target_gm2, 10.0, 300.0))

        plans = self._collect_plans(gas_type, strip_width_mm, target, bath_temp_c, gap_options, cal, strict=True)
        relaxed = False
        if not plans:
            plans = self._collect_plans(
                gas_type, strip_width_mm, target, bath_temp_c, gap_options, cal, strict=False
            )
            relaxed = True

        if not plans:
            return {
                "status": "infeasible",
                "target_gm2": round(target, 1),
                "relaxed": False,
                "plans": [],
                "message": (
                    f"目標 {target:.1f} g/m² を標準的な操業範囲（通板速度 {SPEED_MIN_MPM:.0f}〜{SPEED_MAX_MPM:.0f} m/min、"
                    f"ノズルすき間 {min(gap_options):.1f}〜{max(gap_options):.1f} mm）で達成できる条件が見つかりませんでした。"
                    "目標値を見直すか、「条件設計」タブで固定条件を直接指定してください。"
                ),
            }

        recommended = plans[0]
        message = (
            f"目標 {target:.1f} g/m² を安全に出せる条件を {len(plans)} 通り見つけました。"
            "そのうち生産性（通板速度）とスプラッシュ余裕・圧力余裕のバランスが最も良い案を推奨します。"
        )
        if relaxed:
            message += "（標準基準では解が無いため、判定基準を緩めた結果です。余裕が小さいので実績校正を強く推奨します。）"
        return {
            "status": "ok",
            "target_gm2": round(target, 1),
            "relaxed": relaxed,
            "gas_type": normalize_gas_type(str(gas_type)),
            "strip_width_mm": round(float(strip_width_mm), 0),
            "bath_temp_c": round(float(bath_temp_c), 1),
            "plans": plans,
            "recommended": recommended,
            "message": message,
        }

    def _collect_plans(
        self,
        gas_type: str,
        strip_width_mm: float,
        target: float,
        bath_temp_c: float,
        gap_options: tuple[float, ...],
        cal: dict,
        strict: bool,
    ) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        for gap in gap_options:
            base = self._normalize_base(
                {
                    "gas_type": gas_type,
                    "strip_width_mm": strip_width_mm,
                    "bath_temp_c": bath_temp_c,
                    "nozzle_gap_mm": gap,
                    "line_speed_mpm": 120.0,
                },
                target,
            )
            speed = self._max_safe_speed(base, target, cal, strict=strict)
            if speed is None:
                continue
            base["line_speed_mpm"] = speed
            candidates = self._candidates(base, target, cal)
            safe = [c for c in candidates if self._is_safe(c)]
            if not safe:
                continue
            if strict:
                # 速度探索と同じ厳しい基準を満たす候補を優先する。
                preferred = [
                    c
                    for c in safe
                    if c["splash_level"] == "低"
                    and c["model_confidence"] != "低"
                    and QUICK_PRESSURE_BAND_KPA[0] <= c["plenum_pressure_kpa"] <= QUICK_PRESSURE_BAND_KPA[1]
                ]
                safe = preferred or safe
            best = safe[0]
            best["rank"] = 1
            plans.append(
                {
                    "nozzle_gap_mm": round(gap, 2),
                    "line_speed_mpm": round(speed, 0),
                    "bath_temp_c": round(float(bath_temp_c), 1),
                    "condition": best,
                    "alternatives": safe[1:3],
                    "plan_score": self._plan_score(speed, best),
                }
            )
        plans.sort(key=lambda p: p["plan_score"])
        return plans

    @staticmethod
    def _plan_score(speed: float, candidate: dict) -> float:
        """生産性重視で順位付け（小さいほど良い）。"""
        score = -speed / 100.0
        score += 2.0 * max(0.0, candidate["splash_score"] - 0.4)
        score += _CONFIDENCE_PENALTY.get(candidate["model_confidence"], 1.0) * 0.5
        pressure = candidate["plenum_pressure_kpa"]
        p_low, p_high = PREFERRED_PRESSURE_KPA
        if pressure < p_low:
            score += (p_low - pressure) / p_low
        elif pressure > p_high:
            score += (pressure - p_high) / p_high
        return float(score)

    def _max_safe_speed(self, base: dict, target_gm2: float, cal: dict, strict: bool = True) -> float | None:
        """目標を安全に達成できる最大通板速度を二分探索で求める（5 m/min 単位）。

        strict=True: スプラッシュ「低」・モデル適用域内・圧力に上下の余裕がある条件のみ可とする。
        strict=False: スプラッシュ発生域とチョークだけを除外する（最後の手段）。
        """

        def acceptable(candidate: dict) -> bool:
            if not strict:
                return self._is_comfortable(candidate)
            pressure = candidate["plenum_pressure_kpa"]
            return (
                candidate["splash_level"] == "低"
                and candidate["model_confidence"] != "低"
                and not candidate["choked"]
                and QUICK_PRESSURE_BAND_KPA[0] <= pressure <= QUICK_PRESSURE_BAND_KPA[1]
            )

        def works(speed: float) -> bool:
            trial = dict(base)
            trial["line_speed_mpm"] = float(speed)
            if not self._range_covers(trial, target_gm2, cal):
                return False
            candidates = self._candidates(
                trial, target_gm2, cal, distances=(6.0, 10.0, 16.0), with_sensitivity=False
            )
            return any(acceptable(c) for c in candidates)

        if works(SPEED_MAX_MPM):
            return SPEED_MAX_MPM
        if not works(SPEED_MIN_MPM):
            return None
        low, high = SPEED_MIN_MPM, SPEED_MAX_MPM
        for _ in range(6):
            mid = 0.5 * (low + high)
            if works(mid):
                low = mid
            else:
                high = mid
        return float(np.floor(low / SPEED_FINE_STEP) * SPEED_FINE_STEP)

    # ==============================================================
    # 公開API 4: 感度カーブ（UI表示用）
    # ==============================================================
    def response_curves(
        self,
        condition: dict,
        points: int = 25,
        calibration: dict | None = None,
    ) -> dict[str, Any]:
        """噴射圧力・ノズル距離・通板速度それぞれに対する片面めっき付着量の応答曲線。"""
        cal = calibration if calibration is not None else load_calibration_coefficients()
        base = dict(condition)
        base.setdefault("project_name", "ResponseCurve")

        pressure = float(base["plenum_pressure_kpa"])
        distance = float(base["nozzle_strip_distance_mm"])
        speed = float(base["line_speed_mpm"])

        def sweep(key: str, values: np.ndarray) -> list[float]:
            output: list[float] = []
            for value in values:
                cond = dict(base)
                cond[key] = float(value)
                result = self.analysis.analyze(cond, calibration=cal, include_profile=False)
                output.append(round(float(result["cw_one_side_gm2"]), 2))
            return output

        p_values = np.linspace(PRESSURE_MIN_KPA, PRESSURE_MAX_KPA, points)
        z_values = np.linspace(DISTANCE_MIN_MM, DISTANCE_MAX_MM, points)
        v_values = np.linspace(SPEED_MIN_MPM, SPEED_MAX_MPM, points)

        return {
            "pressure": {"x": [round(v, 2) for v in p_values.tolist()], "y": sweep("plenum_pressure_kpa", p_values), "current": round(pressure, 2)},
            "distance": {"x": [round(v, 2) for v in z_values.tolist()], "y": sweep("nozzle_strip_distance_mm", z_values), "current": round(distance, 2)},
            "speed": {"x": [round(v, 1) for v in v_values.tolist()], "y": sweep("line_speed_mpm", v_values), "current": round(speed, 1)},
        }

    def coating_map(
        self,
        condition: dict,
        points: int = 15,
        calibration: dict | None = None,
    ) -> dict[str, Any]:
        """噴射圧力 × ノズル距離 の片面めっき付着量マップ（等高線表示用）。"""
        cal = calibration if calibration is not None else load_calibration_coefficients()
        base = dict(condition)
        base.setdefault("project_name", "CoatingMap")

        x_values = np.linspace(PRESSURE_MIN_KPA, PRESSURE_MAX_KPA, points)
        y_values = np.linspace(DISTANCE_MIN_MM, DISTANCE_MAX_MM, points)
        matrix: list[list[float]] = []
        splash: list[list[float]] = []
        for y_value in y_values:
            row: list[float] = []
            splash_row: list[float] = []
            for x_value in x_values:
                cond = dict(base)
                cond["plenum_pressure_kpa"] = float(x_value)
                cond["nozzle_strip_distance_mm"] = float(y_value)
                result = self.analysis.analyze(cond, calibration=cal, include_profile=False)
                row.append(round(float(result["cw_one_side_gm2"]), 2))
                splash_row.append(round(float(result["splash_score"]), 3))
            matrix.append(row)
            splash.append(splash_row)
        return {
            "x": [round(v, 2) for v in x_values.tolist()],
            "y": [round(v, 2) for v in y_values.tolist()],
            "z": matrix,
            "splash": splash,
        }
