"""実機モデルによる条件設計（逆算）。

物理モデル側の design_service と役割は同じだが、こちらは実機データ由来の
CatBoost モデルを前向きに評価して数値的に反転する。

反転にニュートン法や二分法を使わない理由:
    実測では GL の操業レンジが極端に狭く（圧力 10.3〜10.9 kPa など）、
    モデル応答は必ずしも単調ではない。区間全体を走査して交差点を全部拾い、
    「いまの条件からの変更量が最小」の解を選ぶ。

変更提案の優先順位（ユーザー承認済み）:
    1. Yノズル位置   … 実績で最も効くレバー（記号内相関 GI +0.42 / GL +0.35）
    2. 通板速度      … 効きは弱い（速度が上がると持ち出しが増え目付は増加）
    3. ノズル圧力    … 実績ではほぼ固定。動かすと流量も連動させる必要がある
提案しない項目: 製品板厚・製品板幅・付着量記号・ライン（製品仕様/設備仕様のため）
"""
from __future__ import annotations

import math
from typing import Any, Callable

from gaswipelab.ml.ood import CAUTION, IN_RANGE, OUT_OF_DISTRIBUTION, RangeChecker
from gaswipelab.ml.predictor import GasWipingPredictor

#: 目標に到達したとみなす許容差 [g/m²]。両面合計に対する値。
SOLVE_TOLERANCE_GM2 = 1.0

#: 走査の粗さ。レバー範囲をこの数に分割して交差を探す。
SCAN_POINTS = 41

#: これ未満の実績件数しかない付着量記号は、予測も操業範囲も信頼できない。
SMALL_SAMPLE_LIMIT = 20

#: 局所精密化の反復回数（交差区間を二分する）。
REFINE_STEPS = 24

FRONT_POSITION = "表ノズル位置平均_mm"
BACK_POSITION = "裏ノズル位置平均_mm"
FRONT_PRESSURE = "表ノズル圧力_kPa"
BACK_PRESSURE = "裏ノズル圧力_kPa"
SPEED = "中央速度_m_min"
FLOW = "ノズル吹込流量_Nm3H"

FIXED_NOTE = (
    "製品板厚・製品板幅・付着量記号・ラインは製品仕様／設備仕様のため、変更提案の対象にしていません。"
)
INTERPOLATION_NOTE = (
    "この結果は過去実績の範囲内での統計的補間です。設備条件を独立に動かしたときの"
    "因果効果を保証するものではありません。"
)


class Lever:
    """逆算で動かす操作量。表裏はまとめて動かす（実績で強く連動しているため）。"""

    def __init__(self, key: str, label: str, unit: str, digits: int,
                 driver: str, apply: Callable[[dict, float], None]) -> None:
        self.key = key
        self.label = label
        self.unit = unit
        self.digits = digits
        self.driver = driver  # 範囲判定に使う代表列
        self.apply = apply


def _apply_position(condition: dict[str, Any], value: float) -> None:
    # 表裏の差は保ったまま、表を基準に平行移動する。
    offset = float(condition.get(BACK_POSITION, value)) - float(condition.get(FRONT_POSITION, value))
    condition[FRONT_POSITION] = value
    condition[BACK_POSITION] = value + offset


def _apply_speed(condition: dict[str, Any], value: float) -> None:
    condition[SPEED] = value


def _apply_pressure(condition: dict[str, Any], value: float) -> None:
    offset = float(condition.get(BACK_PRESSURE, value)) - float(condition.get(FRONT_PRESSURE, value))
    condition[FRONT_PRESSURE] = value
    condition[BACK_PRESSURE] = value + offset


LEVERS: tuple[Lever, ...] = (
    Lever("nozzle_position", "Yノズル位置", "mm", 2, FRONT_POSITION, _apply_position),
    Lever("line_speed", "通板速度", "m/min", 1, SPEED, _apply_speed),
    Lever("nozzle_pressure", "ノズル圧力", "kPa", 2, FRONT_PRESSURE, _apply_pressure),
)


class MachineDesignService:
    def __init__(self, predictor: GasWipingPredictor, checker: RangeChecker) -> None:
        self.predictor = predictor
        self.checker = checker

    # ------------------------------------------------------------------
    def _sync_flow(self, condition: dict[str, Any], line: str) -> None:
        """圧力を動かしたら、実績の圧力-流量関係にそって流量も追従させる。"""
        try:
            mean_pressure = (float(condition[FRONT_PRESSURE]) + float(condition[BACK_PRESSURE])) / 2.0
            speed = float(condition[SPEED])
        except (KeyError, TypeError, ValueError):
            return
        expected = self.checker.expected_flow(line, mean_pressure, speed)
        if expected is not None:
            condition[FLOW] = round(expected, 1)

    def _ch(self, condition: dict[str, Any]) -> float:
        return self.predictor.predict_ch(condition)

    def _evaluate(self, condition: dict[str, Any]) -> dict[str, Any]:
        result = self.predictor.predict(condition)
        judgement = self.checker.evaluate(
            result["line"], result["coating_code"], condition,
            consistency_gap=result["CH_direct_minus_sum_g_m2"],
        )
        result["range_status"] = judgement["status"]
        result["range_fields"] = judgement["fields"]
        result["warnings"] = judgement["warnings"]
        result["note"] = INTERPOLATION_NOTE
        return result

    # ------------------------------------------------------------------
    def _solve_lever(self, base: dict[str, Any], lever: Lever, target: float,
                     line: str, code: str) -> dict[str, Any] | None:
        """1本のレバーだけを動かして目標に到達する値を探す。

        範囲全体を走査して目標を跨ぐ区間をすべて拾い、現在値から最も近い解を返す。
        """
        bounds = self.checker.bounds(line, code, lever.driver)
        if not bounds:
            return None
        low, high = bounds
        if not (high > low):
            return None
        current = float(base.get(lever.driver, low))

        def ch_at(value: float) -> float:
            trial = dict(base)
            lever.apply(trial, value)
            if lever.key == "nozzle_pressure":
                self._sync_flow(trial, line)
            return self._ch(trial)

        step = (high - low) / (SCAN_POINTS - 1)
        xs = [low + step * i for i in range(SCAN_POINTS)]
        ys = [ch_at(x) for x in xs]

        brackets: list[tuple[float, float]] = []
        for i in range(len(xs) - 1):
            if (ys[i] - target) == 0.0:
                brackets.append((xs[i], xs[i]))
            elif (ys[i] - target) * (ys[i + 1] - target) < 0.0:
                brackets.append((xs[i], xs[i + 1]))
        if not brackets:
            return None

        # 現在値からの移動量が最小になる区間を選ぶ
        brackets.sort(key=lambda b: min(abs(b[0] - current), abs(b[1] - current)))
        lo, hi = brackets[0]
        for _ in range(REFINE_STEPS):
            if hi - lo < 1.0e-6:
                break
            mid = (lo + hi) / 2.0
            if (ch_at(lo) - target) * (ch_at(mid) - target) <= 0.0:
                hi = mid
            else:
                lo = mid
        value = round((lo + hi) / 2.0, lever.digits)

        candidate = dict(base)
        lever.apply(candidate, value)
        if lever.key == "nozzle_pressure":
            self._sync_flow(candidate, line)
        result = self._evaluate(candidate)
        if result["range_status"] == OUT_OF_DISTRIBUTION:
            return None
        return {
            "lever": lever.key,
            "label": lever.label,
            "unit": lever.unit,
            "current": round(current, lever.digits),
            "suggested": value,
            "delta": round(value - current, lever.digits),
            "condition": candidate,
            "result": result,
            "error_gm2": round(result["CH_sum_pred_g_m2"] - target, 2),
        }

    def _best_effort_lever(self, base: dict[str, Any], lever: Lever, target: float,
                           line: str, code: str) -> dict[str, Any] | None:
        """目標をちょうど跨げないとき、いちばん近づける値を返す。

        「届きません」で終わらせず、あと一歩の条件を示すために使う。
        """
        bounds = self.checker.bounds(line, code, lever.driver)
        if not bounds:
            return None
        low, high = bounds
        if not (high > low):
            return None
        current = float(base.get(lever.driver, low))
        best = None
        for i in range(SCAN_POINTS):
            value = low + (high - low) * i / (SCAN_POINTS - 1)
            trial = dict(base)
            lever.apply(trial, value)
            if lever.key == "nozzle_pressure":
                self._sync_flow(trial, line)
            error = abs(self._ch(trial) - target)
            if best is None or error < best[0]:
                best = (error, round(value, lever.digits), trial)
        if best is None:
            return None
        error, value, candidate = best
        result = self._evaluate(candidate)
        if result["range_status"] == OUT_OF_DISTRIBUTION:
            return None
        return {
            "lever": lever.key, "label": lever.label, "unit": lever.unit,
            "current": round(current, lever.digits), "suggested": value,
            "delta": round(value - current, lever.digits),
            "condition": candidate, "result": result,
            "error_gm2": round(result["CH_sum_pred_g_m2"] - target, 2),
            "approximate": True,
        }

    def achievable_range(self, base: dict[str, Any], line: str, code: str) -> dict[str, float]:
        """Yノズル位置を実績範囲いっぱいに振ったときに届くCHの範囲。"""
        bounds = self.checker.bounds(line, code, FRONT_POSITION)
        if not bounds:
            return {}
        low, high = bounds
        values = []
        for i in range(SCAN_POINTS):
            trial = dict(base)
            _apply_position(trial, low + (high - low) * i / (SCAN_POINTS - 1))
            values.append(self._ch(trial))
        return {"min": round(min(values), 1), "max": round(max(values), 1),
                "position_min": round(low, 2), "position_max": round(high, 2)}

    # ------------------------------------------------------------------
    def design(self, condition: dict[str, Any], target_ch_gm2: float) -> dict[str, Any]:
        line = str(condition.get("line", "")).strip()
        code = str(condition.get("coating_code", "")).strip()
        base = dict(condition)
        current = self._evaluate(base)
        current_ch = current["CH_sum_pred_g_m2"]

        payload: dict[str, Any] = {
            "line": line,
            "coating_code": code,
            "target_ch_gm2": target_ch_gm2,
            "current": current,
            "current_condition": base,
            "current_ch_gm2": round(current_ch, 1),
            "fixed_note": FIXED_NOTE,
            "note": INTERPOLATION_NOTE,
            "proposals": [],
        }

        stats = self.checker.code_stats(line, code) or {}
        sample = int(stats.get("n", 0))
        if 0 < sample < SMALL_SAMPLE_LIMIT:
            payload["sample_warning"] = (
                f"{line}の付着量記号 {code} は実績が {sample} 件しかありません。"
                "予測値も操業範囲も信頼度が低いため、参考程度に扱ってください。"
            )

        if abs(current_ch - target_ch_gm2) <= SOLVE_TOLERANCE_GM2:
            payload["status"] = "ok"
            payload["message"] = (
                f"いまの条件のままで目標 {target_ch_gm2:.1f} g/m² に届いています"
                f"（予測 {current_ch:.1f} g/m²・両面合計）。"
            )
            return payload

        for lever in LEVERS:
            found = self._solve_lever(base, lever, target_ch_gm2, line, code)
            if found is None:
                payload["proposals"].append({
                    "lever": lever.key, "label": lever.label, "found": False,
                    "reason": f"{lever.label}の実績操業範囲内では目標に届きませんでした。",
                })
                continue
            found["found"] = True
            payload["proposals"].append(found)
            payload["status"] = "ok"
            direction = "上げて" if found["delta"] > 0 else "下げて"
            payload["message"] = (
                f"{lever.label}を {found['current']}{lever.unit} から "
                f"{found['suggested']}{lever.unit} へ{direction}ください"
                f"（{found['delta']:+.{lever.digits}f}{lever.unit}）。"
                f"予測 {found['result']['CH_sum_pred_g_m2']:.1f} g/m²（両面合計）。"
            )
            return payload

        # どのレバーでもちょうど到達できないとき、いちばん近づく案を出す。
        attempts = [self._best_effort_lever(base, lever, target_ch_gm2, line, code) for lever in LEVERS]
        attempts = [a for a in attempts if a is not None]
        current_error = abs(current_ch - target_ch_gm2)
        if attempts:
            best = min(attempts, key=lambda a: abs(a["error_gm2"]))
            # 現状より明確に近づく場合だけ提案する
            if abs(best["error_gm2"]) < current_error * 0.8:
                payload["status"] = "approximate"
                payload["proposals"].append(dict(best, found=True))
                payload["message"] = (
                    f"目標 {target_ch_gm2:.1f} g/m² ちょうどには届きませんが、"
                    f"{best['label']}を {best['current']}{best['unit']} から {best['suggested']}{best['unit']} へ"
                    f"変えると最も近づきます（予測 {best['result']['CH_sum_pred_g_m2']:.1f} g/m²・"
                    f"目標との差 {best['error_gm2']:+.1f} g/m²）。"
                )
                payload["achievable"] = self.achievable_range(base, line, code)
                return payload

        payload["status"] = "infeasible"
        reach = self.achievable_range(base, line, code)
        payload["achievable"] = reach
        if reach:
            payload["message"] = (
                f"実績の操業範囲内では目標 {target_ch_gm2:.1f} g/m² に届きませんでした。"
                f"この条件でYノズル位置を {reach['position_min']}〜{reach['position_max']} mm に"
                f"振って届く範囲は {reach['min']}〜{reach['max']} g/m²（両面合計）です。"
                "目標値、または製品条件・付着量記号の選択を見直してください。"
            )
        else:
            payload["message"] = "実績の操業範囲内では目標に届きませんでした。"
        return payload

    # ------------------------------------------------------------------
    def predict(self, condition: dict[str, Any]) -> dict[str, Any]:
        return self._evaluate(condition)

    def compare(self, base: dict[str, Any], changed: dict[str, Any]) -> dict[str, Any]:
        left = self._evaluate(base)
        right = self._evaluate(changed)
        diffs = []
        for key in sorted(set(base) | set(changed)):
            if key in ("line", "coating_code"):
                continue
            a, b = base.get(key), changed.get(key)
            try:
                fa, fb = float(a), float(b)
            except (TypeError, ValueError):
                if str(a) != str(b):
                    diffs.append({"field": key, "before": a, "after": b, "delta": None})
                continue
            if abs(fa - fb) > 1.0e-9:
                diffs.append({"field": key, "before": fa, "after": fb, "delta": round(fb - fa, 4)})
        return {
            "before": left,
            "after": right,
            "diffs": diffs,
            "delta_ch_gm2": round(right["CH_sum_pred_g_m2"] - left["CH_sum_pred_g_m2"], 2),
            "delta_cf_gm2": round(right["CF_pred_g_m2"] - left["CF_pred_g_m2"], 2),
            "delta_cg_gm2": round(right["CG_pred_g_m2"] - left["CG_pred_g_m2"], 2),
            "note": INTERPOLATION_NOTE,
        }

    def response_curve(self, condition: dict[str, Any], lever_key: str,
                       points: int = 25) -> dict[str, Any]:
        """1本のレバーを実績範囲で振ったときのCH応答。グラフ用。"""
        lever = next((x for x in LEVERS if x.key == lever_key), None)
        if lever is None:
            raise ValueError(f"未知のレバー: {lever_key}")
        line = str(condition.get("line", "")).strip()
        code = str(condition.get("coating_code", "")).strip()
        bounds = self.checker.bounds(line, code, lever.driver)
        if not bounds:
            return {"x": [], "y": []}
        low, high = bounds
        xs, ys = [], []
        for i in range(points):
            value = low + (high - low) * i / (points - 1)
            trial = dict(condition)
            lever.apply(trial, value)
            if lever.key == "nozzle_pressure":
                self._sync_flow(trial, line)
            xs.append(round(value, 3))
            ys.append(round(self._ch(trial), 2))
        stat = self.checker.bounds(line, code, lever.driver, low="p10", high="p90")
        return {
            "x": xs, "y": ys,
            "label": lever.label, "unit": lever.unit,
            "current": condition.get(lever.driver),
            "band": {"low": stat[0], "high": stat[1]} if stat else None,
        }
