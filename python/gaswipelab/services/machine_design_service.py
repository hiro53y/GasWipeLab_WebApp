"""実績モデルによる条件設計（逆算）。

物理モデル側の design_service と役割は同じだが、こちらは実機データ由来の
CatBoost モデルを前向きに評価して数値的に反転する。

反転にニュートン法や二分法を使わない理由:
    実測では GL の操業レンジが極端に狭く（圧力 10.3〜10.9 kPa など）、
    モデル応答は必ずしも単調ではない。区間全体を走査して交差点を全部拾い、
    「いまの条件からの変更量が最小」の解を選ぶ。

変更提案の優先順位は **操業方針であって、モデルが決めたものではない**:
    1. Yノズル位置   … 単独で動かしやすく、実績のばらつきも最もよく説明する
    2. 通板速度      … 生産計画に影響するため位置の次
    3. ノズル圧力    … 実績ではほぼ固定運用。動かすと流量も連動するので最後
モデルが言えるのは「実績のばらつきを説明する寄与の大きさ」までで、
どれから動かすべきかは操業側の取り決めである（LEVER_ORDER_BASIS を参照）。

提案しない項目: 製品板厚・製品板幅・付着量記号・ライン（製品仕様/設備仕様のため）
"""
from __future__ import annotations

import math
from typing import Any, Callable

from gaswipelab.ml.ood import CAUTION, IN_RANGE, OUT_OF_DISTRIBUTION, RangeChecker
from gaswipelab.ml.predictor import (
    CODE_CHANGE_ERROR_FACTOR,
    ERROR_REFERENCE,
    MODEL_SKILL,
    GasWipingPredictor,
)

#: 直前のコイルから付着量記号が変わったかどうかを受け取る条件キー。
#: モデルの入力特徴量ではない（誤差の見積もりにだけ使う）。
CODE_CHANGED_KEY = "直前コイルから記号変更"

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

#: 提案順の根拠。モデルの出力ではなく操業側の取り決めであることを画面に明示するために持つ。
LEVER_ORDER_BASIS = {
    "source": "operational_policy",
    "text": (
        "変更を試す順番（Yノズル位置 → 通板速度 → ノズル圧力）は操業上の取り決めです。"
        "モデルが決めた順位ではありません。"
        "実績データから言えるのは各項目が実績のばらつきをどれだけ説明するかまでで、"
        "圧力・流量・Yノズル位置は実機で連動して動かされてきたため、"
        "1項目だけ動かしたときの効果を切り分けることはできません。"
    ),
}


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

#: レバーが動かす列（効果の大きさを合算して評価するため）
LEVER_FEATURES: dict[str, tuple[str, ...]] = {
    "nozzle_position": (FRONT_POSITION, BACK_POSITION),
    "line_speed": (SPEED,),
    "nozzle_pressure": (FRONT_PRESSURE, BACK_PRESSURE),
}

#: 実績のばらつき全体を使ってもこれ未満しか動かない項目は、変更提案に使わない。
#: 例: GLのノズル圧力は実績が 9.9〜14.4 kPa と狭く、効果を実績から特定できない。
MIN_LEVER_EFFECT_GM2 = 2.0


class MachineDesignService:
    def __init__(self, predictor: GasWipingPredictor, checker: RangeChecker) -> None:
        self.predictor = predictor
        self.checker = checker
        self._range_cache: dict[str, dict[str, tuple[float, float]]] = {}
        self._sign_cache: dict[str, dict[str, float]] = {}

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

    def _ranges(self, line: str) -> dict[str, tuple[float, float]]:
        """外挿の基準になる学習範囲。ライン単位でキャッシュする。"""
        cached = self._range_cache.get(line)
        if cached is None:
            cached = self.checker.model_ranges(line)
            self._range_cache[line] = cached
        return cached

    def _signs(self, line: str) -> dict[str, float]:
        """実データ上の効果の向き。外挿を実データと逆向きに延ばさないために使う。"""
        cached = self._sign_cache.get(line)
        if cached is None:
            cached = {name: self.checker.trend_sign(line, name)
                      for name in self.checker.trends(line)}
            self._sign_cache[line] = cached
        return cached

    def lever_effect(self, line: str, lever: "Lever") -> float:
        """そのレバーを実績のばらつき全体で動かしたときのめっき付着量の変化量 [g/m²]。

        表裏をまとめて動かすレバーは、両方の効果を合算して評価する。
        """
        trends = self.checker.trends(line)
        total = 0.0
        for feature in LEVER_FEATURES.get(lever.key, (lever.driver,)):
            entry = trends.get(feature)
            if entry:
                total += abs(float(entry.get("swing_gm2", 0.0)))
        return total

    def _ch(self, condition: dict[str, Any]) -> float:
        line = str(condition.get("line", "")).strip()
        return self.predictor.predict_ch(condition, self._ranges(line), self._signs(line))

    def _evaluate(self, condition: dict[str, Any]) -> dict[str, Any]:
        line = str(condition.get("line", "")).strip()
        result = self.predictor.predict(condition, self._ranges(line), self._signs(line))
        judgement = self.checker.evaluate(
            result["line"], result["coating_code"], condition,
            consistency_gap=result["CH_direct_minus_sum_g_m2"],
        )
        result["range_status"] = judgement["status"]
        result["range_fields"] = judgement["fields"]
        result["warnings"] = judgement["warnings"]
        result["combination"] = judgement.get("combination")
        result["note"] = INTERPOLATION_NOTE
        return result

    # ------------------------------------------------------------------
    def _solve_lever(self, base: dict[str, Any], lever: Lever, target: float,
                     line: str, code: str, wide: bool = False) -> dict[str, Any] | None:
        """1本のレバーだけを動かして目標に到達する値を探す。

        範囲全体を走査して目標を跨ぐ区間をすべて拾い、現在値から最も近い解を返す。

        wide=False : 実績の操業範囲内だけを探す（既定。ここで見つかれば最も確か）
        wide=True  : 設備仕様の範囲まで広げて探す。実績外の解も返すが範囲外として扱う
        """
        bounds = (self.checker.ui_bounds(line, lever.driver) if wide
                  else self.checker.bounds(line, code, lever.driver))
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
        # 実績範囲内での探索では、範囲外に出る解は採用しない。
        # 広域探索では範囲外も返すが、範囲外である旨を必ず添える。
        if result["range_status"] == OUT_OF_DISTRIBUTION and not wide:
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
            "out_of_range": result["range_status"] == OUT_OF_DISTRIBUTION,
            "extrapolated": bool(result.get("extrapolated_features")),
        }

    def _best_effort_lever(self, base: dict[str, Any], lever: Lever, target: float,
                           line: str, code: str, wide: bool = False) -> dict[str, Any] | None:
        """目標をちょうど跨げないとき、いちばん近づける値を返す。

        「届きません」で終わらせず、あと一歩の条件を示すために使う。
        wide=True では設備仕様の範囲まで広げて探す。
        """
        bounds = (self.checker.ui_bounds(line, lever.driver) if wide
                  else self.checker.bounds(line, code, lever.driver))
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
        if result["range_status"] == OUT_OF_DISTRIBUTION and not wide:
            return None
        return {
            "lever": lever.key, "label": lever.label, "unit": lever.unit,
            "current": round(current, lever.digits), "suggested": value,
            "delta": round(value - current, lever.digits),
            "condition": candidate, "result": result,
            "error_gm2": round(result["CH_sum_pred_g_m2"] - target, 2),
            "approximate": True,
            "out_of_range": result["range_status"] == OUT_OF_DISTRIBUTION,
            "extrapolated": bool(result.get("extrapolated_features")),
        }

    # ------------------------------------------------------------------
    def _resolution(self, line: str, difference: float,
                    code_changed: bool = False) -> dict[str, Any]:
        """目標との差が、このモデルで判別できる大きさかどうかを述べる。

        MAE は誤差の平均であって、個々の予測の信頼区間ではない。
        「MAE より小さいから変更不要」という言い方はできないので、
        ここでは判別可能性だけを述べ、操業判断は利用者に委ねる。

        code_changed=True（直前のコイルから付着量記号が変わった）のときは、
        実測の誤差倍率をかけて判定を甘くする。切替直後は実際に外れやすいため。
        """
        reference = ERROR_REFERENCE.get(line, {})
        factor = CODE_CHANGE_ERROR_FACTOR.get(line, 1.0) if code_changed else 1.0
        mae = float(reference.get("CH_MAE", SOLVE_TOLERANCE_GM2)) * factor
        p90 = float(reference.get("CH_P90", mae * 2.0)) * factor
        worst = float(reference.get("CH_MAE_worst", mae)) * factor
        if difference <= mae:
            level = "below_mae"
            statement = (
                f"目標との差 {difference:.1f} g/m² は、このモデルの平均絶対誤差 {mae:.1f} g/m²"
                f"（月別の最悪月で {worst:.1f}）より小さい差です。"
                "モデルはこの大きさの差を判別できません。"
                "実機で差が出るかどうかは、このツールでは判断できません。"
            )
        elif difference <= p90:
            level = "below_p90"
            statement = (
                f"目標との差 {difference:.1f} g/m² は平均絶対誤差 {mae:.1f} g/m² より大きいものの、"
                f"予測の10件に1件は {p90:.1f} g/m² 以上外れます。"
                "差の向きは参考になりますが、値そのものは幅を持って読んでください。"
            )
        else:
            level = "above_p90"
            statement = (
                f"目標との差 {difference:.1f} g/m² は、このモデルの誤差の大きさ"
                f"（平均 {mae:.1f} / 90%点 {p90:.1f} g/m²）を上回ります。"
                "条件を見直す根拠になる差です。"
            )
        if code_changed:
            statement += (
                f"なお直前のコイルから付着量記号が変わるため、誤差の見積もりを実測にもとづき"
                f"{factor:.2f}倍にしています（切替直後は実際に外れやすいことを確認しています）。"
            )
        return {
            "gap_gm2": round(difference, 2),
            "mae_gm2": round(mae, 2),
            "mae_worst_gm2": round(worst, 2),
            "p90_gm2": round(p90, 2),
            "level": level,
            "statement": statement,
            "basis": reference.get("basis", "rolling"),
            "code_changed": bool(code_changed),
            "error_factor": round(factor, 2),
        }

    @staticmethod
    def _add_out_of_range_caution(payload: dict[str, Any], proposal: dict[str, Any]) -> None:
        """実績の外へ出る提案に、操業側の注意を添える。

        これは統計的な主張ではない。「モデルが判別しきれない差を埋めるために、
        根拠のない領域へ出るのは避ける」という運用上の判断であることを明示する。
        """
        if not proposal.get("out_of_range"):
            return
        level = payload.get("resolution", {}).get("level")
        if level == "above_p90":
            return
        resolution = payload.get("resolution", {})
        gap = resolution.get("gap_gm2")
        mae = resolution.get("mae_gm2")
        p90 = resolution.get("p90_gm2")
        if level == "below_mae":
            head = (f"目標との差 {gap} g/m² は、このモデルの平均絶対誤差 {mae} g/m² より小さく、"
                    "モデルはこの差を判別できません。")
        else:
            head = (f"目標との差 {gap} g/m² は平均絶対誤差 {mae} g/m² を超えますが、"
                    f"予測の10件に1件は ±{p90} g/m² 以上外れる水準です。")
        payload["operational_caution"] = (
            head
            + f"その差を埋めるために{proposal['label']}を実績にない値まで動かすことになります。"
            "統計的にこの変更が不要と言えるわけではありませんが、"
            "確かめようのない差のために実績外へ出る操作は勧めません"
            "（モデルの主張ではなく、運用側の判断です）。"
        )

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

        # 目標との差を、このモデルの誤差分布と突き合わせる。
        #
        # MAE は個々の予測の信頼区間ではない。「差が MAE より小さいから変えなくてよい」
        # とは言えないので、ここでは「モデルがこの差を判別できるかどうか」だけを述べ、
        # 変える／変えないの判断は利用者に残す。
        difference = abs(current_ch - target_ch_gm2)
        payload["resolution"] = self._resolution(line, difference, bool(condition.get(CODE_CHANGED_KEY)))
        payload["model_skill"] = MODEL_SKILL.get(line)
        payload["model_mae_gm2"] = payload["resolution"]["mae_gm2"]

        if difference <= SOLVE_TOLERANCE_GM2:
            # モデルの出力そのものが目標に一致している。ここは統計的主張ではなく計算結果。
            payload["status"] = "ok"
            payload["message"] = (
                f"いまの条件のままで、モデルの予測は目標 {target_ch_gm2:.1f} g/m² と"
                f"ほぼ同じです（予測 {current_ch:.1f} g/m²・両面合計）。"
            )
            return payload

        # 第1段: 実績の操業範囲内で、効き目の大きい順に1項目ずつ試す。
        # 実績からめっき付着量への効果を特定できない項目は、そもそも提案に使わない。
        usable: list[Lever] = []
        for lever in LEVERS:
            effect = self.lever_effect(line, lever)
            if effect < MIN_LEVER_EFFECT_GM2:
                payload["proposals"].append({
                    "lever": lever.key, "label": lever.label, "found": False,
                    "unusable": True,
                    "reason": (
                        f"{line}の実績では{lever.label}がほとんど動いておらず、"
                        f"めっき付着量のばらつきをこの項目で説明できません"
                        f"（実績の振れ幅ぜんぶで {effect:.1f} g/m² 相当）。"
                        "実機で効かないという意味ではなく、実績データからは判断できないという意味です。"
                        "このため変更提案には使いません。"
                    ),
                })
                continue
            usable.append(lever)
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

        # 第2段: 設備仕様の範囲まで広げて探す。実績外なので必ず警告を添える。
        for lever in usable:
            found = self._solve_lever(base, lever, target_ch_gm2, line, code, wide=True)
            if found is None:
                continue
            found["found"] = True
            payload["proposals"].append(found)
            payload["status"] = "out_of_range"
            direction = "上げて" if found["delta"] > 0 else "下げて"
            payload["message"] = (
                f"実績の操業範囲内では届かないため、範囲を広げて求めました。"
                f"{lever.label}を {found['current']}{lever.unit} から "
                f"{found['suggested']}{lever.unit} へ{direction}ください"
                f"（{found['delta']:+.{lever.digits}f}{lever.unit}）。"
                f"予測 {found['result']['CH_sum_pred_g_m2']:.1f} g/m²（両面合計）。"
                "実績にない領域のため、境界の傾きを延長した参考値です。"
            )
            self._add_out_of_range_caution(payload, found)
            return payload

        # どのレバーでもちょうど到達できないとき、いちばん近づく案を出す。
        # まず実績範囲内で探し、まだ遠いときは設備仕様の範囲まで広げて探し直して、
        # 目標に近いほうを採用する。
        attempts = [self._best_effort_lever(base, lever, target_ch_gm2, line, code)
                    for lever in usable]
        attempts = [a for a in attempts if a is not None]
        closest = min((abs(a["error_gm2"]) for a in attempts), default=None)
        if closest is None or closest > SOLVE_TOLERANCE_GM2:
            wider = [self._best_effort_lever(base, lever, target_ch_gm2, line, code, wide=True)
                     for lever in usable]
            attempts += [a for a in wider if a is not None]
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
                self._add_out_of_range_caution(payload, best)
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
        # スライダーで選べる範囲と同じ幅で描く（画面と食い違わないように）
        bounds = self.checker.ui_bounds(line, lever.driver) or \
            self.checker.bounds(line, code, lever.driver)
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
