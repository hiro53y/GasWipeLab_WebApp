"""入力が実績分布のどこにいるかを判定する（OOD判定）。

実績モデルは観測データの統計的補間であり、学習範囲の外では保証がない。
そのため予測を返す前に、入力が実績のどこに位置するかを必ず添えて返す。

判定:
    in_range            : ライン・付着量記号別の P01〜P99 の内側
    caution             : 記号別範囲の外だがライン全体の min〜max の内側 / 未学習カテゴリ /
                          圧力-流量の組合せが実績からずれている
    out_of_distribution : ライン全体の min〜max の外 / 未学習の付着量記号 / 主要入力が欠損
"""
from __future__ import annotations

import bisect
import math
from typing import Any

IN_RANGE = "in_range"
CAUTION = "caution"
OUT_OF_DISTRIBUTION = "out_of_distribution"

_RANK = {IN_RANGE: 0, CAUTION: 1, OUT_OF_DISTRIBUTION: 2}

#: これが欠けていると予測が成り立たない項目。
REQUIRED_FEATURES = (
    "製品板厚_mm", "製品板幅_mm",
    "表ノズル位置平均_mm", "裏ノズル位置平均_mm",
    "表ノズル圧力_kPa", "裏ノズル圧力_kPa",
    "中央速度_m_min",
)

#: 圧力-流量の実績関係からの許容ずれ（残差標準偏差の何倍まで許すか）。
FLOW_SIGMA_LIMIT = 3.0

#: CH_direct と CF+CG の食い違いがこれを超えたら注意（引継ぎ仕様の診断指標）。
CONSISTENCY_GAP_LIMIT_GM2 = 8.0

#: 画面で選べる範囲。実績が狭くても設備仕様の範囲までは選べるようにする。
#: ここを外れた入力は外挿（境界の傾きを延長）で計算し、実績範囲外として警告する。
UI_RANGE_OVERRIDES: dict[str, tuple[float, float]] = {
    "製品板厚_mm": (0.20, 3.2),
    "製品板幅_mm": (600.0, 1350.0),
    "中央速度_m_min": (40.0, 165.0),
}

#: 上表にない項目は、実績スパンのこの割合だけ外側へ広げて選べるようにする。
UI_RANGE_EXPAND = 0.25

#: 目標めっき付着量（両面合計）の入力範囲。
TARGET_RANGE_GM2 = (40.0, 300.0)

#: 画面を開いたときの製品サイズ。ライン・付着量記号によらずここから始める
#: （実績中央値だと記号を変えるたびにサイズが動いて比べにくいため）。
DEFAULT_PRODUCT_SIZE: dict[str, float] = {
    "製品板厚_mm": 0.40,
    "製品板幅_mm": 1219.0,
}


def _worse(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


class RangeChecker:
    def __init__(self, reference: dict[str, Any]) -> None:
        self.lines: dict[str, Any] = reference["lines"]

    # ------------------------------------------------------------------
    def known_codes(self, line: str) -> list[str]:
        info = self.lines.get(line)
        return sorted(info["codes"]) if info else []

    def code_stats(self, line: str, code: str) -> dict[str, Any] | None:
        info = self.lines.get(line)
        if not info:
            return None
        entry = info["codes"].get(code)
        return entry["stats"] if entry else None

    def defaults(self, line: str, code: str) -> dict[str, float]:
        """実機設計の初期条件。設備条件はその記号の実績中央値を使う。

        製品板厚・製品板幅だけは `DEFAULT_PRODUCT_SIZE` で固定する。
        速度は記号別中央値のままだと固定した初期サイズと噛み合わないことがある
        （例: GI記号「122」の記号別中央値は57.0 m/minだが、0.40mm×1219mmの実績中央値は
        125.0 m/min）。`speed_for_size` が求まればその値で上書きし、求まらない記号は
        従来どおり記号別中央値をフォールバックとして使う。
        """
        info = self.lines.get(line)
        if not info:
            return {}
        entry = info["codes"].get(code)
        source = entry["features"] if entry else info["features"]
        values = {name: stat["median"] for name, stat in source.items() if stat}
        for name, fixed in DEFAULT_PRODUCT_SIZE.items():
            if name in values:
                values[name] = fixed
        if "中央速度_m_min" in values:
            hint = self.speed_for_size(
                line, DEFAULT_PRODUCT_SIZE["製品板厚_mm"], DEFAULT_PRODUCT_SIZE["製品板幅_mm"])
            if hint is not None:
                values["中央速度_m_min"] = hint["value"]
        return values

    def bounds(self, line: str, code: str, feature: str,
               low: str = "p01", high: str = "p99") -> tuple[float, float] | None:
        """探索に使う範囲。記号別の実績があればそれを、無ければライン全体を使う。"""
        info = self.lines.get(line)
        if not info:
            return None
        entry = info["codes"].get(code)
        stat = (entry["features"].get(feature) if entry else None) or info["features"].get(feature)
        if not stat:
            return None
        lo, hi = stat[low], stat[high]
        return (lo, hi) if hi > lo else (stat["min"], stat["max"])

    def ui_bounds(self, line: str, feature: str) -> tuple[float, float] | None:
        """画面のスライダーで選べる範囲。実績より広く取る。

        設備仕様として指定された項目は固定値、それ以外は実績スパンを外側へ広げる。
        """
        override = UI_RANGE_OVERRIDES.get(feature)
        if override:
            return override
        info = self.lines.get(line)
        stat = info["features"].get(feature) if info else None
        if not stat:
            return None
        low, high = float(stat["min"]), float(stat["max"])
        span = high - low
        if span <= 0:
            return (low - 1.0, high + 1.0)
        margin = span * UI_RANGE_EXPAND
        expanded_low = low - margin
        # 実績が全て0以上の量（位置・圧力・流量など）は負にしない。
        # WS-DS差のように実績が負を含む項目は、そのまま下へ広げる。
        if low > 0.0:
            # さらに、実績最小値の半分より下へは広げない。
            # Yノズル位置のようにレンジが広い項目は margin が最小値を上回り、
            # 単純に0で止めると「Yノズル位置 0 mm」という設備座標の原点を
            # 提案してしまう（実際に GL/AZM100 で発生した）。
            expanded_low = max(expanded_low, low * 0.5)
        elif low == 0.0:
            expanded_low = 0.0
        return (expanded_low, high + margin)

    def ui_ranges(self, line: str) -> dict[str, list[float]]:
        """画面へ渡すスライダー範囲一式。"""
        info = self.lines.get(line)
        if not info:
            return {}
        out: dict[str, list[float]] = {}
        for feature in info["features"]:
            bounds = self.ui_bounds(line, feature)
            if bounds:
                out[feature] = [round(bounds[0], 4), round(bounds[1], 4)]
        return out

    def trends(self, line: str) -> dict[str, dict[str, float]]:
        """実データから求めた偏効果（記号内で中心化した多変量回帰の係数）。

        モデルの局所傾きが実データと逆向きになったときの検証と、
        効果が特定できない項目を提案から外す判定に使う。
        """
        info = self.lines.get(line)
        return (info or {}).get("trends", {})

    def trend_sign(self, line: str, feature: str) -> float:
        """実データ上の効果の向き。判定できないときは 0。"""
        entry = self.trends(line).get(feature)
        if not entry:
            return 0.0
        # ばらつき全体でほとんど動かない項目は、向きも信頼できない
        if entry.get("swing_gm2", 0.0) < 0.5:
            return 0.0
        beta = entry.get("beta_gm2_per_unit", 0.0)
        return 1.0 if beta > 0 else (-1.0 if beta < 0 else 0.0)

    # ------------------------------------------------------------------
    # 組合せとして実績にあるか（多変量の近傍判定）
    #
    # 項目ごとのレンジ判定だけでは、「板厚0.4 も 速度160 もそれぞれ実績内だが、
    # その2つを同時に満たすコイルは1本も無い」という入力を通してしまう。
    # 付着量記号ごとの実績分布に対するマハラノビス距離で、共起関係まで見る。
    # ------------------------------------------------------------------
    def combination_distance(self, line: str, code: str,
                             values: dict[str, Any]) -> dict[str, Any] | None:
        """実績の雲からの距離。判定できないときは None。"""
        info = self.lines.get(line)
        entry = (info or {}).get("codes", {}).get(code)
        envelope = (entry or {}).get("envelope")
        if not envelope:
            return None
        features = envelope["features"]
        vector = []
        for name in features:
            raw = values.get(name)
            if raw is None:
                return None
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return None
            if math.isnan(number):
                return None
            vector.append(number)

        mean = envelope["mean"]
        inverse = envelope["inv_cov"]
        delta = [vector[i] - mean[i] for i in range(len(features))]
        # d^2 = delta^T * inv_cov * delta
        squared = 0.0
        for i, row in enumerate(inverse):
            squared += delta[i] * sum(row[j] * delta[j] for j in range(len(delta)))
        distance = math.sqrt(max(squared, 0.0))

        p90 = float(envelope["d_p90"])
        p99 = float(envelope["d_p99"])
        if distance <= p90:
            level = "typical"
        elif distance <= p99:
            level = "unusual"
        else:
            level = "unseen"
        return {
            "distance": round(distance, 3),
            "d_p90": round(p90, 3),
            "d_p99": round(p99, 3),
            "level": level,
            "features": features,
            "n": envelope["n"],
        }

    def model_ranges(self, line: str) -> dict[str, tuple[float, float]]:
        """モデルが実際に学習した範囲。外挿量の判定に使う。"""
        info = self.lines.get(line)
        if not info:
            return {}
        return {name: (float(stat["min"]), float(stat["max"]))
                for name, stat in info["features"].items() if stat}

    def expected_flow(self, line: str, pressure_kpa: float, speed_mpm: float) -> float | None:
        """圧力・速度から実績どおりのノズル吹込流量を推定する。"""
        info = self.lines.get(line)
        if not info or pressure_kpa <= 0 or speed_mpm <= 0:
            return None
        flow = info["flow"]
        ln = (flow["intercept"]
              + flow["ln_pressure"] * math.log(pressure_kpa)
              + flow["ln_speed"] * math.log(speed_mpm))
        return math.exp(ln)

    def speed_for_size(self, line: str, thickness_mm: float, width_mm: float) -> dict[str, Any] | None:
        """製品板厚・製品板幅から、実績に基づく通板速度を返す。

        ライン単位の板厚×板幅グリッド（`tools/build_reference_data.py` の
        `fit_speed_by_size` が作る `speed_by_size`）を、セル→同じ板厚帯内の
        近いセル→板厚帯→近い板厚帯→ライン中央値の順にたどる。
        実績の外側へは絶対に延ばさない（外挿しない）。
        `speed_by_size` が無い旧版の reference.json や未知のラインでは None を返す。
        """
        try:
            info = self.lines.get(line)
            if not info:
                return None
            speed = info.get("speed_by_size")
            if not speed:
                return None
            thickness = float(thickness_mm)
            width = float(width_mm)
            if math.isnan(thickness) or math.isnan(width):
                return None

            thickness_edges = speed["thickness_edges"]
            width_edges = speed["width_edges"]
            cells = speed.get("cells", {})
            thickness_rows = speed.get("thickness_rows", {})

            def bucket(edges: list[float], value: float) -> int:
                idx = bisect.bisect_right(edges, value) - 1
                return max(0, min(idx, len(edges) - 2))

            ti = bucket(thickness_edges, thickness)
            wi = bucket(width_edges, width)
            clamped = (thickness < thickness_edges[0] or thickness > thickness_edges[-1]
                       or width < width_edges[0] or width > width_edges[-1])

            cell = cells.get(f"{ti},{wi}")
            if cell is not None:
                return {
                    "value": float(cell["median"]),
                    "n": cell.get("n"),
                    "basis": "cell",
                    "thickness_range": [thickness_edges[ti], thickness_edges[ti + 1]],
                    "width_range": [width_edges[wi], width_edges[wi + 1]],
                    "clamped": clamped,
                }

            # 同じ板厚帯の中で、板幅が最も近いセルを探す
            same_ti_widths = [int(key.split(",")[1]) for key in cells
                              if int(key.split(",")[0]) == ti]
            if same_ti_widths:
                nearest_wi = min(same_ti_widths, key=lambda w: abs(w - wi))
                cell = cells[f"{ti},{nearest_wi}"]
                return {
                    "value": float(cell["median"]),
                    "n": cell.get("n"),
                    "basis": "cell",
                    "thickness_range": [thickness_edges[ti], thickness_edges[ti + 1]],
                    "width_range": [width_edges[nearest_wi], width_edges[nearest_wi + 1]],
                    "clamped": clamped,
                }

            row = thickness_rows.get(str(ti))
            if row is not None:
                return {
                    "value": float(row["median"]),
                    "n": row.get("n"),
                    "basis": "thickness_row",
                    "thickness_range": [thickness_edges[ti], thickness_edges[ti + 1]],
                    "width_range": None,
                    "clamped": clamped,
                }

            if thickness_rows:
                nearest_ti = min((int(key) for key in thickness_rows), key=lambda t: abs(t - ti))
                row = thickness_rows[str(nearest_ti)]
                return {
                    "value": float(row["median"]),
                    "n": row.get("n"),
                    "basis": "thickness_row",
                    "thickness_range": [thickness_edges[nearest_ti], thickness_edges[nearest_ti + 1]],
                    "width_range": None,
                    "clamped": clamped,
                }

            return {
                "value": float(speed["line_median"]),
                "n": speed.get("n"),
                "basis": "line_median",
                "thickness_range": None,
                "width_range": None,
                "clamped": clamped,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    def evaluate(self, line: str, code: str, values: dict[str, Any],
                 consistency_gap: float | None = None) -> dict[str, Any]:
        info = self.lines.get(line)
        if not info:
            return {"status": OUT_OF_DISTRIBUTION, "fields": {},
                    "warnings": [f"ライン区分 '{line}' は学習対象外です。"]}

        warnings: list[str] = []
        fields: dict[str, str] = {}
        status = IN_RANGE

        entry = info["codes"].get(code)
        if entry is None:
            status = OUT_OF_DISTRIBUTION
            warnings.append(
                f"付着量記号 '{code}' は学習データにありません（{line}の学習済み記号: "
                f"{', '.join(sorted(info['codes']))}）。"
            )

        for name, stat in info["features"].items():
            raw = values.get(name)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                if name in REQUIRED_FEATURES:
                    fields[name] = OUT_OF_DISTRIBUTION
                    status = OUT_OF_DISTRIBUTION
                    warnings.append(f"{name} が未入力です。予測に必須の項目です。")
                continue
            value = float(raw)
            if value < stat["min"] or value > stat["max"]:
                fields[name] = OUT_OF_DISTRIBUTION
                status = OUT_OF_DISTRIBUTION
                warnings.append(
                    f"{name} = {value:g} は{line}の実績範囲（{stat['min']:g}〜{stat['max']:g}）の外です。"
                )
                continue
            code_stat = entry["features"].get(name) if entry else None
            if code_stat and not (code_stat["p01"] <= value <= code_stat["p99"]):
                fields[name] = CAUTION
                status = _worse(status, CAUTION)
                warnings.append(
                    f"{name} = {value:g} は{code}の通常操業域（{code_stat['p01']:g}〜"
                    f"{code_stat['p99']:g}）から外れています。"
                )
            else:
                fields[name] = IN_RANGE

        for name, options in info["categories"].items():
            raw = values.get(name)
            if raw is None:
                continue
            text = str(raw).strip() or "__MISSING__"
            if text not in {option[0] for option in options}:
                fields[name] = CAUTION
                status = _worse(status, CAUTION)
                warnings.append(f"{name} = '{text}' は{line}の学習データにない区分です。")

        # 圧力と流量は実績上ほぼ連動している。片方だけ動かした入力を検出する。
        pressure = values.get("表ノズル圧力_kPa")
        back = values.get("裏ノズル圧力_kPa")
        speed = values.get("中央速度_m_min")
        flow = values.get("ノズル吹込流量_Nm3H")
        if None not in (pressure, back, speed, flow):
            mean_pressure = (float(pressure) + float(back)) / 2.0
            expected = self.expected_flow(line, mean_pressure, float(speed))
            if expected and float(flow) > 0:
                deviation = abs(math.log(float(flow) / expected))
                limit = FLOW_SIGMA_LIMIT * info["flow"]["resid_std"]
                if deviation > limit:
                    status = _worse(status, CAUTION)
                    warnings.append(
                        f"ノズル吹込流量 {float(flow):.0f} Nm³/h は、この圧力での実績値"
                        f"（約 {expected:.0f} Nm³/h）から外れています。実績では圧力と流量は連動しています。"
                    )

        # 各項目が範囲内でも、その組合せが実績に無いことがある。
        # 判定は CAUTION 止まりにする（個々の値は実績内なので「範囲外」とは言えない）。
        combination = self.combination_distance(line, code, values)
        if combination:
            if combination["level"] == "unseen":
                status = _worse(status, CAUTION)
                warnings.append(
                    f"各項目は実績範囲内ですが、この {len(combination['features'])} 項目の"
                    f"組合せは{code}の実績（{combination['n']}件）に見当たりません"
                    f"（実績分布からの距離 {combination['distance']:.1f}／実績の99%は "
                    f"{combination['d_p99']:.1f} 以内）。予測の信頼度が下がります。"
                )
            elif combination["level"] == "unusual":
                warnings.append(
                    f"この組合せは{code}の実績としては珍しい部類です"
                    f"（実績分布からの距離 {combination['distance']:.1f}／実績の90%は "
                    f"{combination['d_p90']:.1f} 以内）。"
                )

        if consistency_gap is not None and abs(consistency_gap) > CONSISTENCY_GAP_LIMIT_GM2:
            status = _worse(status, CAUTION)
            warnings.append(
                f"表裏合算モデルと両面直接モデルの差が {consistency_gap:+.1f} g/m² あります。"
                "予測の不確かさが大きい条件です。"
            )

        return {"status": status, "fields": fields, "warnings": warnings,
                "combination": combination}
