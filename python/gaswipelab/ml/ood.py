"""入力が実績分布のどこにいるかを判定する（OOD判定）。

実機モデルは観測データの統計的補間であり、学習範囲の外では保証がない。
そのため予測を返す前に、入力が実績のどこに位置するかを必ず添えて返す。

判定:
    in_range            : ライン・付着量記号別の P01〜P99 の内側
    caution             : 記号別範囲の外だがライン全体の min〜max の内側 / 未学習カテゴリ /
                          圧力-流量の組合せが実績からずれている
    out_of_distribution : ライン全体の min〜max の外 / 未学習の付着量記号 / 主要入力が欠損
"""
from __future__ import annotations

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
        if low >= 0.0:
            expanded_low = max(0.0, expanded_low)
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

        if consistency_gap is not None and abs(consistency_gap) > CONSISTENCY_GAP_LIMIT_GM2:
            status = _worse(status, CAUTION)
            warnings.append(
                f"表裏合算モデルと両面直接モデルの差が {consistency_gap:+.1f} g/m² あります。"
                "予測の不確かさが大きい条件です。"
            )

        return {"status": status, "fields": fields, "warnings": warnings}
