"""CatBoost oblivious tree の純Python推論器（Pyodide 上で動かすため外部依存なし）。

`tools/convert_models.py` が出力するコンパクト形式を読み、CatBoost 本体と
同一の予測値を返す。ゴールデン30件・実データ1000行で最大絶対差 5.7e-14 を確認済み。

CTR（カテゴリ特徴の統計量）の扱い:
    投影ハッシュ  h = MAGIC * (h_prev + MAGIC * value)   （MAGIC=0x4906ba494954cb65）
    カテゴリ値は int32 ハッシュを符号拡張して 64bit として畳み込む。
    Borders : ctr = (境界超え件数 + prior_num) / (総件数 + prior_den)
    Counter : ctr = (件数 + prior_num) / (counter_denominator + prior_den)
    いずれも ctr * scale + shift を borders で二値化する。
"""
from __future__ import annotations

import math
from typing import Any

from gaswipelab.ml.cityhash import M64, cat_feature_hash

MAGIC = 0x4906BA494954CB65

SPLIT_FLOAT = 0
SPLIT_ONEHOT = 1
SPLIT_CTR = 2

CTR_BORDERS = 0
CTR_COUNTER = 1

MISSING = "__MISSING__"


def _calc_hash(previous: int, value: int) -> int:
    return (MAGIC * ((previous + (MAGIC * value & M64)) & M64)) & M64


def to_number(value: Any) -> float:
    """数値化できない値は NaN（CatBoost の欠損扱いと同じ）。"""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def to_category(value: Any) -> str:
    """空欄・None は __MISSING__ に寄せる。前後空白は除去する。"""
    text = "" if value is None else str(value).strip()
    return text or MISSING


class CompactModel:
    """コンパクト形式1本ぶんの推論器。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.depths: list[int] = payload["depths"]
        self.splits: list[list] = payload["splits"]
        self.leaves: list[float] = payload["leaves"]
        self.scale: float = payload["scale"]
        self.bias: float = payload["bias"]
        self.ctrs: list[dict[str, Any]] = payload["ctrs"]
        # ハッシュ表は辞書化しておく（変換時に空きスロットは除去済み）
        self._tables = [dict(zip(c["k"], c["v"])) for c in self.ctrs]

    def _ctr_value(self, index: int, floats: list[float], cats: list[int]) -> float:
        ctr = self.ctrs[index]
        h = 0
        for element in ctr["e"]:
            if element[0] == 0:
                h = _calc_hash(h, cats[element[1]] & M64)
            else:
                value = floats[element[1]]
                h = _calc_hash(h, 1 if (value == value and value > element[2]) else 0)
        entry = self._tables[index].get(h)
        prior_num = ctr["pn"]
        prior_den = ctr["pd"]
        if ctr["t"] == CTR_BORDERS:
            if entry is None:
                good = total = 0
            else:
                good = entry[1]
                total = entry[0] + entry[1]
            value = (good + prior_num) / (total + prior_den)
        else:
            count = entry[0] if entry is not None else 0
            value = (count + prior_num) / (ctr["den"] + prior_den)
        return value * ctr["sc"] + ctr["sh"]

    def predict(self, floats: list[float], cats: list[int]) -> float:
        ctr_cache: dict[int, float] = {}
        splits = self.splits
        leaves = self.leaves
        total = 0.0
        split_pos = 0
        leaf_pos = 0
        for depth in self.depths:
            index = 0
            for bit in range(depth):
                kind, a, b = splits[split_pos + bit]
                if kind == SPLIT_FLOAT:
                    value = floats[a]
                    # NaN はどちらの側にも倒さない（CatBoost の AsIs 準拠）
                    on = value == value and value > b
                elif kind == SPLIT_ONEHOT:
                    on = cats[a] == b
                else:
                    cached = ctr_cache.get(a)
                    if cached is None:
                        cached = self._ctr_value(a, floats, cats)
                        ctr_cache[a] = cached
                    on = cached > b
                if on:
                    index |= 1 << bit
            total += leaves[leaf_pos + index]
            split_pos += depth
            leaf_pos += 1 << depth
        return total * self.scale + self.bias
