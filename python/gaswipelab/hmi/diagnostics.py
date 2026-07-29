"""装置画面から計算できる診断値。

計算に必要な値が取れていない項目は、0やダミーで埋めずに value=None で返す。
「差が0」と「そもそも測れていない」を混同させないため。
"""
from __future__ import annotations

from typing import Any

#: 画面のノズル全間隔と、片側＋板厚＋反対側 の食い違いの許容幅 [mm]
GAP_CONSISTENCY_TOLERANCE_MM = 0.1

#: 装置側の合計と CF+CG の食い違いの許容幅 [g/m²]
COATING_CONSISTENCY_TOLERANCE_GM2 = 0.5


def _value(field: dict[str, Any] | None) -> float | None:
    if not field:
        return None
    value = field.get("value")
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _diff(label: str, a: float | None, b: float | None, unit: str,
          note: str = "", tolerance: float | None = None) -> dict[str, Any]:
    if a is None or b is None:
        return {"label": label, "value": None, "unit": unit,
                "available": False, "warning": False,
                "note": note or "必要な値が取得できていないため計算できません。"}
    delta = a - b
    warning = tolerance is not None and abs(delta) > tolerance
    return {"label": label, "value": round(delta, 3), "unit": unit,
            "available": True, "warning": warning, "note": note}


def compute(screen: dict[str, Any]) -> list[dict[str, Any]]:
    """指示書 §4 の診断値。計算できないものは available=False で返す。"""
    coating = screen.get("coating", {})
    nozzle = screen.get("nozzle", {})
    rolls = screen.get("rolls", {})
    air = screen.get("air", {})

    front_cw = _value(coating.get("front"))
    back_cw = _value(coating.get("back"))
    total_cw = _value(coating.get("total"))
    device_total = _value(coating.get("device_total"))
    direct_total = _value(coating.get("direct_total"))

    results = [
        _diff("表裏の付着量差（表−裏）", front_cw, back_cw, "g/m²"),
        _diff("装置側の和 −（表＋裏）", device_total, total_cw, "g/m²",
              tolerance=COATING_CONSISTENCY_TOLERANCE_GM2),
        _diff("両面直接モデル −（表＋裏）", direct_total, total_cw, "g/m²",
              note="モデル内部の整合診断です。", tolerance=None),
        _diff("表裏のノズル圧力差（表−裏）", _value(nozzle.get("pressure_front")),
              _value(nozzle.get("pressure_back")), "kPa"),
        _diff("ノズル間隔の WS−DS 差", _value(nozzle.get("total_gap_ws")),
              _value(nozzle.get("total_gap_ds")), "mm",
              note="片側値＋板厚から導出した参考値どうしの差です。"),
        _diff("コレクティングロールの WS−DS 差", _value(rolls.get("collecting_ws")),
              _value(rolls.get("collecting_ds")), "mm"),
        _diff("サポートロール回転差（WS−DS）", _value(rolls.get("support_roll_ws")),
              _value(rolls.get("support_roll_ds")), "%"),
        _diff("吐出圧力 − ワイピング圧力", _value(air.get("compressor_discharge_pressure")),
              _value(air.get("wiping_header_pressure")), "kPa"),
        _diff("ワイピング圧力 − ノズル圧力", _value(air.get("wiping_header_pressure")),
              _value(nozzle.get("pressure_front")), "kPa"),
        _diff("吐出圧力 − ノズル圧力", _value(air.get("compressor_discharge_pressure")),
              _value(nozzle.get("pressure_front")), "kPa"),
    ]
    return results


def gap_consistency(total_gap: float | None, side_a: float | None,
                    thickness: float | None, side_b: float | None) -> dict[str, Any]:
    """画面のノズル全間隔と「片側＋板厚＋反対側」の整合を確認する。

    受入テスト（指示書 §6）で使う:
        9.6 + 0.4 + 9.5 = 19.5 → 一致
        9.6 + 0.4 + 9.4 = 19.4 → 一致
    """
    if None in (total_gap, side_a, thickness, side_b):
        return {"available": False, "consistent": None, "difference": None,
                "note": "必要な値が取得できていないため確認できません。"}
    computed = side_a + thickness + side_b
    difference = total_gap - computed
    consistent = abs(difference) <= GAP_CONSISTENCY_TOLERANCE_MM
    return {
        "available": True,
        "consistent": consistent,
        "computed": round(computed, 3),
        "difference": round(difference, 3),
        "note": "" if consistent else
                f"画面の全間隔と片側値の合計が {difference:+.2f} mm 食い違っています。",
    }


def coating_consistency(device_total: float | None, front: float | None,
                        back: float | None) -> dict[str, Any]:
    """装置側の合計と（表＋裏）の整合を確認する。

    実機画面では 66.2 + 66.2 = 132.4 に対し装置表示が 131.2 で、1.2 g/m² 食い違う。
    表・裏・和の更新時刻または定義が異なる可能性があるため WARNING とする。
    """
    if None in (device_total, front, back):
        return {"available": False, "consistent": None, "difference": None,
                "note": "装置側の合計値がProCにないため確認できません。"}
    computed = front + back
    difference = device_total - computed
    consistent = abs(difference) <= COATING_CONSISTENCY_TOLERANCE_GM2
    return {
        "available": True,
        "consistent": consistent,
        "computed": round(computed, 3),
        "difference": round(difference, 3),
        "note": "" if consistent else
                f"装置側の和が（表＋裏）より {difference:+.1f} g/m² ずれています。"
                "表・裏・和の更新時刻または定義が異なる可能性があります。",
    }
