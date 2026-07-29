"""実機YG装置監視画面の再現。

ProC由来の条件と実機モデルの予測を、実機HMIと同じ並びの構造化データへ組み替える。
ここでは推論を一切行わない（渡された予測結果をそのまま配置するだけ）。

各値には次を必ず添える:
    source     : proc    ProCに実在する列から取得
                 model   実機モデルの予測値
                 derived ProCの値から算術的に導出（新しい仮定は置かない）
                 none    ProCに該当列が無く再現できない（未取得）
    confidence : confirmed  実データ照合で対応が確定
                 likely     ほぼ確実だが正式なタグ定義は未確認
                 unverified 対応関係が未確認。同義扱いしてはいけない
"""
from __future__ import annotations

from typing import Any

from gaswipelab.hmi import control_modes

PROC = "proc"
MODEL = "model"
DERIVED = "derived"
NONE = "none"

CONFIRMED = "confirmed"
LIKELY = "likely"
UNVERIFIED = "unverified"

#: FL〜FO は「設備座標・駆動軸位置」であり、鋼帯からの実距離とは確認できていない。
Y_COORD_NOTE = "設備Y座標。鋼帯表面からの実距離であるとは確認できていません。"

#: ProC に該当列が存在しない項目に共通で添える説明。
NO_SOURCE_NOTE = "ProCに該当列がないため再現できません。"


def _field(value: Any, unit: str = "", source: str = PROC,
           confidence: str = CONFIRMED, note: str = "", label: str = "") -> dict[str, Any]:
    return {"label": label, "value": value, "unit": unit,
            "source": source, "confidence": confidence, "note": note}


def _missing(label: str, unit: str = "", note: str = "") -> dict[str, Any]:
    """未取得。0埋めせず値なしで返す。"""
    return _field(None, unit, NONE, UNVERIFIED, note or NO_SOURCE_NOTE, label)


def _num(condition: dict[str, Any], key: str) -> float | None:
    value = condition.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # NaN を除く


def split_ws_ds(mean: float | None, ws_minus_ds: float | None) -> tuple[float | None, float | None]:
    """「平均」と「WS-DS差」から WS 側・DS 側の値を復元する。

        平均 = (WS + DS) / 2 , 差 = WS - DS
        →  WS = 平均 + 差/2 ,  DS = 平均 - 差/2

    既存特徴量の定義そのままの逆算で、新しい仮定は置いていない。
    """
    if mean is None:
        return None, None
    delta = 0.0 if ws_minus_ds is None else ws_minus_ds
    return mean + delta / 2.0, mean - delta / 2.0


def build_screen(condition: dict[str, Any], prediction: dict[str, Any] | None = None,
                 modes: dict[str, Any] | None = None,
                 model_version: str = "") -> dict[str, Any]:
    """装置画面の表示データを組み立てる。

    Parameters
    ----------
    condition : ProC正準名の条件（実機モードで使っているもの）
    prediction : 実機モデルの予測結果（`GasWipingPredictor.predict` の戻り値）
    modes : 制御モードの状態。取得できていなければ None
    """
    thickness = _num(condition, "製品板厚_mm")

    front_mean = _num(condition, "表ノズル位置平均_mm")
    front_diff = _num(condition, "表ノズル位置WS-DS_mm")
    back_mean = _num(condition, "裏ノズル位置平均_mm")
    back_diff = _num(condition, "裏ノズル位置WS-DS_mm")
    front_ws, front_ds = split_ws_ds(front_mean, front_diff)
    back_ws, back_ds = split_ws_ds(back_mean, back_diff)

    height_ws, height_ds = split_ws_ds(_num(condition, "ノズル高さ平均_mm"),
                                       _num(condition, "ノズル高さWS-DS_mm"))
    roll_ws, roll_ds = split_ws_ds(_num(condition, "コレクティングロール平均_mm"),
                                   _num(condition, "コレクティングロールWS-DS_mm"))

    # 画面の「ノズル間隔（全）」は 片側＋板厚＋反対側。FL〜FO が片側ギャップである確認が
    # 取れていないため、あくまで参考値として unverified を付ける。
    def total_gap(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or thickness is None:
            return None
        return a + thickness + b

    gap_note = ("参考値。片側値＋板厚＋反対側値で計算しています。"
                "画面の全間隔と同じ量であるかは未確認です。")

    coating: dict[str, Any] = {}
    if prediction:
        coating = {
            "back": _field(prediction.get("CG_pred_g_m2"), "g/m²", MODEL, UNVERIFIED,
                           "裏面平均CGの予測値。画面の表裏の向きとCF/CGの対応は未確認です。", "製品裏"),
            "total": _field(prediction.get("CH_sum_pred_g_m2"), "g/m²", MODEL, CONFIRMED,
                            "両面合計。CF+CG で算出しています。", "和"),
            "front": _field(prediction.get("CF_pred_g_m2"), "g/m²", MODEL, UNVERIFIED,
                            "表面平均CFの予測値。画面の表裏の向きとCF/CGの対応は未確認です。", "製品表"),
            "device_total": _missing("装置側の和", "g/m²",
                                     "装置が表示する合計値。ProCに該当列がないため取得できません。"),
            "direct_total": _field(prediction.get("CH_direct_pred_g_m2"), "g/m²", MODEL, CONFIRMED,
                                   "両面直接モデルの予測。診断用で、主要値には使いません。", "両面直接モデル"),
        }
    else:
        coating = {
            "back": _missing("製品裏", "g/m²", "予測がまだ計算されていません。"),
            "total": _missing("和", "g/m²", "予測がまだ計算されていません。"),
            "front": _missing("製品表", "g/m²", "予測がまだ計算されていません。"),
            "device_total": _missing("装置側の和", "g/m²"),
            "direct_total": _missing("両面直接モデル", "g/m²"),
        }

    normalized_modes = control_modes.normalize(modes)

    return {
        "title": "YG装置監視",
        "model_version": model_version,
        "line": str(condition.get("line", "")).strip(),
        "coil": {
            "order_no": _missing("指令No."),
            "coil_no": _missing("コイルNo."),
            "thickness": _field(thickness, "mm", PROC, CONFIRMED, "", "板厚"),
            "width": _field(_num(condition, "製品板幅_mm"), "mm", PROC, CONFIRMED, "", "板幅"),
            "coating_code": _field(str(condition.get("coating_code", "")).strip(), "", PROC,
                                   CONFIRMED, "", "メッキ付着量記号"),
        },
        "coating": coating,
        "speed": _field(_num(condition, "中央速度_m_min"), "m/min", PROC, CONFIRMED,
                        "実データで画面値と一致を確認済み。", "中央速度"),
        "nozzle": {
            "total_gap_ws": _field(total_gap(front_ws, back_ws), "mm", DERIVED, UNVERIFIED,
                                   gap_note, "ノズル間隔 WS"),
            "total_gap_ds": _field(total_gap(front_ds, back_ds), "mm", DERIVED, UNVERIFIED,
                                   gap_note, "ノズル間隔 DS"),
            "front_ws": _field(front_ws, "mm", DERIVED, UNVERIFIED, Y_COORD_NOTE, "表 WS"),
            "front_ds": _field(front_ds, "mm", DERIVED, UNVERIFIED, Y_COORD_NOTE, "表 DS"),
            "back_ws": _field(back_ws, "mm", DERIVED, UNVERIFIED, Y_COORD_NOTE, "裏 WS"),
            "back_ds": _field(back_ds, "mm", DERIVED, UNVERIFIED, Y_COORD_NOTE, "裏 DS"),
            "height_ws": _field(height_ws, "mm", DERIVED, LIKELY,
                                "浴面からノズルスリットまでの高さ。", "ノズル高さ WS"),
            "height_ds": _field(height_ds, "mm", DERIVED, LIKELY,
                                "浴面からノズルスリットまでの高さ。", "ノズル高さ DS"),
            "pressure_front": _field(_num(condition, "表ノズル圧力_kPa"), "kPa", PROC, LIKELY,
                                     "実データ照合により、画面の「ノズル圧力」に対応します"
                                     "（ワイピング圧力ではありません）。", "ノズル圧力 表"),
            "pressure_back": _field(_num(condition, "裏ノズル圧力_kPa"), "kPa", PROC, LIKELY,
                                    "実データ照合により、画面の「ノズル圧力」に対応します"
                                    "（ワイピング圧力ではありません）。", "ノズル圧力 裏"),
        },
        "rolls": {
            "support_roll_ws": _missing("サポートロール回転 WS", "%"),
            "support_roll_ds": _missing("サポートロール回転 DS", "%"),
            "collecting_ws": _field(roll_ws, "mm", DERIVED, UNVERIFIED,
                                    "ProCの列名は「コレクティングロール位置」で、"
                                    "画面の「シフト」と同一かは未確認です。", "コレクティングロール WS"),
            "collecting_ds": _field(roll_ds, "mm", DERIVED, UNVERIFIED,
                                    "ProCの列名は「コレクティングロール位置」で、"
                                    "画面の「シフト」と同一かは未確認です。", "コレクティングロール DS"),
            "pot_temp": _field(_num(condition, "使用ポット温度_C"), "℃", PROC, CONFIRMED,
                               "", "ポット温度"),
        },
        "air": {
            "compressor_discharge_pressure": _missing("Yコンプレッサー吐出圧力", "kPa"),
            "wiping_header_pressure": _missing("ワイピング圧力", "kPa"),
            "total_nozzle_flow": _field(_num(condition, "ノズル吹込流量_Nm3H"), "Nm³/h", PROC, LIKELY,
                                        "表裏合計の実測流量（暫定定義）。", "ノズル吹付流量"),
            "vent_valve_mv": _missing("放風弁MV", "%"),
            "compressor_frequency": _missing("No.3Y機周波数", "%"),
            "active_compressor_pattern": _missing("Y機コンプレッサー"),
            "nozzle_air_temp": _field(_num(condition, "ノズル空気温度_C"), "℃", PROC, LIKELY,
                                      "画面には表示がありませんが、モデル入力に使っています。",
                                      "ノズル空気温度"),
        },
        "other": {
            "weld_pot_front": _missing("溶接ポット 前", "m"),
            "weld_pot_back": _missing("溶接ポット 後", "m"),
            "weld_point_process": _missing("溶接点処理"),
            "baffle_plate": _missing("バッフルプレート開閉量"),
        },
        "control_modes": control_modes.rows(normalized_modes),
        "suppression": control_modes.suppression_status(normalized_modes),
    }
