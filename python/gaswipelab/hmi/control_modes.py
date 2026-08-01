"""実機画面に表示されている制御モードの定義。

現時点では取得できている実績データに該当タグが1つも存在しないため、すべて UNKNOWN になる。
将来PLCから取得できるようになった場合に備えて、状態の型と判定だけ先に定義しておく。
"""
from __future__ import annotations

from typing import Any

ON = "ON"
OFF = "OFF"
AUTO = "AUTO"
MANUAL = "MANUAL"
UNKNOWN = "UNKNOWN"

VALID_STATES = (ON, OFF, AUTO, MANUAL, UNKNOWN)

#: (キー, 画面表示名, 取りうる状態の説明)
CONTROL_MODES: tuple[tuple[str, str, str], ...] = (
    ("coating_predictive_control", "付着量予測制御", "切／入"),
    ("observer", "オブザーバー", "無／有"),
    ("nozzle_pressure_fb", "ノズル圧力FB制御", "切／入"),
    ("nozzle_gap_fb", "ノズル間隔FB制御", "切／入"),
    ("coating_side_mode", "付着量 表面／裏面", "一括／個別"),
    ("nozzle_pressure_balance", "ノズル圧力バランス", "切／入"),
    ("baffle_test_mode", "バッフルテストモード", "切／入"),
    ("nozzle_quick_open", "ノズルクイックオープン", "無／有"),
    ("nozzle_cleaning", "ノズル清掃準備", "解除／予約"),
    ("wiping_air_heater", "ワイピングエア加温ヒーター", "遠方／停止／運転"),
    ("baffle_plate_opening", "バッフルプレート開閉量", "大／小"),
)

#: これらが有効なときは通常の予測・推奨を出してはいけない。
SUPPRESS_WHEN_ACTIVE = ("baffle_test_mode", "nozzle_quick_open", "nozzle_cleaning")


def default_modes() -> dict[str, str]:
    """すべて UNKNOWN の初期状態。"""
    return {key: UNKNOWN for key, _, _ in CONTROL_MODES}


def normalize(raw: dict[str, Any] | None) -> dict[str, str]:
    """外部から渡された状態を検証し、未知・未指定は UNKNOWN に寄せる。"""
    modes = default_modes()
    if not raw:
        return modes
    for key in modes:
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip().upper()
        modes[key] = text if text in VALID_STATES else UNKNOWN
    return modes


def suppression_status(modes: dict[str, str]) -> dict[str, Any]:
    """推奨を止めるべき状態かどうかを返す。

    状態が取得できていない場合、「止めるべきではない」と判断してはいけない。
    判定不能であることを呼び出し側へ伝える。
    """
    active = [key for key in SUPPRESS_WHEN_ACTIVE if modes.get(key) == ON]
    unknown = [key for key in SUPPRESS_WHEN_ACTIVE if modes.get(key, UNKNOWN) == UNKNOWN]
    labels = {key: label for key, label, _ in CONTROL_MODES}
    if active:
        return {
            "suppress": True,
            "determinable": True,
            "reason": "、".join(labels[key] for key in active) + " が有効なため、推奨を停止します。",
        }
    if unknown:
        return {
            "suppress": False,
            "determinable": False,
            "reason": (
                "バッフルテスト・ノズルクイックオープン・ノズル清掃の状態は記録がないため、"
                "アプリ側では検知できません。これらの作業中は結果を使わないでください。"
            ),
        }
    return {"suppress": False, "determinable": True, "reason": ""}


def rows(modes: dict[str, str]) -> list[dict[str, str]]:
    """画面表示用。色ではなく文字で状態を示す。"""
    return [
        {"key": key, "label": label, "options": options, "state": modes.get(key, UNKNOWN)}
        for key, label, options in CONTROL_MODES
    ]
