from __future__ import annotations

from typing import Any


def validate_ranges(values: dict[str, Any], ranges: dict[str, dict[str, float]]) -> list[str]:
    errors: list[str] = []
    for key, rule in ranges.items():
        if key not in values:
            continue
        value = values[key]
        if value < rule["min"] or value > rule["max"]:
            errors.append(f"{key}: {rule['min']} - {rule['max']} の範囲外です。")
    return errors


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

