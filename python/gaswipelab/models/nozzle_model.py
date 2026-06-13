from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from gaswipelab.models.gas_properties import GasState
from gaswipelab.models.units import kpa_to_pa


@dataclass(frozen=True)
class NozzleFlow:
    exit_velocity_m_s: float
    mach: float
    dynamic_pressure_pa: float
    mach_warning: bool
    choked: bool = False
    pressure_ratio: float = 1.0
    exit_static_pressure_pa: float = 101325.0
    exit_static_temperature_k: float = 298.15
    exit_density_kg_m3: float = 1.0
    model_name: str = "compressible_isentropic_slot"


def nozzle_exit_flow(plenum_pressure_kpa: float, gas_state: GasState, coefficients: dict) -> NozzleFlow:
    """圧縮性を考慮したスロット出口速度を推定する。

    Mach 0.3を超える条件が実用域に入るため、旧来の非圧縮 Bernoulli 式ではなく
    等エントロピ膨張のオリフィス近似を使う。PyInstaller配布後も係数変更できる
    よう、放出係数はYAMLの値を使う。
    """
    cd = coefficients["nozzle"]["discharge_coefficient"]
    delta_p_pa = kpa_to_pa(plenum_pressure_kpa)
    p0 = max(float(gas_state.absolute_pressure_pa), 1.0)
    p_ambient = max(p0 - delta_p_pa, 1.0)
    gamma = float(gas_state.gamma)
    r_specific = float(gas_state.r_specific_j_kgk)
    t0 = float(gas_state.temperature_k)

    pressure_ratio = min(max(p_ambient / p0, 1.0e-6), 1.0)
    critical_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    choked = pressure_ratio <= critical_ratio
    if choked:
        mach_ideal = 1.0
        exit_pressure = p0 * critical_ratio
        exit_temperature = t0 * (2.0 / (gamma + 1.0))
    else:
        mach_ideal = sqrt(max(0.0, (2.0 / (gamma - 1.0)) * ((1.0 / pressure_ratio) ** ((gamma - 1.0) / gamma) - 1.0)))
        exit_pressure = p_ambient
        exit_temperature = t0 / (1.0 + 0.5 * (gamma - 1.0) * mach_ideal**2)

    speed_of_sound_exit = sqrt(gamma * r_specific * exit_temperature)
    velocity = cd * mach_ideal * speed_of_sound_exit
    # Mach数は等エントロピ理想値を報告する（チョーク時 1.0）。
    # 吐出係数は有効流速・動圧のみに掛ける（面積収縮の効果のため）。
    mach = mach_ideal
    exit_density = exit_pressure / (r_specific * exit_temperature)
    dynamic_pressure = 0.5 * exit_density * velocity**2
    return NozzleFlow(
        exit_velocity_m_s=velocity,
        mach=mach,
        dynamic_pressure_pa=dynamic_pressure,
        mach_warning=mach > 0.3,
        choked=choked,
        pressure_ratio=pressure_ratio,
        exit_static_pressure_pa=exit_pressure,
        exit_static_temperature_k=exit_temperature,
        exit_density_kg_m3=exit_density,
    )
