from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from gaswipelab.models.units import c_to_k, kpa_to_pa


@dataclass(frozen=True)
class GasState:
    key: str
    display_name: str
    temperature_k: float
    absolute_pressure_pa: float
    density_kg_m3: float
    viscosity_pa_s: float
    speed_of_sound_m_s: float
    r_specific_j_kgk: float
    gamma: float


def normalize_gas_type(gas_type: str) -> str:
    lowered = gas_type.strip().lower()
    if lowered in {"n2", "nitrogen", "窒素", "窒素（n2）", "窒素(n2)"}:
        return "nitrogen"
    if lowered in {"air", "空気", "空気（20℃）", "空気(20℃)"}:
        return "air"
    raise ValueError(f"未対応のガス種です: {gas_type}")


def calculate_gas_state(
    gas_type: str,
    plenum_pressure_kpa: float,
    gas_temperature_c: float,
    material_config: dict,
) -> GasState:
    key = normalize_gas_type(gas_type)
    gas_config = material_config["gas"][key]
    constants = material_config["constants"]
    temperature_k = c_to_k(gas_temperature_c)
    absolute_pressure_pa = constants["atmospheric_pressure_pa"] + kpa_to_pa(plenum_pressure_kpa)
    r_specific = gas_config["r_specific_j_kgk"]
    gamma = gas_config["gamma"]
    density = absolute_pressure_pa / (r_specific * temperature_k)
    speed_of_sound = sqrt(gamma * r_specific * temperature_k)
    return GasState(
        key=key,
        display_name=gas_config["display_name_jp"],
        temperature_k=temperature_k,
        absolute_pressure_pa=absolute_pressure_pa,
        density_kg_m3=density,
        viscosity_pa_s=gas_config["viscosity_pa_s_25c"],
        speed_of_sound_m_s=speed_of_sound,
        r_specific_j_kgk=r_specific,
        gamma=gamma,
    )
