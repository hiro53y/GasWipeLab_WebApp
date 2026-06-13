from __future__ import annotations


def mm_to_m(value_mm: float) -> float:
    return value_mm / 1000.0


def m_to_mm(value_m: float) -> float:
    return value_m * 1000.0


def kpa_to_pa(value_kpa: float) -> float:
    return value_kpa * 1000.0


def pa_to_kpa(value_pa: float) -> float:
    return value_pa / 1000.0


def c_to_k(value_c: float) -> float:
    return value_c + 273.15


def mpm_to_mps(value_mpm: float) -> float:
    return value_mpm / 60.0


def m_to_um(value_m: float) -> float:
    return value_m * 1_000_000.0


def um_to_m(value_um: float) -> float:
    return value_um / 1_000_000.0


def kg_m2_to_g_m2(value_kg_m2: float) -> float:
    return value_kg_m2 * 1000.0

