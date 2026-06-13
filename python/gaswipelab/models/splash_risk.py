"""スプラッシュ発生リスク評価 — v2.0

Gosset & Buchlin (J. Fluids Eng. 129, 2007) / Myrillas et al.
(Chem. Eng. Process. 50, 2011) の実験的発生限界に基づく。

    有効Weber数:        We  = rho_g V_wj^2 h0 / sigma
    壁面ジェット速度:    V_wj = U_j / (Z/d) * (1 + sin(alpha))
    液膜Reynolds数:     Re_f = rho_L Vs h_f / mu_L
    発生限界:           We* = exp(A alpha_deg + B) * Re_f^(-n)

We/We* >= 1 でスプラッシュ発生域。係数 A, B, n はノズル設計依存で
文献範囲 A=0.018-0.066, B=5.5-7.9, n=1.44-1.91（configs で調整可能）。
h0 はランバック液膜厚さ（実機代表値 0.3-0.5 mm）。

本指標は相対評価であり、絶対的な発生判定ではない。実機校正を推奨する。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, radians, sin

from gaswipelab.models.units import mpm_to_mps
from gaswipelab.models.zinc_properties import zinc_viscosity_pa_s


@dataclass(frozen=True)
class SplashRisk:
    level: str                 # 低 / 中 / 高
    score: float               # We/We*（1.0 で発生限界）
    weber_number: float        # 有効Weber数
    critical_weber: float      # 発生限界 We*
    wall_jet_velocity_m_s: float
    liquid_reynolds: float


def calculate_splash_risk(
    gas_density_kg_m3: float,
    exit_velocity_m_s: float,
    nozzle_gap_mm: float,
    nozzle_strip_distance_mm: float,
    plenum_pressure_kpa: float,
    line_speed_mpm: float,
    film_thickness_m: float,
    material_config: dict,
    coefficients: dict,
    bath_temp_c: float | None = None,
) -> SplashRisk:
    zinc = material_config["zinc"]
    sigma_l = float(zinc["surface_tension_n_m"])
    rho_l = float(zinc["density_liquid_kg_m3"])
    mu_l = zinc_viscosity_pa_s(bath_temp_c, material_config) if bath_temp_c is not None else float(zinc["viscosity_pa_s"])

    cfg = coefficients.get("splash", {})
    h0_m = float(cfg.get("runback_film_thickness_mm", 0.4)) * 1.0e-3
    alpha_deg = float(cfg.get("nozzle_tilt_deg", 0.0))
    coef_a = float(cfg.get("limit_coef_a", 0.042))
    coef_b = float(cfg.get("limit_coef_b", 6.0))
    coef_n = float(cfg.get("limit_exp_n", 1.6))
    low_thr = float(cfg.get("low_threshold", 0.6))
    high_thr = float(cfg.get("high_threshold", 1.0))

    standoff = max(float(nozzle_strip_distance_mm) / max(float(nozzle_gap_mm), 1.0e-6), 1.0)
    v_wall_jet = float(exit_velocity_m_s) / standoff * (1.0 + sin(radians(alpha_deg)))

    weber = float(gas_density_kg_m3) * v_wall_jet * v_wall_jet * h0_m / max(sigma_l, 1.0e-9)
    re_film = rho_l * mpm_to_mps(line_speed_mpm) * max(float(film_thickness_m), 1.0e-9) / max(mu_l, 1.0e-12)

    we_critical = exp(coef_a * alpha_deg + coef_b) * re_film ** (-coef_n)
    score = weber / max(we_critical, 1.0e-12)

    if score >= high_thr:
        level = "高"
    elif score >= low_thr:
        level = "中"
    else:
        level = "低"

    return SplashRisk(
        level=level,
        score=float(score),
        weber_number=float(weber),
        critical_weber=float(we_critical),
        wall_jet_velocity_m_s=float(v_wall_jet),
        liquid_reynolds=float(re_film),
    )
