"""薄膜方程式の数値解（最大流束原理）— v2.0

潤滑近似に基づく標準モデル（Thornton & Graff 1976 → Tuck 1983 →
Ellen & Tu 1984 → Elsaadawy et al. 2007）を実装する。

無次元化:
    H = h sqrt(rho_L g / (mu_L Vs))          無次元膜厚
    S = tau / sqrt(rho_L mu_L Vs g)          無次元せん断応力（上向き正）
    G = 1 + (1/(rho_L g)) dp/dx              無次元実効重力（x: 引き上げ方向）
    Q = (q/Vs) sqrt(rho_L g / (mu_L Vs))     無次元持ち上げ流束

流束関数（Ellen & Tu 1984）:
    Q(H) = H + S H^2 / 2 - G H^3 / 3

最大流束原理（Tuck 1983）:
    各位置 x で dQ/dH = 0 → H*(x) = [S + sqrt(S^2 + 4G)] / (2G)  (G > 0)
    持ち上げ可能な最大流束 Q_max(x) = Q(H*(x))
    実際の流束はワイピング域のボトルネック: Q_w = min_x Q_max(x)
    最終膜厚: h_f = Q_w sqrt(mu_L Vs / (rho_L g))

噴流なし（S=0, G=1）では H*=1, Q=2/3 となり、自由引き上げの
Thornton-Graff 極限 h_f = (2/3) sqrt(mu_L Vs / (rho_L g)) に帰着する。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from gaswipelab.models.units import mpm_to_mps
from gaswipelab.models.zinc_properties import zinc_viscosity_pa_s


@dataclass(frozen=True)
class FilmSolution:
    film_thickness_m: float          # 最終（ワイピング後）液膜厚さ h_f
    base_thickness_m: float          # 自由引き上げ極限 (2/3) sqrt(mu Vs / rho g)
    dimensionless_flux: float        # Q_w
    free_withdrawal_flux: float      # 2/3
    wiping_strength: float           # ワイピング効率 = t0 / h_f
    metering_position_m: float       # 流束最小点（メータリング点）の x 座標
    g_max: float                     # 無次元実効重力の最大値
    s_at_metering: float             # メータリング点の無次元せん断
    bath_viscosity_pa_s: float
    film_profile_m: np.ndarray       # 表示用の局所膜厚 h(x)
    edge_effect_factor: float


def _flux(h, s, g):
    return h + 0.5 * s * h * h - g * h**3 / 3.0


def solve_film(
    x_m: np.ndarray,
    pressure_gradient_pa_m: np.ndarray,
    shear_pa: np.ndarray,
    line_speed_mpm: float,
    rho_l: float,
    mu_l: float,
    gravity: float,
    edge_effect_factor: float = 1.0,
    compute_profile: bool = True,
) -> FilmSolution:
    """壁面分布から最終膜厚を求める。

    compute_profile=False でスイープ・校正など反復用途向けに
    表示用プロファイル計算（三次方程式の根）を省略する。

    shear_pa は引き上げ方向（上向き）を正とする符号付き分布。
    噴流より下（x<0）では負（下向き）でワイピングを助ける。
    """
    v_strip = max(mpm_to_mps(line_speed_mpm), 1.0e-6)
    length_scale = sqrt(mu_l * v_strip / (rho_l * gravity))  # 膜厚スケール
    s_scale = sqrt(rho_l * mu_l * v_strip * gravity)         # せん断スケール

    s_arr = np.asarray(shear_pa, dtype=float) / s_scale
    g_arr = 1.0 + np.asarray(pressure_gradient_pa_m, dtype=float) / (rho_l * gravity)

    # G > 0 の点でのみ最大流束が有限。G<=0（吸い上げ勾配側）は拘束にならない。
    valid = g_arr > 1.0e-9
    q_max = np.full_like(g_arr, np.inf)
    gv = g_arr[valid]
    sv = s_arr[valid]
    h_star = (sv + np.sqrt(sv * sv + 4.0 * gv)) / (2.0 * gv)
    q_max[valid] = _flux(h_star, sv, gv)

    # メータリング点（流束ボトルネック）
    idx = int(np.argmin(q_max))
    q_w = float(q_max[idx])
    # 自由引き上げ極限（噴流の影響が消える遠方）を上限とする。
    q_w = min(q_w, 2.0 / 3.0)
    q_w = max(q_w, 0.0)

    # エッジ・板幅の経験補正（流束に乗算; 1.0 で無補正）。
    # 自由引き上げ極限 2/3 は物理上限のため補正後にも維持する。
    q_w_eff = min(q_w * float(edge_effect_factor), 2.0 / 3.0)

    h_f = q_w_eff * length_scale
    t0 = (2.0 / 3.0) * length_scale

    if compute_profile:
        film_profile = _film_profile(x_m, s_arr, g_arr, q_w_eff, idx, length_scale)
    else:
        film_profile = np.full_like(np.asarray(x_m, dtype=float), q_w_eff * length_scale)

    return FilmSolution(
        film_thickness_m=float(h_f),
        base_thickness_m=float(t0),
        dimensionless_flux=float(q_w_eff),
        free_withdrawal_flux=2.0 / 3.0,
        wiping_strength=float(t0 / max(h_f, 1.0e-12)),
        metering_position_m=float(x_m[idx]),
        g_max=float(np.max(g_arr)),
        s_at_metering=float(s_arr[idx]),
        bath_viscosity_pa_s=float(mu_l),
        film_profile_m=film_profile,
        edge_effect_factor=float(edge_effect_factor),
    )


def _film_profile(
    x_m: np.ndarray,
    s_arr: np.ndarray,
    g_arr: np.ndarray,
    q_w: float,
    metering_idx: int,
    length_scale: float,
) -> np.ndarray:
    """表示用の定常膜厚プロファイル h(x)。

    各 x で流束保存 Q(H) = q_w を満たす H を三次方程式
        -G/3 H^3 + S/2 H^2 + H - q_w = 0
    の実根から選ぶ。メータリング点より上流（上方）は薄膜枝（最小正根）、
    下流（浴側）はランバック厚膜枝（最大正根）を取る。
    根が存在しない点は H*（臨界膜厚）で代替する。
    """
    n = len(x_m)
    h_dimless = np.empty(n)
    for i in range(n):
        g = float(g_arr[i])
        s = float(s_arr[i])
        if g > 1.0e-9:
            h_crit = (s + sqrt(s * s + 4.0 * g)) / (2.0 * g)
        else:
            h_crit = 1.0
        roots = np.roots([-g / 3.0, 0.5 * s, 1.0, -q_w])
        real_roots = [float(rt.real) for rt in roots if abs(rt.imag) < 1.0e-9 and rt.real > 1.0e-12]
        if not real_roots:
            h_dimless[i] = h_crit
            continue
        if i >= metering_idx:
            # メータリング点より上: 薄膜枝
            h_dimless[i] = min(real_roots)
        else:
            # 浴側: ランバック枝（表示の発散を防ぐため臨界値の8倍まで）
            h_dimless[i] = min(max(real_roots), 8.0 * max(h_crit, 1.0))
    return h_dimless * length_scale


def edge_effect_factor_from_width(strip_width_mm: float | None, coefficients: dict) -> float:
    """板幅の経験補正係数（任意・既定では弱い補正、strength=0 で無効）。"""
    film_cfg = coefficients.get("film", {})
    w_ref = float(film_cfg.get("reference_strip_width_mm", 1200.0))
    strength = float(film_cfg.get("edge_effect_strength", 0.05))
    if strip_width_mm is None or strip_width_mm <= 0.0 or strength == 0.0:
        return 1.0
    factor = 1.0 + strength * (1.0 - float(strip_width_mm) / w_ref)
    return float(min(max(factor, 0.7), 1.3))


@dataclass(frozen=True)
class FilmResult:
    """[後方互換] 旧API互換の結果型。"""

    film_thickness_m: float
    base_thickness_m: float
    wiping_strength: float
    edge_effect_factor: float
    bath_viscosity_pa_s: float


def calculate_film_thickness(
    line_speed_mpm: float,
    representative_gradient_pa_m: float,
    representative_shear_pa: float,
    material_config: dict,
    coefficients: dict,
    calibration: dict | None = None,
    bath_temp_c: float | None = None,
    strip_width_mm: float | None = None,
) -> FilmResult:
    """[後方互換ラッパー] 代表値（最大勾配・最大せん断）による0D近似。

    v2.0 の主経路は solve_film()。本関数は旧呼出しの互換用で、最大勾配点に
    最大せん断が同時に働くと仮定した保守側（過剰ワイピング側）の近似を返す。
    """
    zinc = material_config["zinc"]
    gravity = material_config["constants"]["gravity_m_s2"]
    rho_l = float(zinc["density_liquid_kg_m3"])
    mu_l = zinc_viscosity_pa_s(bath_temp_c, material_config) if bath_temp_c is not None else float(zinc["viscosity_pa_s"])

    v_strip = max(mpm_to_mps(line_speed_mpm), 1.0e-6)
    s_scale = sqrt(rho_l * mu_l * v_strip * gravity)
    g_val = 1.0 + abs(float(representative_gradient_pa_m)) / (rho_l * gravity)
    s_val = -abs(float(representative_shear_pa)) / s_scale  # ワイピングを助ける向き

    h_star = (s_val + sqrt(s_val * s_val + 4.0 * g_val)) / (2.0 * g_val)
    q_w = max(min(_flux(h_star, s_val, g_val), 2.0 / 3.0), 0.0)

    edge_factor = edge_effect_factor_from_width(strip_width_mm, coefficients)
    cal = calibration or {}
    scale = float(cal.get("scale_factor", 1.0))

    length_scale = sqrt(mu_l * v_strip / (rho_l * gravity))
    h_f = max(q_w * edge_factor * scale * length_scale, 1.0e-8)
    t0 = (2.0 / 3.0) * length_scale
    return FilmResult(
        film_thickness_m=h_f,
        base_thickness_m=t0,
        wiping_strength=t0 / h_f,
        edge_effect_factor=edge_factor,
        bath_viscosity_pa_s=mu_l,
    )
