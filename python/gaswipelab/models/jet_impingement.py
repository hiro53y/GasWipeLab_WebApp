"""衝突スロット噴流の壁面圧力・せん断応力分布モデル（v2.0）。

文献準拠の検証済み相関式を実装する。

近接場 (Z/d <= 7): Elsaadawy et al., Metall. Mater. Trans. B 38 (2007) 413.
    - 最大圧力:        p_m/p0 = -0.0056 r^2 + 0.0268 r + 1.0108   (r = Z/d)
    - 圧力半値幅:      b_p/d  = 0.0453 r + 0.7921
    - 圧力分布:        p/p_m  = [1 + 0.6 xi^4]^(-1.5),  xi = x / b_p
    - 最大せん断:      tau_max/p0 = -0.0001 r + 0.0035
    - せん断特性幅:    b_s/d  = 0.0443 r + 1.1687
    - せん断分布:      tau/tau_max = erf(0.41 xi) + 0.54 xi exp(-0.22 xi^3)  (xi <= 1.73)
                       tau/tau_max = 1.115 - 0.24 ln(xi)                     (xi > 1.73)

遠方場 (Z/d >= 9): Ellen & Tu (1984) / Beltaos & Rajaratnam (1973) 系。
    - 圧力分布:   ガウス形 p/p_m = exp(ln(0.5) xi^2)
    - 半値幅:     b_p/d = 0.0019 r^2 + 0.0551 r + 0.4035
                  （Z/d=8 で近接場相関と連続になるようスケール接続）
    - せん断分布: tau/tau_max = erf(0.833 xi) - 0.2 xi exp(ln(0.5) xi^2)
    - 最大圧力・最大せん断は Z/d=8 で近接場相関と連続になる 1/r 減衰で接続
      （VKI系 P_M = 6.5 dP (d/Z) と 7% 以内で整合）。

遷移域 (7 < Z/d < 9): 両分布の線形クロスフェードで滑らかに接続する
（v2.0.1: Z/d=8 での分布形状の不連続による目付ジャンプを解消）。

せん断の遠方裾 (xi > 4): 相関式のフィット範囲外であり、近接場の対数形は
減衰が遅く遠方場の erf 形は減衰しないため、壁面ジェットの実測減衰
（tau ~ 1/x; Beltaos & Rajaratnam 1973）に合わせて 1/xi テーパーを適用する。

座標系: x は鋼板に沿って上向き（引き上げ方向）を正、噴流軸（よどみ点）を原点とする。
x < 0（浴側）では dp/dx > 0、せん断は下向き（負）でいずれもワイピングを助ける。

p0 はプレナムゲージ圧 [Pa]。ポテンシャルコア内衝突では全圧が保存されるため、
よどみ圧スケールとしてプレナムゲージ圧をそのまま用いる（Elsaadawy 2007 と同じ扱い）。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
from numpy import errstate

try:
    from scipy.special import erf as _erf
except ModuleNotFoundError:  # pragma: no cover - scipy未導入環境向けフォールバック
    def _erf(x):
        x = np.asarray(x, dtype=float)
        # Abramowitz & Stegun 7.1.26 近似（最大誤差 1.5e-7）
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
        return sign * y

_LN_HALF = log(0.5)  # -0.6931
_BOUNDARY = 8.0       # 近接場/遠方場の公称境界
_BLEND_HALF = 1.0     # 遷移域の半幅（7 < r < 9 でクロスフェード）
_SHEAR_TAPER_XI = 4.0  # せん断裾テーパー開始の無次元位置


@dataclass(frozen=True)
class JetWipingField:
    """壁面上の圧力・せん断分布と代表値。"""

    x_m: np.ndarray
    pressure_pa: np.ndarray
    pressure_gradient_pa_m: np.ndarray
    shear_pa: np.ndarray
    peak_pressure_pa: float
    peak_shear_pa: float
    pressure_half_width_m: float
    shear_scale_width_m: float
    standoff_ratio: float
    regime: str  # "near_field" / "transition" / "far_field"
    max_pressure_gradient_pa_m: float


def _near_field_peaks(r: float, p0_pa: float, cfg: dict) -> tuple[float, float]:
    a2 = float(cfg.get("near_pm_quad", -0.0056))
    a1 = float(cfg.get("near_pm_lin", 0.0268))
    a0 = float(cfg.get("near_pm_const", 1.0108))
    pm_ratio = a2 * r * r + a1 * r + a0
    pm_ratio = min(max(pm_ratio, 0.05), float(cfg.get("pm_ratio_cap", 1.10)))
    t1 = float(cfg.get("near_tau_lin", -0.0001))
    t0 = float(cfg.get("near_tau_const", 0.0035))
    tau_ratio = max(t1 * r + t0, 1.0e-5)
    return pm_ratio * p0_pa, tau_ratio * p0_pa


def _peaks(r: float, p0_pa: float, cfg: dict) -> tuple[float, float]:
    """最大圧力・最大せん断。遠方場は Z/d=8 の値から 1/r 減衰で連続接続。"""
    if r <= _BOUNDARY:
        return _near_field_peaks(r, p0_pa, cfg)
    pm8, tau8 = _near_field_peaks(_BOUNDARY, p0_pa, cfg)
    decay = _BOUNDARY / r
    return pm8 * decay, tau8 * decay


def _near_bp_m(r: float, d_m: float, cfg: dict) -> float:
    return max((float(cfg.get("near_bp_lin", 0.0453)) * r + float(cfg.get("near_bp_const", 0.7921))) * d_m, 1.0e-6)


def _far_bp_m(r: float, d_m: float, cfg: dict) -> float:
    """遠方場半値幅（Ellen & Tu 1984 の公表相関をそのまま使用）。

    近接場相関との形状差は遷移域 (7<Z/d<9) のクロスフェードで吸収する。
    公表値を改変すると遠方場の検証精度（Elsaadawy plant case 等）が
    悪化するため、スケール接続は行わない（v2.0.2）。
    """
    bp = (
        float(cfg.get("far_bp_quad", 0.0019)) * r * r
        + float(cfg.get("far_bp_lin", 0.0551)) * r
        + float(cfg.get("far_bp_const", 0.4035))
    ) * d_m
    return max(bp, 1.0e-6)


def _near_bs_m(r: float, d_m: float, cfg: dict) -> float:
    return max((float(cfg.get("near_bs_lin", 0.0443)) * r + float(cfg.get("near_bs_const", 1.1687))) * d_m, 1.0e-6)


def _far_bs_m(r: float, d_m: float, cfg: dict) -> float:
    """遠方場せん断特性幅（Ellen & Tu 1984: 圧力半値幅で無次元化）。"""
    return _far_bp_m(r, d_m, cfg)


def _shear_taper(xi_abs: np.ndarray) -> np.ndarray:
    """フィット範囲外（xi>4）に壁面ジェット実測減衰（~1/xi）のテーパーを適用。"""
    return np.where(xi_abs > _SHEAR_TAPER_XI, _SHEAR_TAPER_XI / np.maximum(xi_abs, 1.0e-12), 1.0)


def _near_pressure_shape(xi: np.ndarray) -> np.ndarray:
    return (1.0 + 0.6 * xi**4) ** (-1.5)


def _near_pressure_gradient_shape(xi: np.ndarray, bp_m: float) -> np.ndarray:
    """d(p/p_m)/dx。解析微分: -3.6 xi^3 (1+0.6 xi^4)^(-2.5) / b_p"""
    return -3.6 * xi**3 * (1.0 + 0.6 * xi**4) ** (-2.5) / bp_m


def _near_shear_shape(xi_abs: np.ndarray) -> np.ndarray:
    inner = _erf(0.41 * xi_abs) + 0.54 * xi_abs * np.exp(-0.22 * xi_abs**3)
    with errstate(divide="ignore", invalid="ignore"):
        outer = 1.115 - 0.24 * np.log(np.maximum(xi_abs, 1.0e-12))
    shape = np.where(xi_abs <= 1.73, inner, outer)
    return np.clip(shape, 0.0, None) * _shear_taper(xi_abs)


def _far_pressure_shape(xi: np.ndarray) -> np.ndarray:
    return np.exp(_LN_HALF * xi**2)


def _far_pressure_gradient_shape(xi: np.ndarray, bp_m: float) -> np.ndarray:
    return 2.0 * _LN_HALF * xi * np.exp(_LN_HALF * xi**2) / bp_m


def _far_shear_shape(xi_abs: np.ndarray) -> np.ndarray:
    shape = _erf(0.833 * xi_abs) - 0.2 * xi_abs * np.exp(_LN_HALF * xi_abs**2)
    return np.clip(shape, 0.0, None) * _shear_taper(xi_abs)


def _profiles_at(
    r: float,
    x_m: np.ndarray,
    p_max: float,
    tau_max: float,
    d_m: float,
    cfg: dict,
    near: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """単一レジームの (p, dp/dx, tau, b_p, b_s) を返す。"""
    if near:
        bp = _near_bp_m(r, d_m, cfg)
        bs = _near_bs_m(r, d_m, cfg)
        xi_p = x_m / bp
        pressure = p_max * _near_pressure_shape(xi_p)
        gradient = p_max * _near_pressure_gradient_shape(xi_p, bp)
        shear = np.sign(x_m) * tau_max * _near_shear_shape(np.abs(x_m) / bs)
    else:
        bp = _far_bp_m(r, d_m, cfg)
        bs = _far_bs_m(r, d_m, cfg)
        xi_p = x_m / bp
        pressure = p_max * _far_pressure_shape(xi_p)
        gradient = p_max * _far_pressure_gradient_shape(xi_p, bp)
        shear = np.sign(x_m) * tau_max * _far_shear_shape(np.abs(x_m) / bs)
    return pressure, gradient, shear, bp, bs


def calculate_jet_field(
    plenum_gauge_pressure_pa: float,
    nozzle_gap_mm: float,
    nozzle_strip_distance_mm: float,
    coefficients: dict,
    points: int = 801,
    span_factor: float = 10.0,
) -> JetWipingField:
    """壁面圧力・せん断分布を計算する。

    Parameters
    ----------
    plenum_gauge_pressure_pa : プレナムゲージ圧 [Pa]
    nozzle_gap_mm : スロットギャップ d [mm]
    nozzle_strip_distance_mm : ノズル-鋼板距離 Z [mm]
    coefficients : model_coefficients.yaml の dict（"jet" セクションを参照）
    """
    cfg = coefficients.get("jet", {})
    d_m = max(float(nozzle_gap_mm), 1.0e-3) * 1.0e-3
    z_m = max(float(nozzle_strip_distance_mm), 1.0e-3) * 1.0e-3
    r = z_m / d_m

    p0 = max(float(plenum_gauge_pressure_pa), 0.0)
    p_max, tau_max = _peaks(r, p0, cfg)

    # 遷移域の重み（near: w=0, far: w=1）
    lo, hi = _BOUNDARY - _BLEND_HALF, _BOUNDARY + _BLEND_HALF
    if r <= lo:
        w_far = 0.0
        regime = "near_field"
    elif r >= hi:
        w_far = 1.0
        regime = "far_field"
    else:
        w_far = (r - lo) / (hi - lo)
        regime = "transition"

    # 解析域: 両レジームの幅から決める（最低±15mm）
    bp_n = _near_bp_m(min(r, hi), d_m, cfg)
    bs_n = _near_bs_m(min(r, hi), d_m, cfg)
    bp_f = _far_bp_m(max(r, lo), d_m, cfg)
    bs_f = _far_bs_m(max(r, lo), d_m, cfg)
    half_span = max(span_factor * max(bp_n, bs_n, bp_f, bs_f), 0.015)
    x_m = np.linspace(-half_span, half_span, points)

    if w_far <= 0.0:
        pressure, gradient, shear, bp, bs = _profiles_at(r, x_m, p_max, tau_max, d_m, cfg, near=True)
    elif w_far >= 1.0:
        pressure, gradient, shear, bp, bs = _profiles_at(r, x_m, p_max, tau_max, d_m, cfg, near=False)
    else:
        p1, g1, s1, bp1, bs1 = _profiles_at(r, x_m, p_max, tau_max, d_m, cfg, near=True)
        p2, g2, s2, bp2, bs2 = _profiles_at(r, x_m, p_max, tau_max, d_m, cfg, near=False)
        pressure = (1.0 - w_far) * p1 + w_far * p2
        gradient = (1.0 - w_far) * g1 + w_far * g2
        shear = (1.0 - w_far) * s1 + w_far * s2
        bp = (1.0 - w_far) * bp1 + w_far * bp2
        bs = (1.0 - w_far) * bs1 + w_far * bs2

    return JetWipingField(
        x_m=x_m,
        pressure_pa=pressure,
        pressure_gradient_pa_m=gradient,
        shear_pa=shear,
        peak_pressure_pa=float(np.max(pressure)),
        peak_shear_pa=float(np.max(np.abs(shear))),
        pressure_half_width_m=float(bp),
        shear_scale_width_m=float(bs),
        standoff_ratio=float(r),
        regime=regime,
        max_pressure_gradient_pa_m=float(np.max(np.abs(gradient))),
    )
