"""亜鉛融体の温度依存物性。

Andrade-Arrhenius 型の温度補正式を用いて、浴温に応じた粘度を算出する。
density / surface_tension は温度依存が弱いため、初期版では一定値とする。
"""
from __future__ import annotations

from math import exp

from gaswipelab.models.units import c_to_k


def zinc_viscosity_pa_s(bath_temp_c: float, material_config: dict) -> float:
    """浴温に応じた亜鉛融体の動的粘度 [Pa·s] を返す。

    旧バージョンの設定ファイル（viscosity_ref_pa_s 等の温度依存パラメータ未定義）でも
    動作するように、欠落時は固定値（viscosity_pa_s）にフォールバックする。
    """
    zinc = material_config["zinc"]
    if "viscosity_ref_pa_s" not in zinc or "viscosity_ref_temp_c" not in zinc or "viscosity_e_over_r_k" not in zinc:
        return float(zinc["viscosity_pa_s"])
    mu_ref = float(zinc["viscosity_ref_pa_s"])
    t_ref_k = c_to_k(float(zinc["viscosity_ref_temp_c"]))
    e_over_r = float(zinc["viscosity_e_over_r_k"])
    t_k = c_to_k(bath_temp_c)
    # 物理的に妥当な温度範囲外でも数値破綻しないよう保護。
    if t_k <= 0.0:
        return mu_ref
    return mu_ref * exp(e_over_r * (1.0 / t_k - 1.0 / t_ref_k))
