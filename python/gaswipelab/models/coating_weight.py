from __future__ import annotations


def one_side_coating_weight_g_m2(film_thickness_m: float, density_kg_m3: float) -> float:
    return density_kg_m3 * film_thickness_m * 1000.0


def both_sides_coating_weight_g_m2(film_thickness_m: float, density_kg_m3: float) -> float:
    return 2.0 * one_side_coating_weight_g_m2(film_thickness_m, density_kg_m3)

