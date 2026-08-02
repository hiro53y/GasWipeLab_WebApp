"""実機の操業実績データ由来モデルの推論。`inference_reference.py` と同じ契約を実装する。

GI: 直接回帰（U:めっき付着量記号をカテゴリ特徴に含む）
GL: 付着量記号別中央値（baseline）＋ CatBoost 残差
主予測は CH_sum = CF + CG。CH_direct は診断用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaswipelab.ml.catboost_runtime import CompactModel, to_category, to_number
from gaswipelab.ml.cityhash import cat_feature_hash

MODEL_VERSION = "v2-2026-07-28"
MODEL_KEYS = ("GI_CF", "GI_CG", "GI_CH", "GL_CF", "GL_CG", "GL_CH")
COATING_CODE_FEATURE = "U:めっき付着量記号"

#: ウォーマーコイル。製品ではないため学習・予測の対象外。
EXCLUDED_COATING_CODE = "305"

# ----------------------------------------------------------------------
# 誤差の参考値
#
# 2種類あり、意味がまったく違う。画面に出すのは必ず「時系列検証」のほう。
#
#   report   : 引継ぎパッケージの報告値（2026-06 を評価に使用）。
#              ただし最終モデルの学習には 2026-06 も含まれているため、
#              同じ期間で測った値であり、汎化性能としては楽観側に出る。
#   rolling  : 月ごとに「その月より前だけで学習してその月を予測」した
#              ローリング検証（Mar/Apr/May の3フォールド）。
#              未知のコイルに対して期待できる誤差はこちら。
#
# 実測（tools/evaluate_shipped_model.py）では、配布モデルを学習期間全体に
# かけると GI 5.07 / GL 1.85 g/m² まで下がる。これは標本内の値であり、
# 報告値がさらにその外側、ローリング検証がもっとも保守的という関係にある。
# ----------------------------------------------------------------------

#: 画面表示・判定に使う誤差。ローリング検証（時系列）の値。
ERROR_REFERENCE: dict[str, dict[str, float]] = {
    "GI": {
        "CF_MAE": 3.732, "CG_MAE": 3.529,
        "CH_MAE": 6.715,          # 3フォールド平均
        "CH_MAE_worst": 6.916,    # 最悪フォールド
        "CH_P90": 15.026,         # |誤差| の90パーセンタイル（3フォールド平均）
        "CH_within10": 0.809,
        "basis": "rolling",
    },
    "GL": {
        "CF_MAE": 1.611, "CG_MAE": 1.683,
        "CH_MAE": 3.198,
        "CH_MAE_worst": 4.188,
        "CH_P90": 7.031,
        "CH_within10": 0.943,
        "basis": "rolling",
    },
}

#: 引継ぎ報告値（学習期間と重なる評価）。比較のために残す。画面の主表示には使わない。
ERROR_REFERENCE_REPORTED: dict[str, dict[str, float]] = {
    "GI": {"CH_MAE": 6.621, "CH_P90": 13.200, "CH_within10": 0.820},
    "GL": {"CH_MAE": 2.544, "CH_P90": 5.448, "CH_within10": 0.954},
}

#: GL の「付着量記号別中央値だけ」を使った場合の誤差。
#: CatBoost 残差モデルがどれだけ上乗せしているかを示すために保持する。
#: 標本内（学習期間全体）: 中央値のみ 2.707 → 配布モデル 1.850（tools/evaluate_shipped_model.py）
#: 標本外（2026-06 ホールドアウト・引継ぎ側の比較）: 中央値のみ 2.735 → 残差あり 2.890
#: すなわち、標本外では残差モデルの優位が確認できていない。
GL_BASELINE_ONLY_MAE = {"in_sample": 2.707, "holdout": 2.735}
GL_WITH_RESIDUAL_MAE = {"in_sample": 1.850, "holdout": 2.890}


class OutOfScopeError(ValueError):
    """付着量記号305など、モデルの適用対象外。"""


class GasWipingPredictor:
    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.model_version: str = manifest.get("model_version", MODEL_VERSION)
        self.meta: dict[str, Any] = manifest["models"]
        self.baselines: dict[str, Any] = manifest["baselines"]
        self._models: dict[str, CompactModel] = {}
        # 特徴量順序は変換時にコピーしたものをそのまま使う（並べ替え禁止）
        self._cat_positions: dict[str, list[int]] = {}
        for key, entry in self.meta.items():
            cats = set(entry["categorical_feature_indices"])
            self._cat_positions[key] = sorted(cats)

    def _model(self, key: str) -> CompactModel:
        model = self._models.get(key)
        if model is None:
            payload = json.loads((self.dir / self.meta[key]["file"]).read_text(encoding="utf-8"))
            model = CompactModel(payload)
            self._models[key] = model
        return model

    def coating_codes(self, line: str) -> list[str]:
        """GL の baseline に載っている記号一覧（GI は baseline を持たない）。"""
        key = f"{line}_CH"
        entry = self.baselines.get(key)
        return sorted(entry["by_code"]) if entry else []

    # ------------------------------------------------------------------
    # 学習範囲の外側を「案分」で求める
    #
    # 勾配ブースティング木は学習データの外側では応答が頭打ちになり、
    # いくら値を動かしても予測が変わらなくなる。設備仕様としては選べる範囲まで
    # 計算できるようにするため、境界での傾きをそのまま直線で延長する。
    # これは物理的な裏づけのある外挿ではないので、必ず範囲外として警告する。
    # ------------------------------------------------------------------
    def _clamp(self, record: dict[str, Any],
               ranges: dict[str, tuple[float, float]]) -> tuple[dict[str, Any], list[tuple]]:
        clamped = dict(record)
        outside: list[tuple] = []
        for feature, (low, high) in ranges.items():
            raw = to_number(record.get(feature))
            if raw != raw:  # NaN
                continue
            if raw < low:
                clamped[feature] = low
                outside.append((feature, low, raw - low, high - low))
            elif raw > high:
                clamped[feature] = high
                outside.append((feature, high, raw - high, high - low))
        return clamped, outside

    def _predict_extrapolated(self, record: dict[str, Any], key: str,
                              ranges: dict[str, tuple[float, float]] | None,
                              trend_signs: dict[str, float] | None = None,
                              ) -> tuple[float, list[dict[str, Any]]]:
        if not ranges:
            return self._predict_one(record, key), []
        clamped, outside = self._clamp(record, ranges)
        base = self._predict_one(clamped, key)
        if not outside:
            return base, []
        signs = trend_signs or {}
        total = base
        detail: list[dict[str, Any]] = []
        for feature, boundary, delta, span in outside:
            # 境界の内側との差から傾きを求め、はみ出した分だけ直線で延ばす。
            # 木モデルは学習範囲の端で応答が平らになりやすいので、傾きが0のときは
            # 内側へ窓を広げて、実際に効いている区間の傾きを拾う。
            slope = 0.0
            for fraction in (0.05, 0.20, 0.40):
                step = max(abs(span) * fraction, 1.0e-6)
                inward = boundary - step if delta > 0 else boundary + step
                probe = dict(clamped)
                probe[feature] = inward
                denominator = boundary - inward
                if not denominator:
                    continue
                slope = (base - self._predict_one(probe, key)) / denominator
                if abs(slope) > 1.0e-9:
                    break
            # 学習範囲の端は事例が乏しく、傾きが実データと逆向きに出ることがある。
            # 実データの偏効果と符号が合わないときは延長せず、境界の値で止める。
            # （物理的にありえない向きの推奨を出さないため）
            expected = signs.get(feature, 0.0)
            saturated = bool(expected) and slope * expected < 0.0
            if saturated:
                slope = 0.0
            contribution = slope * delta
            total += contribution
            detail.append({
                "feature": feature,
                "boundary": round(boundary, 4),
                "excess": round(delta, 4),
                "slope_per_unit": round(slope, 6),
                "contribution_gm2": round(contribution, 3),
                "saturated": saturated,
            })
        return total, detail

    def _predict_one(self, record: dict[str, Any], key: str) -> float:
        entry = self.meta[key]
        cat_indices = set(entry["categorical_feature_indices"])
        floats: list[float] = []
        cats: list[int] = []
        for index, name in enumerate(entry["feature_names"]):
            if name == COATING_CODE_FEATURE:
                value = record.get(name, record.get("coating_code"))
            else:
                value = record.get(name)
            if index in cat_indices:
                cats.append(cat_feature_hash(to_category(value)))
            else:
                floats.append(to_number(value))
        raw = self._model(key).predict(floats, cats)
        if entry["line"] == "GL":
            baseline = self.baselines[key]
            code = to_category(record.get("coating_code"))
            raw += float(baseline["by_code"].get(code, baseline["global"]))
        return raw

    def predict(self, record: dict[str, Any],
                ranges: dict[str, tuple[float, float]] | None = None,
                trend_signs: dict[str, float] | None = None) -> dict[str, Any]:
        """1条件の予測。

        `ranges` を渡すと、その範囲を外れた入力について境界の傾きで外挿する。
        省略時（既定）は従来どおりの素の推論で、結果は完全に同一になる。
        """
        line = to_category(record.get("line"))
        code = to_category(record.get("coating_code"))
        if line not in ("GI", "GL"):
            raise ValueError("ライン区分は GI または GL を指定してください。")
        if code == EXCLUDED_COATING_CODE:
            raise OutOfScopeError(
                "付着量記号305はラインスタート時のウォーマーコイルで製品ではないため、予測対象外です。"
            )
        cf, cf_detail = self._predict_extrapolated(record, f"{line}_CF", ranges, trend_signs)
        cg, cg_detail = self._predict_extrapolated(record, f"{line}_CG", ranges, trend_signs)
        ch_direct, _ = self._predict_extrapolated(record, f"{line}_CH", ranges, trend_signs)
        ch_sum = cf + cg
        details = cf_detail + cg_detail
        extrapolated = sorted({d["feature"] for d in details})
        saturated = sorted({d["feature"] for d in details if d.get("saturated")})
        return {
            "model_version": self.model_version,
            "line": line,
            "coating_code": code,
            "CF_pred_g_m2": cf,
            "CG_pred_g_m2": cg,
            "CH_sum_pred_g_m2": ch_sum,
            "CH_direct_pred_g_m2": ch_direct,
            "CH_direct_minus_sum_g_m2": ch_direct - ch_sum,
            "error_reference": ERROR_REFERENCE[line],
            "extrapolated_features": extrapolated,
            "saturated_features": saturated,
            "extrapolation_detail": details,
        }

    def predict_ch(self, record: dict[str, Any],
                   ranges: dict[str, tuple[float, float]] | None = None,
                   trend_signs: dict[str, float] | None = None) -> float:
        """逆算ループ用。両面合計のみを返す（CH_direct を計算しないぶん速い）。"""
        line = to_category(record.get("line"))
        cf, _ = self._predict_extrapolated(record, f"{line}_CF", ranges, trend_signs)
        cg, _ = self._predict_extrapolated(record, f"{line}_CG", ranges, trend_signs)
        return cf + cg
