"""実機プロコンデータ由来モデルの推論。`inference_reference.py` と同じ契約を実装する。

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

#: ホールドアウト（2026-06以降）実測値。予測値に併記する誤差参考値。
ERROR_REFERENCE: dict[str, dict[str, float]] = {
    "GI": {"CF_MAE": 3.732, "CG_MAE": 3.529, "CH_MAE": 6.621, "CH_P90": 13.200, "CH_within10": 0.820},
    "GL": {"CF_MAE": 1.611, "CG_MAE": 1.683, "CH_MAE": 2.544, "CH_P90": 5.448, "CH_within10": 0.954},
}


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

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        line = to_category(record.get("line"))
        code = to_category(record.get("coating_code"))
        if line not in ("GI", "GL"):
            raise ValueError("ライン区分は GI または GL を指定してください。")
        if code == EXCLUDED_COATING_CODE:
            raise OutOfScopeError(
                "付着量記号305はラインスタート時のウォーマーコイルで製品ではないため、予測対象外です。"
            )
        cf = self._predict_one(record, f"{line}_CF")
        cg = self._predict_one(record, f"{line}_CG")
        ch_direct = self._predict_one(record, f"{line}_CH")
        ch_sum = cf + cg
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
        }

    def predict_ch(self, record: dict[str, Any]) -> float:
        """逆算ループ用。両面合計のみを返す（CH_direct を計算しないぶん速い）。"""
        line = to_category(record.get("line"))
        return self._predict_one(record, f"{line}_CF") + self._predict_one(record, f"{line}_CG")
