"""Degerlendirme metrikleri.

ANA METRIK: cosine similarity.
    cos(y, yhat) = sum(y*yhat) / (||y|| * ||yhat||)

Onemli ozellikleri (modelleme kararlarini dogrudan etkiler):
  * OLCEK-DEGISMEZ: yhat'i pozitif bir sabitle carpmak skoru DEGISTIRMEZ
    -> tahmin buyuklugu kalibrasyonuna efor harcamak anlamsiz.
  * KAYDIRMA-DEGISMEZ DEGIL: yhat'a sabit bias eklemek skoru BOZAR.
    (Pearson korelasyonundan farki budur; Pearson once ortalamayi cikarir.)
    -> tahminler sifir etrafinda tutulmali.
  * MSE/Huber egitimi mesru bir vekildir ama model SECIMI cosine ile yapilir.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _as_1d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64).ravel()
    return arr


def cosine_similarity(y_true, y_pred) -> float:
    """Yarismanin ana metrigi. Iki vektor de sifirsa 0 doner."""
    y, p = _as_1d(y_true), _as_1d(y_pred)
    if y.shape != p.shape:
        raise ValueError(f"sekil uyusmuyor: {y.shape} vs {p.shape}")
    denom = np.linalg.norm(y) * np.linalg.norm(p)
    if denom < _EPS:
        return 0.0
    return float(np.dot(y, p) / denom)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_as_1d(y_true) - _as_1d(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((_as_1d(y_true) - _as_1d(y_pred)) ** 2)))


def pearson(y_true, y_pred) -> float:
    y, p = _as_1d(y_true), _as_1d(y_pred)
    if np.std(y) < _EPS or np.std(p) < _EPS:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def directional_accuracy(y_true, y_pred, *, ignore_zeros: bool = True) -> float:
    """Isaret dogrulugu. Target'in %5.5'i TAM SIFIR oldugu icin varsayilan
    olarak bu satirlar disarida birakilir (aksi halde metrik yaniltici olur)."""
    y, p = _as_1d(y_true), _as_1d(y_pred)
    mask = y != 0 if ignore_zeros else np.ones_like(y, dtype=bool)
    if not mask.any():
        return 0.0
    return float(np.mean(np.sign(y[mask]) == np.sign(p[mask])))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Tum metrikleri tek seferde dondurur."""
    return {
        "cosine": cosine_similarity(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def lgb_cosine_eval(y_pred, dataset):
    """LightGBM custom eval: erken durdurma cosine uzerinden yapilir."""
    y_true = dataset.get_label()
    return "cosine", cosine_similarity(y_true, y_pred), True  # True = buyuk daha iyi


def xgb_cosine_eval(y_pred, dmatrix):
    """XGBoost custom eval (feval). XGBoost minimize ettigi icin negatifi doner."""
    return "neg_cosine", -cosine_similarity(dmatrix.get_label(), y_pred)
