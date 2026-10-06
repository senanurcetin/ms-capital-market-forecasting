"""Do four extra members improve the LightGBM + XGBoost + Ridge ensemble?

`make ensemble-probe`. The members are in extra_members.py: a Huber-loss LightGBM, an
extra-randomised shallow LightGBM, CatBoost and an MLP. They were picked for how their errors might
differ from the base three, not for their own score.

DESIGN. The split of `make regime-clusters`, so the numbers are comparable with it:
    train  months 0-39     fit every member
    val    months 40-47    early stopping, and the blend weights - nothing else
    test   months 48-70    scored as one mixed pool, per month and pooled
Every member's validation and test predictions are written to disk, so a restart costs one fit.

THE COMPARISON. Base = NNLS blend of lightgbm, xgboost, ridge. Extended = NNLS blend of all seven.
Both have their weights fitted on the validation months only, by the same procedure, so the only
difference is the four extra members. The paired per-month gain Extended - Base is read over the 23
test months with a t-interval. (Adding each extra member to the base on its own is reported too,
as an exploration: four more comparisons, none of them the question.)

DECISION RULE, written before the run.
  adds signal      iff the 95% t-interval of mean(Extended - Base) lies above zero.
  worth shipping   iff, in addition, the mean gain is at least 0.0010 - the size of the gain the README
                   measured for building the ensemble in the first place, and so the smallest one
                   that has been judged worth the extra moving parts. Shipping also means four more
                   artefacts to load, two more libraries in the serving image and more memory per
                   request, which a gain of a fraction of that does not repay.
FORECAST, written before the run: Extended - Base between -0.0005 and +0.0020, interval including
zero; the MLP is the member most likely to get a non-zero weight. (The README's record on this
problem is five forecasts, five overshoots, all optimistic.)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.evaluation.regime_clusters import (
    TEST_MONTHS,
    TRAIN_MONTHS,
    VAL_MONTHS,
    _in,
    monthly_cosine,
)
from src.evaluation.regime_clusters import paired_verdict as _paired
from src.models.base import Model, feature_columns
from src.models.ensemble import CosineOptimalEnsemble

log = logging.getLogger(__name__)

BASE = ("lightgbm", "xgboost", "ridge")
EXTRA = ("lightgbm_huber", "lightgbm_extra", "catboost", "mlp")
SHIP_GAIN = 0.0010


def make_member(name: str) -> Model:
    from src.models.baseline import RidgeModel
    from src.models.extra_members import CatBoostModel, ExtraTreesLightGBM, HuberLightGBM, MLPModel
    from src.models.lightgbm_model import LightGBMModel
    from src.models.xgboost_model import XGBoostModel

    factories: dict[str, Callable[[], Model]] = {
        "lightgbm": lambda: LightGBMModel(num_boost_round=2000, early_stopping_rounds=100),
        "xgboost": lambda: XGBoostModel(num_boost_round=2000, early_stopping_rounds=100),
        "ridge": lambda: RidgeModel(alpha=10.0),
        "lightgbm_huber": lambda: HuberLightGBM(num_boost_round=2000, early_stopping_rounds=100),
        "lightgbm_extra": lambda: ExtraTreesLightGBM(num_boost_round=3000,
                                                     early_stopping_rounds=100),
        "catboost": lambda: CatBoostModel(),
        "mlp": lambda: MLPModel(),
    }
    return factories[name]()


def blend(names: tuple[str, ...], val: dict[str, np.ndarray], test: dict[str, np.ndarray],
          y_val: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """NNLS weights fitted on the validation predictions, applied to the test predictions."""
    ens = CosineOptimalEnsemble(list(names)).fit(np.column_stack([val[n] for n in names]), y_val)
    return ens.predict(np.column_stack([test[n] for n in names])), ens.weight_map()


def verdict(diffs: np.ndarray) -> dict:
    """The regime-clusters interval, with this experiment's own shipping threshold."""
    v = _paired(diffs)
    v.pop("worth_keeping", None)
    v.pop("material_gain_threshold", None)
    v["worth_shipping"] = bool(v["adds_signal"] and v["mean_gain"] >= SHIP_GAIN)
    v["ship_gain_threshold"] = SHIP_GAIN
    v["verdict"] = ("the extra members improve the ensemble" if v["adds_signal"] else
                    "not shown: the interval includes zero, so a gain of this size cannot be told "
                    "from none")
    return v


def run(*, members: tuple[str, ...] = BASE + EXTRA, out_dir: Path | None = None) -> dict:
    cfg = load_config()
    df = pd.read_parquet(Path(cfg.paths.features) / "dataset_train.parquet")
    months, y = df["month"].to_numpy(), df["target"].to_numpy(dtype=np.float64)
    tr, va, te = (np.flatnonzero(_in(months, s)) for s in (TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS))
    base_cols = feature_columns(df)
    log.info("rows: train %s, val %s, test %s", f"{len(tr):,}", f"{len(va):,}", f"{len(te):,}")

    cache = Path(cfg.paths.features) / "ensemble_cache"
    cache.mkdir(parents=True, exist_ok=True)
    val: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    for name in members:
        kept = cache / f"{name}.npz"
        if kept.exists():
            z = np.load(kept)
            val[name], test[name], seconds[name] = z["val"], z["test"], float(z["seconds"])
            log.info("%-15s reused from %s", name, kept.name)
            continue
        t0 = time.perf_counter()
        model = make_member(name)
        model.fit(df.iloc[tr][base_cols], y[tr], eval_set=(df.iloc[va][base_cols], y[va]))
        val[name] = np.asarray(model.predict(df.iloc[va][base_cols]), dtype=np.float64)
        test[name] = np.asarray(model.predict(df.iloc[te][base_cols]), dtype=np.float64)
        seconds[name] = time.perf_counter() - t0
        np.savez(kept, val=val[name], test=test[name], seconds=seconds[name])
        log.info("%-15s val %+.5f  test %+.5f  (%.0fs)", name,
                 cosine_similarity(y[va], val[name]), cosine_similarity(y[te], test[name]),
                 seconds[name])
        del model

    m_te = months[te]
    y_va, y_te = y[va], y[te]
    result: dict[str, Any] = {
        "split": {"train": TRAIN_MONTHS, "val": VAL_MONTHS, "test": TEST_MONTHS,
                  "n_train": len(tr), "n_val": len(va), "n_test": len(te)},
        "members": list(members), "base": list(BASE), "extra": [m for m in members if m in EXTRA],
        "fit_seconds": seconds,
        "alone": {n: {"val": cosine_similarity(y_va, val[n]),
                      "test": cosine_similarity(y_te, test[n])} for n in members},
        "prediction_correlation_test": pd.DataFrame(test).corr().round(4).to_dict(),
    }
    base_pred, base_w = blend(BASE, val, test, y_va)
    ext_names = tuple(members)
    ext_pred, ext_w = blend(ext_names, val, test, y_va)
    per_base, per_ext = monthly_cosine(y_te, base_pred, m_te), monthly_cosine(y_te, ext_pred, m_te)
    diffs = np.array([per_ext[m] - per_base[m] for m in sorted(per_base)])
    result.update({
        "pooled_test_cosine": {"base": cosine_similarity(y_te, base_pred),
                               "extended": cosine_similarity(y_te, ext_pred)},
        "weights": {"base": base_w, "extended": ext_w},
        "per_month_test": {"base": per_base, "extended": per_ext},
        "paired_extended_vs_base": verdict(diffs),
    })
    adds = {}
    for extra in (m for m in members if m in EXTRA):
        p, w = blend((*BASE, extra), val, test, y_va)
        d = np.array([monthly_cosine(y_te, p, m_te)[m] - per_base[m] for m in sorted(per_base)])
        adds[extra] = {"pooled_test_cosine": cosine_similarity(y_te, p), "weights": w,
                       "mean_gain_over_base": float(d.mean()), "months_improved": int((d > 0).sum())}
    result["each_extra_added_to_base"] = adds

    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ensemble_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"month": sorted(per_base), "base": [per_base[m] for m in sorted(per_base)],
                  "extended": [per_ext[m] for m in sorted(per_base)]}
                 ).to_csv(out_dir / "ensemble_probe_months.csv", index=False, float_format="%.6f")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--members", default=",".join(BASE + EXTRA))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    r = run(members=tuple(args.members.split(",")), out_dir=Path(args.out))
    print(json.dumps({k: r[k] for k in ("alone", "pooled_test_cosine", "weights",
                                        "paired_extended_vs_base")}, indent=2))


if __name__ == "__main__":
    main()
