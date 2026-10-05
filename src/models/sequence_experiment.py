"""Does a learned sequence representation know something the tabular model does not?

`make sequence` (needs PyTorch, which is deliberately not a dependency of CI or the serving
image: `pip install torch`). See sequence_cnn.py for the model and the reason for the test.

DESIGN. One split, fixed before anything was run, over a 1-in-3 subsample of the samples of
months 40-70 (the subsample keeps the raw-table download to ~32M rows):

    train  months 40-59   fit both models
    val    months 60-62   early stopping, and the blend weights - nothing else
    test   months 63-70   never touched until the end; eight months, scored one by one

A: LightGBM on the 292 features.   B: the CNN on the 176 x 16 snapshot tensor.
C: a cosine-optimal (NNLS) blend of A and B, weights fitted on val.

The only question is whether C beats A on the test months, as a paired difference per month.
Comparing B with A would answer the wrong thing: a weaker model can still carry information
the stronger one lacks, and a stronger one may carry none that is new.

DECISION RULE, written before the run.
  adds signal   iff the 95% CI of mean(C - A) over the eight test months lies above zero.
  worth shipping iff, in addition, the mean gain reaches 0.0041, the fold-to-fold std the README
                 treats as this problem's resolution limit.
With eight months the interval is wide (t, 7 d.o.f.), so a "no" here means "not shown", and the
result file says so rather than reading a point estimate as a finding.

FORECAST, written before the run (the README table of overshoots is why it is written down):
  A is about 0.12-0.14 (it sees 20 months and a third of the samples, so a little below the
  shipped model); B is 0.03-0.09; mean(C - A) is between -0.001 and +0.003, and the interval
  includes zero.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.models import sequence_cnn as sq
from src.models.base import feature_columns
from src.models.ensemble import CosineOptimalEnsemble

log = logging.getLogger(__name__)

TRAIN_MONTHS = (40, 59)
VAL_MONTHS = (60, 62)
TEST_MONTHS = (63, 70)
MATERIAL_GAIN = 0.0041


def _in(months: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    return (months >= span[0]) & (months <= span[1])


def monthly_cosine(y: np.ndarray, p: np.ndarray, months: np.ndarray) -> dict[int, float]:
    """Cosine within each month - the unit the paired comparison is made in."""
    return {int(m): cosine_similarity(y[months == m], p[months == m]) for m in np.unique(months)}


def paired_verdict(diffs: np.ndarray) -> dict:
    """Paired per-month gain of the blend over the tabular model, with a t-interval.

    t rather than the normal 1.96 that feature_prune uses: here n is 8, not 6 folds x seeds,
    and the difference between the two quantiles (2.36 vs 1.96) is not small at this size.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    gain = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * se
    lo, hi = gain - half, gain + half
    adds = lo > 0
    return {
        "mean_gain": gain, "se": se, "ci_low": lo, "ci_high": hi, "n_months": n,
        "months_improved": int((d > 0).sum()),
        "adds_signal": bool(adds), "worth_shipping": bool(adds and gain >= MATERIAL_GAIN),
        "material_gain_threshold": MATERIAL_GAIN,
        "verdict": ("the sequence model adds signal the tabular model lacks" if adds else
                    "not shown: the interval includes zero, so a gain of this size cannot be "
                    "told from none at eight months"),
    }


def run(*, epochs: int = 12, seeds: tuple[int, ...] = (0,), out_dir: Path | None = None) -> dict:
    import lightgbm  # noqa: F401  (fail early, before the long steps)

    from src.models.lightgbm_model import LightGBMModel

    cfg = load_config()
    root = Path(cfg.paths.data_root) / "seq"
    ids = np.load(root / "ids.npy")
    X = np.load(root / "X.npy", mmap_mode="r")
    tab = pd.read_parquet(Path(cfg.paths.features) / "dataset_train.parquet")
    tab = tab.set_index("sample_id", drop=False).loc[ids].reset_index(drop=True)
    assert (tab["sample_id"].to_numpy() == ids).all()
    months, y = tab["month"].to_numpy(), tab["target"].to_numpy(dtype=np.float64)
    tr, va, te = (np.flatnonzero(_in(months, s)) for s in (TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS))
    log.info("samples: train %s, val %s, test %s", f"{len(tr):,}", f"{len(va):,}", f"{len(te):,}")

    # ---- A: the tabular model ---------------------------------------------------------
    t0 = time.perf_counter()
    feats = feature_columns(tab)
    lgbm = LightGBMModel(num_boost_round=2000, early_stopping_rounds=100)
    lgbm.fit(tab.iloc[tr][feats], y[tr], eval_set=(tab.iloc[va][feats], y[va]))
    a_va, a_te = lgbm.predict(tab.iloc[va][feats]), lgbm.predict(tab.iloc[te][feats])
    log.info("A (LightGBM) val %+.5f  test %+.5f  (%.0fs)", cosine_similarity(y[va], a_va),
             cosine_similarity(y[te], a_te), time.perf_counter() - t0)
    del tab

    # ---- B: the sequence model, averaged over seeds -----------------------------------
    mu, sd = sq.fit_scaler(np.asarray(X[tr]))
    Xtr, Xva, Xte = (sq.apply_scaler(np.asarray(X[i]), mu, sd) for i in (tr, va, te))
    b_va, b_te, histories = [], [], []
    for seed in seeds:
        t0 = time.perf_counter()
        net, hist = sq.train(Xtr, y[tr], Xva, y[va], epochs=epochs, seed=seed, log=log.info)
        b_va.append(sq.predict(net, Xva))
        b_te.append(sq.predict(net, Xte))
        histories.append(hist)
        log.info("B seed %d val %+.5f test %+.5f  (%.0fs)", seed,
                 cosine_similarity(y[va], b_va[-1]), cosine_similarity(y[te], b_te[-1]),
                 time.perf_counter() - t0)
    b_va_m, b_te_m = np.mean(b_va, axis=0), np.mean(b_te, axis=0)

    # ---- C: the blend, weights from val only ------------------------------------------
    ens = CosineOptimalEnsemble(["tabular", "sequence"]).fit(np.column_stack([a_va, b_va_m]), y[va])
    c_te = ens.predict(np.column_stack([a_te, b_te_m]))

    m_te = months[te]
    per_a, per_b, per_c = (monthly_cosine(y[te], p, m_te) for p in (a_te, b_te_m, c_te))
    diffs = np.array([per_c[m] - per_a[m] for m in sorted(per_a)])
    result = {
        "split": {"train": TRAIN_MONTHS, "val": VAL_MONTHS, "test": TEST_MONTHS,
                  "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                  "subsample": "sample_id % 3 == 0"},
        "seeds": list(seeds), "epochs": epochs,
        "test_cosine": {"tabular": cosine_similarity(y[te], a_te),
                        "sequence": cosine_similarity(y[te], b_te_m),
                        "blend": cosine_similarity(y[te], c_te)},
        "val_cosine": {"tabular": cosine_similarity(y[va], a_va),
                       "sequence": cosine_similarity(y[va], b_va_m)},
        "blend_weights": ens.weight_map(),
        "prediction_correlation_test": float(np.corrcoef(a_te, b_te_m)[0, 1]),
        "per_month_test": {"tabular": per_a, "sequence": per_b, "blend": per_c},
        "paired": paired_verdict(diffs),
        "training_curves": histories,
    }
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sequence_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = [{"month": m, "tabular": per_a[m], "sequence": per_b[m], "blend": per_c[m],
             "blend_minus_tabular": per_c[m] - per_a[m]} for m in sorted(per_a)]
    pd.DataFrame(rows).to_csv(out_dir / "sequence_probe_months.csv", index=False,
                              float_format="%.6f")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    r = run(epochs=args.epochs, seeds=seeds, out_dir=Path(args.out))
    print(json.dumps({k: r[k] for k in ("test_cosine", "blend_weights", "paired")}, indent=2))


if __name__ == "__main__":
    main()
