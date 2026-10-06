"""Can a regime estimated from the feature space, not from sample order, help the model?

WHY THIS FORM. `make test-order` showed that `sample_id` carries no order in the test set, so every
neighbour-based notion of "what the market is doing around this sample" is unavailable there. What
the test set does have is its own features, all 647,896 of them, a pool about 37 months' worth.
So the regime has to be a function of a pool's features alone.

THE DESCRIPTORS. A k-means over a handful of market-state signals, fitted on a pool's own features
(no labels, no order). Each sample is described by the *centre of its cluster in the original
units*, the cluster's share of the pool, and its own distance to the centre. Cluster ids are never
used: ids from two separate fits are not comparable, whereas a centre in volatility, spread, depth
and intensity units means the same thing in a training pool and in the test pool.

TWO VARIANTS, because they ask different questions.
  static        fitted once on the training pool and applied to the others. Nothing is learned from
                the pool being scored, so any gain is just a nonlinear function of a sample's own
                features, which a gradient-boosted tree could have built itself.
  transductive  fitted on each pool's own features, as the test pool would be. It sees how the pool
                is composed, which a single sample cannot tell the model. This is the one the
                question is about.
If transductive beats static, the pool's composition carries something. If neither beats the plain
292 features, the model already had what a cluster could add.

SPLIT, chosen to look like the real thing: the test pool is a mix of many months with no month label,
so the scored pool here is 23 months mixed, not one month at a time.
    train  months 0-39     fit the model, and the training pool's clusters
    val    months 40-47    early stopping only (its own pool, for the descriptors)
    test   months 48-70    scored, per month and pooled; descriptors from this pool alone

DECISION RULE, written before the run.
  adds signal   iff the 95% t-interval of the paired per-month gain over the plain 292 features, for
                the TRANSDUCTIVE variant, lies above zero (23 months, so 22 d.o.f.).
  worth keeping iff, in addition, the mean gain reaches 0.0041, the fold-to-fold std the README
                treats as this problem's resolution limit.
FORECAST, written before the run: transductive minus plain between -0.002 and +0.002, interval
including zero; static no different from plain. (The README's record is five forecasts, five
overshoots, all optimistic.)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.models.base import feature_columns

log = logging.getLogger(__name__)

# Chosen by what they measure (volatility, spread, depth, trading and order intensity) before any
# result was seen, not by importance: importance is computed from the target.
STATE_FEATURES = ("mkt_mid_std_60s", "mkt_mid_std_300s", "mkt_rel_spread_mean_60s",
                  "mkt_total_depth_mean_60s", "txn_intensity_60s", "ord_n_total")
N_CLUSTERS = 8
LOG_FLOOR = 1e-6  # mid_std is exactly 0 for 29% of samples (a flat price), and log(0) is undefined
TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS = (0, 39), (40, 47), (48, 70)
MATERIAL_GAIN = 0.0041


def descriptor_names() -> list[str]:
    return [f"regime_centre_{c}" for c in STATE_FEATURES] + ["regime_share", "regime_dist"]


class RegimeDescriber:
    """K-means over log market-state signals, reported in original units. Uses features only."""

    def __init__(self, n_clusters: int = N_CLUSTERS, seed: int = 0) -> None:
        self.n_clusters, self.seed = n_clusters, seed
        self.km: Any = None
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.fill_: np.ndarray | None = None
        self.share_: np.ndarray | None = None

    def _log(self, df: pd.DataFrame) -> np.ndarray:
        return np.log(df[list(STATE_FEATURES)].to_numpy(dtype=np.float64) + LOG_FLOOR)

    def fit(self, pool: pd.DataFrame) -> RegimeDescriber:
        from sklearn.cluster import MiniBatchKMeans

        x = self._log(pool)
        self.fill_ = np.nanmedian(x, axis=0)  # a NaN signal is filled by the pool's own median
        x = np.where(np.isnan(x), self.fill_, x)
        self.mu_, self.sd_ = x.mean(0), x.std(0) + 1e-12
        self.km = MiniBatchKMeans(self.n_clusters, random_state=self.seed, n_init=3,
                                  batch_size=8192).fit((x - self.mu_) / self.sd_)
        labels = self.km.predict((x - self.mu_) / self.sd_)
        self.share_ = np.bincount(labels, minlength=self.n_clusters) / len(labels)
        return self

    def shares_descending(self) -> list[float]:
        """The pool's cluster shares, largest first (the 'how is this pool composed' summary)."""
        assert self.share_ is not None, "fit() must be called first"
        return [float(v) for v in np.sort(self.share_)[::-1]]

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descriptors for `df`, with this describer's centres and shares (fitted on some pool)."""
        assert self.km is not None and self.mu_ is not None and self.sd_ is not None
        assert self.fill_ is not None and self.share_ is not None
        x = self._log(df)
        x = np.where(np.isnan(x), self.fill_, x)
        z = (x - self.mu_) / self.sd_
        labels = self.km.predict(z)
        centres = self.km.cluster_centers_[labels] * self.sd_ + self.mu_  # back to log units
        dist = np.linalg.norm(z - self.km.cluster_centers_[labels], axis=1)
        out = pd.DataFrame(centres.astype(np.float32), index=df.index,
                           columns=descriptor_names()[:-2])
        out["regime_share"] = self.share_[labels].astype(np.float32)
        out["regime_dist"] = dist.astype(np.float32)
        return out


def with_descriptors(df: pd.DataFrame, describer: RegimeDescriber) -> pd.DataFrame:
    return pd.concat([df, describer.describe(df)], axis=1)


def monthly_cosine(y: np.ndarray, p: np.ndarray, months: np.ndarray) -> dict[int, float]:
    return {int(m): cosine_similarity(y[months == m], p[months == m]) for m in np.unique(months)}


def paired_verdict(diffs: np.ndarray) -> dict:
    """Paired per-month gain with a t-interval (n months, n - 1 d.o.f.)."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    gain = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * se
    lo, hi = gain - half, gain + half
    adds = lo > 0
    return {"mean_gain": gain, "se": se, "ci_low": lo, "ci_high": hi, "n_months": n,
            "months_improved": int((d > 0).sum()), "adds_signal": bool(adds),
            "worth_keeping": bool(adds and gain >= MATERIAL_GAIN),
            "material_gain_threshold": MATERIAL_GAIN,
            "verdict": ("regime descriptors add signal the 292 features lack" if adds else
                        "not shown: the interval includes zero, so a gain of this size cannot be "
                        "told from none")}


def _in(months: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    return (months >= span[0]) & (months <= span[1])


def run(*, seeds: tuple[int, ...] = (0, 1), out_dir: Path | None = None) -> dict:
    from src.models.lightgbm_model import LightGBMModel

    cfg = load_config()
    df = pd.read_parquet(Path(cfg.paths.features) / "dataset_train.parquet")
    months, y = df["month"].to_numpy(), df["target"].to_numpy(dtype=np.float64)
    tr, va, te = (np.flatnonzero(_in(months, s)) for s in (TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS))
    base = feature_columns(df)
    log.info("rows: train %s, val %s, test %s", f"{len(tr):,}", f"{len(va):,}", f"{len(te):,}")

    t0 = time.perf_counter()
    d_train = RegimeDescriber().fit(df.iloc[tr])  # the training pool, and the "static" describer
    d_val, d_test = RegimeDescriber().fit(df.iloc[va]), RegimeDescriber().fit(df.iloc[te])
    log.info("describers fitted (%.0fs); shares test %s", time.perf_counter() - t0,
             np.round(d_test.shares_descending(), 3).tolist())

    # (frame for train, for val, for test) per variant. "plain" adds nothing.
    sets = {
        "plain": (df.iloc[tr], df.iloc[va], df.iloc[te]),
        "static": (with_descriptors(df.iloc[tr], d_train), with_descriptors(df.iloc[va], d_train),
                   with_descriptors(df.iloc[te], d_train)),
        "transductive": (with_descriptors(df.iloc[tr], d_train),
                         with_descriptors(df.iloc[va], d_val),
                         with_descriptors(df.iloc[te], d_test)),
    }
    del df
    # Each fit's test predictions are kept on disk, so a restart of the machine costs one fit,
    # not all six (this run was once lost to exactly that).
    cache = Path(cfg.paths.features) / "regime_cache"
    cache.mkdir(parents=True, exist_ok=True)
    preds: dict[str, list[np.ndarray]] = {k: [] for k in sets}
    for name, (a, b, c) in sets.items():
        cols = base if name == "plain" else base + descriptor_names()
        for seed in seeds:
            kept = cache / f"{name}_seed{seed}.npy"
            if kept.exists():
                preds[name].append(np.load(kept))
                log.info("%-12s seed %d: reused from %s", name, seed, kept.name)
                continue
            t0 = time.perf_counter()
            model = LightGBMModel(params={"seed": seed}, num_boost_round=2000,
                                  early_stopping_rounds=100)
            model.fit(a[cols], y[tr], eval_set=(b[cols], y[va]))
            preds[name].append(model.predict(c[cols]))
            np.save(kept, preds[name][-1])
            log.info("%-12s seed %d: test cosine %+.5f  rounds %s (%.0fs)", name, seed,
                     cosine_similarity(y[te], preds[name][-1]), model.best_iteration_,
                     time.perf_counter() - t0)

    m_te = months[te]
    per = {k: [monthly_cosine(y[te], p, m_te) for p in v] for k, v in preds.items()}
    mean_per = {k: {m: float(np.mean([s[m] for s in v])) for m in sorted(v[0])}
                for k, v in per.items()}
    result: dict[str, Any] = {
        "split": {"train": TRAIN_MONTHS, "val": VAL_MONTHS, "test": TEST_MONTHS,
                  "n_train": len(tr), "n_val": len(va), "n_test": len(te)},
        "seeds": list(seeds), "n_clusters": N_CLUSTERS, "state_features": list(STATE_FEATURES),
        "pooled_test_cosine": {k: [cosine_similarity(y[te], p) for p in v]
                               for k, v in preds.items()},
        "per_month_test": mean_per,
        "cluster_shares_test": d_test.shares_descending(),
    }
    for variant in ("static", "transductive"):
        diffs = np.array([mean_per[variant][m] - mean_per["plain"][m] for m in mean_per["plain"]])
        result[f"paired_{variant}"] = paired_verdict(diffs)
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regime_clusters.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"month": sorted(mean_per["plain"]),
                  **{k: [mean_per[k][m] for m in sorted(mean_per["plain"])] for k in mean_per}}
                 ).to_csv(out_dir / "regime_clusters_months.csv", index=False, float_format="%.6f")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    r = run(seeds=tuple(args.seeds), out_dir=Path(args.out))
    print(json.dumps({k: r[k] for k in ("pooled_test_cosine", "paired_static",
                                        "paired_transductive")}, indent=2))


if __name__ == "__main__":
    main()
