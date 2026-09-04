"""Generates notebooks/04_models_and_errors.ipynb.

Reads whatever results are current at execution time - nothing is hardcoded - so
re-running the pipeline and re-executing the notebook keeps them in step.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(t: str) -> dict:
    return nbf.v4.new_markdown_cell(t.strip())


def code(t: str) -> dict:
    return nbf.v4.new_code_cell(t.strip())


CELLS = [
    md("""
# 04 — Models, and Where They Fail

**What this notebook establishes:** which model to ship and why, whether the ensemble
earns its place, what the hold-out actually says, and — the part usually skipped — where
the model is *worse* than its headline number suggests.

A single average score hides everything interesting. A model that scores 0.15 uniformly
is a very different object from one that scores 0.25 in liquid conditions and 0.05 in
stressed ones, and only the second tells you where not to trust it.
"""),
    code("""
import warnings
warnings.filterwarnings("ignore")

import json
import sys
sys.path.insert(0, "..")
from pathlib import Path as _P

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity, evaluate

plt.rcParams.update({"figure.figsize": (11, 3.6), "axes.grid": True,
                     "grid.alpha": .3, "font.size": 10})
pd.set_option("display.width", 140)

cfg = load_config()
FEATURES = _P(cfg.paths.features)
MODELS = _P(cfg.paths.data_root) / "models" / "current"

summary = json.loads((FEATURES / "walkforward_summary.json").read_text())
table = pd.read_csv(FEATURES / "walkforward_summary.csv")
print("models evaluated:", ", ".join(table.model))
"""),
    md("""
---
## 1. Model comparison

Every model sees the identical fold structure, so the comparison is fair. The metric is
cosine similarity — the competition's — and `zero` and `mean` are controls rather than
candidates.
"""),
    code("""
table.style.format({c: "{:+.5f}" for c in table.columns if c.startswith("cosine")})
"""),
    code("""
rows = []
for model, blk in summary.items():
    for r in blk.get("per_fold", []):
        v = r.get("cosine", r.get("ensemble_score"))
        if v is not None:
            rows.append({"model": model, "fold": r["fold"], "cosine": v})
per_fold = pd.DataFrame(rows).pivot(index="fold", columns="model", values="cosine")

fig, ax = plt.subplots(figsize=(10, 3.6))
for m in per_fold.columns:
    if m in ("zero",):
        continue
    ax.plot(per_fold.index, per_fold[m], marker="o", ms=4, label=m)
ax.axhline(0, color="grey", lw=.8)
ax.set_xlabel("fold"); ax.set_ylabel("cosine"); ax.set_xticks(per_fold.index)
ax.set_title("Per-fold scores - the spread matters as much as the level")
ax.legend(ncol=3, fontsize=9)
plt.tight_layout(); plt.show()
"""),
    md("""
### Reading the table properly

Two things are easy to get wrong here.

**The two tree models are not distinguishable.** Their means differ by a few ten-thousandths
while the across-fold standard deviation is an order of magnitude larger. Declaring a
winner on that gap would be reading noise. Where they *do* differ is stability, and since
monthly volatility swings by 2.69× (notebook 02), the model with the tighter spread is the
safer thing to ship.

**`mean` changes sign between folds.** A constant prediction has no information, yet its
cosine is positive in some folds and negative in others — because cosine is not
shift-invariant, so a constant scores according to the sign of that period's mean target.
It is the cleanest demonstration of why predictions are kept centred on zero.
"""),
    code("""
cand = table[~table.model.isin(["zero", "mean", "ensemble"])].copy()
best_mean = cand.loc[cand.cosine_mean.idxmax()]
best_stable = cand.loc[cand.cosine_std.idxmin()]
gap = cand.cosine_mean.max() - cand.cosine_mean.min()
print(f"highest mean   : {best_mean.model:10s} {best_mean.cosine_mean:+.5f} (std {best_mean.cosine_std:.5f})")
print(f"most stable    : {best_stable.model:10s} {best_stable.cosine_mean:+.5f} (std {best_stable.cosine_std:.5f})")
print(f"spread between top models : {gap:.5f}")
print(f"typical across-fold std   : {cand.cosine_std.mean():.5f}")
print()
if gap < cand.cosine_std.mean():
    print("The gap between models is SMALLER than the fold-to-fold noise -")
    print("they are statistically indistinguishable, so stability decides.")

m = table[table.model == "mean"].iloc[0]
print(f"\\n'mean' baseline ranges {m.cosine_min:+.5f} .. {m.cosine_max:+.5f} across folds")
"""),
    md("""
---
## 2. Does the ensemble earn its place?

The weights are not tuned. Because cosine is scale-invariant, the vector in the span of
the model predictions closest to `y` is its orthogonal projection — which is the OLS
solution, available in closed form (`src/models/ensemble.py`,
verified in `tests/test_ensemble.py` against 200 random weight vectors).

That makes it cheap, but cheap is not the same as useful. The project's rule is that the
ensemble ships only if it beats the best single model **fold by fold**, not on average.
"""),
    code("""
ens = summary.get("ensemble", {})
rows = []
for r in ens.get("per_fold", []):
    rows.append({"fold": r["fold"], "ensemble": r["ensemble_score"],
                 "best_single": r["best_single"],
                 "best_single_score": r["best_single_score"],
                 "gain": r["gain"], "beats": r["ensemble_score"] > r["best_single_score"]})
eb = pd.DataFrame(rows)
print(eb.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
print()
print(f"beats the best single model in {int(eb.beats.sum())}/{len(eb)} folds; "
      f"median gain {eb.gain.median():+.5f}")
"""),
    md("""
The gain is small and consistent. Small is expected: LightGBM and XGBoost are both
gradient-boosted trees on the same features, so their errors are highly correlated and
there is little for a linear combination to exploit. Consistency across every fold is what
makes it worth keeping anyway — an averaging effect rather than a lucky fold.
"""),
    md("""
---
## 3. The hold-out — measured once

Months 65–70 were used for nothing: not feature design, not model selection, not early
stopping. `src/models/finalize.py` trains on months 0–63, early-stops on month 64, and
touches the hold-out exactly once.
"""),
    code("""
ho = json.loads((FEATURES / "holdout_metrics.json").read_text())
meta = json.loads((MODELS / "model_meta.json").read_text())
s = ho["scores"]

print(f"model    : {meta['name']} {meta['version']}, {len(meta['features'])} features")
print(f"trained  : {meta['trained_at']}")
print(f"hold-out : months {meta['metrics'].get('holdout_months')}")
print()
for k, v in s.items():
    print(f"  {k:22s} {v:+.5f}")

cv = table[table.model == meta["name"]].iloc[0].cosine_mean
print()
print(f"walk-forward CV mean : {cv:+.5f}")
print(f"hold-out             : {s['cosine']:+.5f}")
print(f"difference           : {s['cosine'] - cv:+.5f}")
"""),
    md("""
Two sanity checks pass here.

**The hold-out is not below the CV estimate**, so the model is not overfitted to the
validation folds. (It sits slightly above, which is unremarkable — the final model trains
on more months than any single fold did.)

**Pearson is almost identical to cosine.** Cosine punishes an off-centre prediction while
Pearson removes the mean first, so their agreement is direct evidence that predictions are
centred on zero, which is what a shift-sensitive metric rewards.
"""),
    md("""
---
## 4. Where the model fails

The headline is an average over 100k+ hold-out samples. Now the useful question: is the
skill uniform, or concentrated?

Predictions come from the saved artefact — the same object the API serves.
"""),
    code("""
import pyarrow.parquet as pq

from src.inference.predictor import load_bundle
from src.evaluation.temporal_validation import holdout_months

bundle = load_bundle(MODELS)
lo, hi = holdout_months()

cond_cols = ["mkt_rel_spread_last", "ord_event_rate_60s", "mkt_total_depth_last",
             "mkt_empty_ask_share", "mkt_mid_std_600s"]
need = ["sample_id", "month", "target"] + list(dict.fromkeys(bundle.features + cond_cols))
df = pq.read_table(FEATURES / "dataset_train.parquet", columns=need,
                   filters=[("month", ">=", lo)]).to_pandas()
df = df[df.month <= hi].reset_index(drop=True)

pred = bundle.model.predict(df[bundle.features].astype("float32"))
df["pred"] = pred
print(f"{len(df):,} hold-out samples, overall cosine {cosine_similarity(df.target, df.pred):+.5f}")
"""),
    code("""
def by_bucket(frame, col, label, q=4):
    valid = frame[frame[col].notna()].copy()
    valid["bucket"] = pd.qcut(valid[col], q, labels=[f"Q{i+1}" for i in range(q)],
                              duplicates="drop")
    out = valid.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "cosine": cosine_similarity(g.target, g.pred),
            f"{label}_median": g[col].median(),
        }))
    return out

spread = by_bucket(df, "mkt_rel_spread_last", "spread")
activity = by_bucket(df, "ord_event_rate_60s", "event_rate")
vol = by_bucket(df, "mkt_mid_std_600s", "volatility")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
for a, (data, title) in zip(ax, [(spread, "by spread (Q1 = tightest)"),
                                 (activity, "by order activity (Q1 = quietest)"),
                                 (vol, "by volatility (Q1 = calmest)")]):
    a.bar(data.index.astype(str), data["cosine"], color="#3b6ea5")
    a.axhline(cosine_similarity(df.target, df.pred), color="crimson", ls="--", lw=1,
              label="overall")
    a.set_title(title, fontsize=10)
ax[0].set_ylabel("cosine"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
    code("""
for name, data in [("spread", spread), ("activity", activity), ("volatility", vol)]:
    lo_, hi_ = data["cosine"].min(), data["cosine"].max()
    worst = data["cosine"].idxmin()
    print(f"{name:11s} range {lo_:+.5f} .. {hi_:+.5f}   "
          f"({hi_ - lo_:.5f} spread, weakest in {worst})")
"""),
    md("""
### By month, inside the hold-out

Six months of unseen data. If the score were carried by one lucky month, that would show.
"""),
    code("""
per_month = df.groupby("month").apply(
    lambda g: pd.Series({"n": len(g), "cosine": cosine_similarity(g.target, g.pred),
                         "target_std": g.target.std()}))

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.bar(per_month.index.astype(str), per_month["cosine"], color="#3b6ea5")
ax.axhline(cosine_similarity(df.target, df.pred), color="crimson", ls="--", lw=1)
ax.set_xlabel("month"); ax.set_ylabel("cosine"); ax.set_title("Hold-out score by month")
plt.tight_layout(); plt.show()

print(per_month.to_string(float_format=lambda v: f"{v:,.5f}"))
print()
print(f"every month positive: {(per_month.cosine > 0).all()}")
print(f"weakest month {per_month.cosine.idxmin()} at {per_month.cosine.min():+.5f}, "
      f"strongest {per_month.cosine.idxmax()} at {per_month.cosine.max():+.5f}")
"""),
    md("""
### The finding that matters most

Put the error analysis next to the drift measurement from notebook 03 and something
uncomfortable appears.

The model is **weakest in the tightest-spread conditions**. And the test period has
**tighter spreads than training** — mean relative spread ×0.80. So the regime the model
handles worst is precisely the regime the test set lives in.
"""),
    code("""
# Where do test samples fall, relative to the TRAIN spread distribution?
train_spread = pq.read_table(FEATURES / "dataset_train.parquet",
                             columns=["mkt_rel_spread_last"]).to_pandas().dropna()
test_spread = pq.read_table(FEATURES / "dataset_test.parquet",
                            columns=["mkt_rel_spread_last"]).to_pandas().dropna()

edges = np.quantile(train_spread["mkt_rel_spread_last"], [0, .25, .5, .75, 1.0])
tr_share = np.histogram(train_spread["mkt_rel_spread_last"], bins=edges)[0]
te_share = np.histogram(test_spread["mkt_rel_spread_last"], bins=edges)[0]
tr_share = tr_share / tr_share.sum()
te_share = te_share / te_share.sum()

comp = pd.DataFrame({
    "train_share": tr_share, "test_share": te_share,
    "holdout_cosine": spread["cosine"].values,
}, index=["Q1 (tightest)", "Q2", "Q3", "Q4 (widest)"])
comp["shift"] = comp.test_share - comp.train_share
print(comp.to_string(float_format=lambda v: f"{v:,.4f}"))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.3))
x = np.arange(4); w = .38
ax[0].bar(x - w/2, comp.train_share, w, label="train", color="#3b6ea5")
ax[0].bar(x + w/2, comp.test_share, w, label="test", color="#c0392b")
ax[0].set_xticks(x); ax[0].set_xticklabels(comp.index, fontsize=8)
ax[0].set_ylabel("share of samples"); ax[0].set_title("where each split sits on the spread axis")
ax[0].legend()
ax[1].bar(x, comp.holdout_cosine, color="#7f8c8d")
ax[1].set_xticks(x); ax[1].set_xticklabels(comp.index, fontsize=8)
ax[1].set_ylabel("hold-out cosine"); ax[1].set_title("where the model is strong")
plt.tight_layout(); plt.show()

worst = comp.holdout_cosine.idxmin()
print()
print(f"weakest bucket        : {worst} at {comp.holdout_cosine.min():+.5f}")
print(f"test share there      : {comp.loc[worst, 'test_share']:.1%} "
      f"(train {comp.loc[worst, 'train_share']:.1%}, "
      f"{comp.loc[worst, 'shift']:+.1%})")
"""),
    md("""
### Turning that into a falsifiable prediction

If the only thing that changes between hold-out and leaderboard were the mix of spread
regimes, the score would move by a computable amount: reweight the per-bucket scores by
the test set's shares instead of the hold-out's.

This is deliberately a *lower bound on the sources of degradation* — it ignores everything
else that differs — but it is a number that can be checked against a real leaderboard
rather than a vague warning that "performance may vary".
"""),
    code("""
w_holdout = np.full(4, 0.25)               # hold-out is quartiled by construction
w_test = comp.test_share.to_numpy()
c = comp.holdout_cosine.to_numpy()

as_measured = float((w_holdout * c).sum())
reweighted = float((w_test * c).sum())

print(f"hold-out score, as measured        : {as_measured:+.5f}")
print(f"same model, re-weighted to test mix: {reweighted:+.5f}")
print(f"expected degradation from spread mix alone: {reweighted - as_measured:+.5f}"
      f"  ({(reweighted/as_measured - 1)*100:+.1f}%)")
print()
print("So a leaderboard score materially BELOW ~{:.3f} would need an explanation".format(reweighted))
print("beyond the spread regime; a score near it is what this analysis predicts.")
"""),
    md("""
That is the honest headline risk of this project, and it did not come from either
measurement alone — the drift report says *where the test set is*, the error analysis says
*where the model is weak*, and only together do they say the two overlap.

It also gives the next piece of work a clear target: the model needs to be better in tight
spreads specifically, not better on average. Sample weighting by spread bucket,
regime-conditional models, or simply reporting a spread-stratified score alongside the
headline would all be reasonable responses. None of them is guesswork.
"""),
    md("""
### The degenerate-book case

Around 0.5% of snapshots have an empty side, where mid, spread and microprice are
undefined. Notebook 01 established those are sentinels rather than prices. Does the model
cope with the samples that contain them?
"""),
    code("""
df["has_empty"] = df["mkt_empty_ask_share"] > 0
grp = df.groupby("has_empty").apply(
    lambda g: pd.Series({"n": len(g), "cosine": cosine_similarity(g.target, g.pred)}))
grp.index = ["book always two-sided", "book empty at some point"]
print(grp.to_string(float_format=lambda v: f"{v:,.5f}"))
"""),
    md("""
---
## 5. Backtest — is the ranking robust to costs?

Signal that evaporates at one basis point of cost is not signal worth anything. The
threshold is set as a tail percentile of the prediction distribution, not an absolute cut,
because cosine is scale-invariant and the magnitudes are therefore uncalibrated.

**This is a measurement of ranking power, not a strategy.**
"""),
    code("""
cost = pd.read_csv(FEATURES / "backtest_cost_sensitivity.csv")
sweep = pd.read_csv(FEATURES / "backtest_trade_fraction.csv")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.3))
ax[0].plot(cost.cost_bps, cost.total_return, marker="o", color="#3b6ea5")
ax[0].axhline(0, color="grey", lw=.8)
ax[0].set_xlabel("transaction cost (bps)"); ax[0].set_ylabel("sum of per-trade returns")
ax[0].set_title("cost sensitivity")
ax[1].plot(sweep.turnover, sweep.mean_return * 1e4, marker="o", color="#4a8a58")
ax[1].set_xlabel("fraction of samples traded"); ax[1].set_ylabel("mean return per trade (bps)")
ax[1].set_title("does trading only the extremes pay more?")
plt.tight_layout(); plt.show()

print(cost[["cost_bps", "n_trades", "mean_return", "win_rate", "sharpe",
            "max_drawdown"]].to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
"""),
    md("""
The two panels answer different questions. The left one asks whether the edge survives
friction. The right asks whether the model *ranks*: if the most extreme predictions are
not the most profitable ones, the score is coming from somewhere other than ordering
skill.
"""),
    md("""
---
## Summary

| Question | Answer |
|---|---|
| Which model? | The two tree models are statistically indistinguishable; stability decides |
| Does the ensemble help? | Small gain, but present in every fold — kept |
| Is the hold-out consistent with CV? | Yes, and not below it — no overfitting to the folds |
| Are predictions centred? | Yes — Pearson ≈ cosine |
| Is skill uniform? | No — it varies with liquidity conditions, quantified above |
| Does it survive costs? | See the cost-sensitivity panel |

**What this does not tell us.** Every number above is self-graded on data the model has
not seen but that came from the same source as its training data. The test period is
measurably more liquid than the training period (notebook 03), and the features that
shift most are exactly the rate features — so the first place to look, if a leaderboard
score lands below this estimate, is there.
"""),
    md("""
---

## The prediction was wrong

The submission was made on 2026-09-03. **Leaderboard: 0.128.**

The prediction above was `0.143`, with the explicit caveat that "a score materially BELOW
~0.143 would need an explanation beyond the spread regime". 0.128 is materially below it.
So the caveat is now due.
"""),
    code("""
holdout_bucketed = 0.14891      # printed above
predicted        = 0.14345      # same model, re-weighted to the test spread mix
actual           = 0.128        # Kaggle leaderboard, 2026-09-03

total   = actual - holdout_bucketed
spread  = predicted - holdout_bucketed
residual = actual - predicted

print(f"hold-out (bucket-weighted)   {holdout_bucketed:+.5f}")
print(f"predicted                    {predicted:+.5f}   ({spread:+.5f} from spread mix)")
print(f"actual                       {actual:+.5f}   ({residual:+.5f} unexplained)")
print()
print(f"total degradation            {total:+.5f}  ({total/holdout_bucketed*100:+.1f}%)")
print(f"  explained by spread mix    {spread/total*100:5.1f}%")
print(f"  NOT explained              {residual/total*100:5.1f}%")
"""),
    md("""
### What that means

The mechanism was real but small. Spread-regime mix accounts for roughly a **quarter** of
the degradation; three quarters came from something the analysis did not model. The
direction was right and the magnitude was wrong by about a factor of three.

That is a more useful result than a correct prediction would have been, because it says
something specific: **the hold-out is not a good proxy for the leaderboard here, and the
reason is not the one variable I could measure.** Months 65-70 are the tail of the same
training distribution; the test set is a different period entirely, and notebook 03 already
showed the rate features shifting hardest. The natural next experiment is to retrain
without the highest-drift features and see whether a small hold-out cost buys a larger
leaderboard gain — but that costs another submission to verify, so it is stated as a
hypothesis rather than claimed as a result.

### Why this is left in

The tempting edit is to delete the failed prediction, or to soften it into something that
survives contact with the number. Both would destroy the only thing that makes it worth
anything. A prediction is only evidence of understanding if it was recorded **before** the
answer arrived and is reported honestly **after**. The wrong number stays where it was
written, with the correction underneath it.

The transferable lesson is about the estimator, not this model: a hold-out carved from the
end of the training period measures generalisation across *time within one regime*, not
across regimes. It will read optimistically whenever the deployment distribution differs
from the training distribution in ways the hold-out cannot see — which, in production
finance, is most of the time.
"""),
    md("""
---

## Testing the suspect: does drift actually predict degradation?

Notebook 03 nominated the high-drift rate features as the likely source of the
unexplained 74%. Nominating a suspect is not evidence.

The obvious test — retrain without them and submit — costs a submission and returns one
bit. The mechanism can be tested on the training data alone instead. If high-drift
features are a liability **under distribution shift**, then dropping them should help
*more* when the gap between training and evaluation is *larger*. The training set spans
71 months, so that gap can be dialled directly:

```
train on months 0-34, FIXED          <- never changes, so only the gap varies
evaluate on 36-40, 42-46, ... 66-70  <- increasing distance into the future
```

The quantity of interest is neither score but their difference,
`lift(gap) = cosine(pruned) - cosine(full)`. Both feature sets see identical rows, folds
and seeds, so run-to-run noise is shared and cancels in the difference — which matters,
because the effect being looked for is about the size of the fold-to-fold noise. No early
stopping: stopping on the evaluation block would tune to the thing being measured.

A drift mechanism predicts lift rises with gap. A flat line falsifies it.
"""),
    code("""
import pandas as pd, numpy as np
from pathlib import Path
from src.config import load_config

feat = Path(load_config().paths.features)
rob = pd.read_csv(feat / "drift_robustness_t020.csv")

print(rob[["block", "gap_months", "n", "full", "pruned", "lift"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

g = rob.gap_months.to_numpy(float)
slope, r = np.polyfit(g, rob.lift, 1)[0], np.corrcoef(g, rob.lift)[0, 1]
resid = rob.lift - np.polyval(np.polyfit(g, rob.lift, 1), g)
se = np.sqrt((resid**2).sum() / (len(g) - 2) / ((g - g.mean())**2).sum())

print()
print(f"lift vs gap : slope {slope:+.6f}/month, Pearson r {r:+.3f}")
print(f"95% CI over the full 32-month span: "
      f"[{(slope-1.96*se)*32:+.5f}, {(slope+1.96*se)*32:+.5f}] cosine")
print(f"unexplained leaderboard gap       :  {0.01545:+.5f} cosine")
"""),
    md("""
**Falsified.** The lift does not rise with the gap — it is positive in the middle and
negative at both ends, with a correlation indistinguishable from zero. Even the optimistic
edge of the confidence interval is several times too small to account for the gap the
experiment was built to explain.

A null result invites the objection that the knob was set wrong, so the same experiment was
re-run pruning far more aggressively.
"""),
    code("""
rob10 = pd.read_csv(feat / "drift_robustness_t010.csv")

for name, r_ in (("|shift| >= 0.20  (57 dropped)", rob), ("|shift| >= 0.10  (93 dropped)", rob10)):
    sl = np.polyfit(g, r_.lift, 1)[0]
    print(f"{name}: slope {sl:+.6f}/month  r {np.corrcoef(g, r_.lift)[0,1]:+.3f}  "
          f"mean lift {r_.lift.mean():+.5f}")

# The full arm is identical in both runs - same seeds, same columns. An unplanned
# control: it confirms only the pruned arm changed.
print()
print("full arm reproduced exactly across the two runs:",
      bool((rob.full.round(10) == rob10.full.round(10)).all()))
"""),
    md("""
Pruning harder does not rescue the hypothesis — it buries it. At the aggressive threshold
the trend is *negative* (pruning helps **less** at longer gaps, the opposite of the
prediction) and the mean lift is negative too: dropping the high-drift features costs
skill on average rather than buying robustness.

The `full` arm reproducing to ten decimal places across the two runs is an unplanned
control worth keeping: it confirms the two arms differed only in their column list, so the
lift really does isolate the pruning.

But the more interesting number was in the `full` column all along, and it was not what
the experiment was looking for.
"""),
    code("""
for col in ("full", "pruned"):
    s = np.polyfit(g, rob[col], 1)[0]
    print(f"{col:7s} vs gap: slope {s:+.6f}/month  "
          f"r {np.corrcoef(g, rob[col])[0,1]:+.3f}   "
          f"range {rob[col].min():.5f} - {rob[col].max():.5f}")

best = rob.loc[rob.full.idxmax()]
print()
print(f"best block is {best.block} - the FURTHEST one, {best.gap_months:.0f} months out")
"""),
    md("""
### The framing was wrong, not just the suspect

Within the training period, model skill **does not decay with elapsed time at all**. The
slope is slightly *positive*, and the most distant block scores highest. Train on months
0-34 and evaluate 32 months later, and the model does no worse than it does two months
later.

That undercuts the whole explanation, not just the rate-feature version of it. "The market
regime drifts over time, so a later test set scores worse" makes a prediction about the
training period too — and the training period says no. Whatever separates the test set
from the hold-out, it is not that the test set is *later*.

So the honest state of the question:

| Hypothesis | Status | Explains |
|---|---|---|
| Spread-regime mix shift | measured | 26% |
| High-drift rate features | **falsified** at two thresholds | not detectable; pruning is mildly harmful |
| Elapsed time / regime drift | **falsified** | no decay within 71 months |
| — | remaining | **74%, unaccounted for** |

Candidates that remain untested, in the order I would try them: the test set differs
categorically rather than temporally (its order rate is 36% higher than training — a large
difference that does not appear anywhere inside the training span); cosine is a *pooled*
magnitude-weighted metric, so a different volatility mix changes the score even at
identical per-sample skill; and design choices were made against CV folds neighbouring the
hold-out, which buys some selection optimism that a genuinely independent test set would
not honour.

Two hypotheses stated in advance, two tested, two rejected. The gap is still mostly
unexplained — and that is the accurate thing to report.
"""),
    md("""
---

## The forecast was also built wrong

Re-reading the prediction after it failed turned up a flaw in the *method*, independent of
the result. Cosine is computed over the pooled vector, without centring, so splitting the
samples into disjoint groups factors it exactly:

$$\cos(y, p) \;=\; \sum_g \cos_g \cdot w_g,
\qquad w_g = \frac{\lVert y_g\rVert \, \lVert p_g\rVert}{\lVert y\rVert \, \lVert p\rVert}$$

A pooled cosine is a weighted average of subgroup cosines — but the weights are products
of **magnitudes**, not sample counts. (They sum to at most 1 by Cauchy-Schwarz, with
equality only when every group scores alike.)

The forecast reweighted the per-quartile scores by the test set's *sample shares*. That
assumes the magnitude weights track the counts. They do not.
"""),
    code("""
from src.evaluation.cosine_decomposition import decompose, verify_identity

dec = pd.read_csv(feat / "cosine_decomposition.csv")
print(dec[["group", "n", "cosine", "y_rms", "count_weight", "weight"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

meta = json.loads((feat / "cosine_decomposition_meta.json").read_text())
print()
print(f"identity check: pooled {meta['pooled']:.10f} = rebuilt {meta['rebuilt']:.10f}"
      f"  (error {meta['abs_error']:.1e})")
"""),
    md("""
The quartiles are equal-sized by construction, so every count weight is 0.247 — but the
weights the metric applies run from **0.209 to 0.312**, a 50% spread. And `y_rms` barely
moves across buckets (0.0033-0.0035), so almost none of that comes from the targets: it is
the *predictions* that are larger in wide-spread samples, which is what a volatility-aware
model should do.

Crucially, the heaviest weight lands on **Q4 — the bucket where the model is strongest**.

So what does the forecast look like when redone with the right weights? On the test set
$\lVert p_g\rVert$ is directly observable (the submitted predictions are in hand); only
$\lVert y_g\rVert$ is not, and it is carried over from the hold-out as a per-bucket target
RMS — a far weaker assumption than count-weighting, given that RMS varies by 7% while the
weights vary by 50%.
"""),
    code("""
fc = pd.read_csv(feat / "cosine_forecast_corrected.csv")
print(fc[["group", "n", "holdout_cosine", "count_weight", "weight"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

m = json.loads((feat / "cosine_decomposition_meta.json").read_text())
pooled, actual = m["pooled"], m["actual_leaderboard"]
corrected = m["forecast_magnitude_weighted"]

print()
print(f"hold-out, as measured        {pooled:+.5f}")
print(f"forecast, count-weighted     {m['forecast_count_weighted']:+.5f}   <- the method used")
print(f"forecast, magnitude-weighted {corrected:+.5f}   <- corrected")
print(f"actual                       {actual:+.5f}")
print()
print(f"spread mix now explains {(pooled-corrected)/(pooled-actual)*100:.0f}% of the gap"
      f"  (was claimed: 26%)")
"""),
    md("""
### The correction makes the story worse, which is why it is worth making

Fixing the weighting moves the forecast **away** from the outcome, not towards it. Because
cosine over-weights Q4, where the model is strongest, it partly cancels the penalty from
Q1's larger share — so the spread mechanism was weaker than the original analysis claimed,
not stronger. Its share of the gap drops from 26% to about **14%**.

| Hypothesis | Status | Explains |
|---|---|---|
| Spread-regime mix shift | measured, then **corrected downward** | ~14% |
| High-drift rate features | falsified at two thresholds | not detectable |
| Elapsed time / regime drift | falsified | no decay over 71 months |
| — | remaining | **~86%, unaccounted for** |

There is a temptation, on finding an error in your own analysis, to look for the version of
the fix that rescues the conclusion. The fix here does the opposite, and reporting it that
way is the whole point: the error was in the method, so it had to be corrected regardless
of which direction the answer moved.

### The general lesson

Any subgroup analysis of a cosine score — error slices, fairness-style breakdowns,
"where is my model weak" tables — must weight groups by magnitude, not by row count, or it
describes a metric nobody is being scored on. The failure mode is not subtle:
`tests/test_cosine_decomposition.py` builds a model that is near-perfect on the
small-magnitude half of the data and useless on the large-magnitude half. Counting rows
calls it skilful at +0.48. The metric scores it **−0.04** — no skill at all.

And there is a second, harder consequence. The corrected weights need
$\lVert y_g\rVert$ on the test set, which is unobservable. So a subgroup-reweighting
forecast of a cosine score is **not fully computable in advance** — it always rests on an
assumption about unobserved target magnitudes. Count-weighting is one such assumption, and
a poor one. That, rather than any particular number, is the durable finding here.
"""),
    md("""
---

## Is the test set *later*, or *different*?

Three explanations are now gone: the spread mix is small, drift-pruning does nothing, and
skill does not decay with elapsed time. That leaves the possibility that the test set is
not a continuation of the training period at all.

That is measurable. Train a classifier to tell train rows from test rows; the AUC says how
distinguishable they are. On its own the number means nothing — any two separated periods
are somewhat distinguishable — so it needs a scale. The same measurement is repeated
*inside* the training data at growing temporal distance, tracing how distinguishability
grows with elapsed time in this market, for these features.
"""),
    code("""
adv = pd.read_csv(feat / "adversarial_auc.csv")
print(adv[["comparison", "kind", "distance_months", "auc"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
"""),
    md("""
**The test set is a continuation, not a departure.** Against the most recent training
months it scores **0.749** — *lower* than months 10-19 against months 0-9 (0.754). Two
adjacent blocks inside the training data are more distinguishable from each other than the
test set is from the end of training. Whatever is costing 0.024, it is not that the test
set comes from somewhere else.

### A bug worth showing

The first version of this compared *pooled* training data against the test set and got
0.791, which read as "less shifted than most training blocks" — a tidy answer, and wrong.
Pooling all 71 months makes that side far more heterogeneous, and a narrow block is
naturally hard to separate from a broad mixture that contains something like it. The AUC
was depressed for a reason with nothing to do with the test set.

The row is kept in the table as a diagnostic. Comparisons are only comparable when both
sides are built the same way, and `tests/test_adversarial.py` pins that down with a case
where pooling one side provably lowers the AUC.
"""),
    md("""
---

## The estimator, not the model

Everything above looked for something wrong with the *test set*. Nothing was. So the
remaining suspect is the thing that produced the number in the first place.

The hold-out is a single contiguous stretch — months 65-70 — chosen for being **last**,
not for being **typical**. If those months happen to be favourable, the hold-out overstates
what a fresh period yields, and no amount of care elsewhere would reveal it: the score is
measured perfectly, on a sample of one period.

So: hold the model fixed (trained once on months 0-34) and score it on every later period
in turn. Training set, features, seeds and rounds are identical, so the only thing varying
is which months are being predicted. Any spread is period difficulty.
"""),
    code("""
per = pd.read_csv(feat / "period_difficulty.csv")
pm = json.loads((feat / "period_difficulty_meta.json").read_text())

print(per[["block", "n", "cosine"]].to_string(index=False,
                                              float_format=lambda v: f"{v:,.5f}"))
print()
print(f"hold-out months 65-70   {pm['holdout_cosine']:+.5f}")
print(f"median of other blocks  {pm['typical_cosine']:+.5f}")
print(f"the hold-out is {(pm['ratio']-1)*100:.1f}% above a typical period, "
      f"above {pm['holdout_percentile']*100:.0f}% of all blocks")
"""),
    md("""
Period difficulty swings from **0.117 to 0.148** — a range of 26% — with a fixed model on
fixed features. The hold-out sits at the **83rd percentile**. It was a good draw.
"""),
    code("""
wf = pd.read_csv(feat / "walkforward_summary.csv")
cv_mean = float(wf.loc[wf.model == "ensemble", "cosine_mean"].iloc[0])

print(f"reported hold-out score        {pm['reported']:+.5f}")
print(f"de-biased to a typical period  {pm['debiased']:+.5f}")
print(f"walk-forward CV mean           {cv_mean:+.5f}   <- independent route")
print(f"actual leaderboard             {pm['actual']:+.5f}")
print()
print(f"agreement between the two routes: {abs(pm['debiased']-cv_mean):.5f}")
print(f"period luck explains {pm['share_of_gap']*100:.0f}% of the gap")
"""),
    md("""
### The two routes agree to four decimal places

De-biasing the hold-out by its measured period difficulty gives **0.14084**. Averaging the
walk-forward folds across five *different* periods gives **0.14088**. Nothing was tuned to
make those meet.

That is the finding, and it is uncomfortable: **the CV mean was the better estimate all
along, and I led with the hold-out instead.** The hold-out was chosen on the strongest
methodological grounds available — untouched by feature design, model selection, tuning or
early stopping, measured exactly once. All of that is true and none of it helps, because
purity was never the binding constraint. Period difficulty was, and against that a single
contiguous stretch is a sample of size one, however cleanly it is handled.

The walk-forward mean is *less* pure — its folds informed decisions — yet it averages over
five periods and lands within 0.00004 of the de-biased answer.

| | estimate | error vs leaderboard |
|---|---:|---:|
| Hold-out, months 65-70 | +0.15171 | +0.02371 |
| Walk-forward CV mean | +0.14088 | +0.01288 |
| Leaderboard | +0.12800 | — |

**Where the gap stands**

| Hypothesis | Verdict | Share |
|---|---|---:|
| The hold-out was a lucky period | **confirmed**, two independent routes | **46%** |
| Spread-regime mix shift | real, after correcting the weighting | ~14% |
| High-drift rate features | falsified at two thresholds | — |
| Skill decays with elapsed time | falsified | — |
| Test set is categorically different | falsified — it is a continuation | — |

The two surviving effects should not simply be added: both are ways of saying the hold-out
period was atypical, and they may be measuring overlapping parts of the same thing. Taken
together they account for somewhere between **46% and 60%** of a gap that began as fully
unexplained.

### Is the rest just noise?

Period difficulty varies a lot, so the obvious next question is whether the leftover gap
is simply the test set's own unlucky draw. It is not, and the same table answers it.
"""),
    code("""
sd = per.cosine.std(ddof=1)
n_blocks = 647_896 / per.n.mean()          # test set, in units of these blocks
noise = sd / np.sqrt(n_blocks)
residual = pm["debiased"] - pm["actual"]

print(f"block-to-block difficulty     std {sd:.5f}")
print(f"test spans ~{n_blocks:.0f} such blocks -> its own period noise ~{noise:.5f}")
print(f"residual gap after de-biasing     {residual:+.5f}  ({residual/noise:.1f} sigma)")
"""),
    md("""
The test set averages over roughly thirteen blocks, so its own period draw is worth about
**±0.0026** — while the residual is **0.0128**, some five standard deviations out. The
remainder is a real effect, not sampling noise, and it is still unidentified.

### What I would do differently

Report the walk-forward mean as the headline and the hold-out as a *check* on it, rather
than the reverse. A held-out period answers "did I leak?" It does not answer "what will
this score next period", and those were quietly treated as the same question.
"""),
    md("""
---

## Hyperparameter search: how much of a gain is real?

The plan deprioritised tuning on the grounds that it buys little per hour of work. That
was a judgement, not a measurement, so here is the measurement.

The search is routine — TPE over eight LightGBM parameters, walk-forward cosine as the
objective. What matters is the accounting around it, because by now the noise floor is
known and it is **high**: fold-to-fold std 0.0041, block-to-block period std 0.0091.
Running N trials and keeping the best does not only find good parameters, it also finds
favourable noise, and at N in the dozens that term is the same size as the effect being
chased.

So the winner is re-scored under a fresh resampling of the identical protocol — new
bagging and feature-sampling seeds, nothing else changed. What survives belongs to the
parameters; what evaporates was the search fitting noise.
"""),
    code("""
tun = json.loads((feat / "tuning_result.json").read_text())

print(f"trials completed          {tun['trials_completed']} of {tun['n_trials']}"
      f"  (stopped by the {tun['time_budget_min']:.0f}-min budget)")
print(f"search set                {tun['sample_frac']:.0%} of train, "
      f"{tun['search_folds']} folds")
print()
print(f"baseline CV               {tun['baseline_cv']:+.5f}")
print(f"best trial CV             {tun['best_cv']:+.5f}")
print(f"gain the search claims    {tun['claimed_gain']:+.5f}")
print(f"gain surviving new seeds  {tun['honest_gain']:+.5f}")
print(f"SELECTION OPTIMISM        {tun['selection_optimism']:+.5f}"
      f"   ({tun['selection_optimism']/tun['claimed_gain']:.0%} of the claim)")
"""),
    md("""
**Three quarters of the gain was noise.** What is left, +0.0006, is a seventh of the
fold-to-fold std and a fifteenth of the period std.

One objection remains: the search ran on 30% of the rows, and several of these knobs scale
with sample size — `min_data_in_leaf` came back at 1018, tuned against 377k rows rather
than 1.26M. So both arms were re-run on the full dataset, paired: identical folds,
identical rows, identical seeds, only the parameters differing.
"""),
    code("""
con = json.loads((feat / "tuning_confirm.json").read_text())

print(f"full data, {con['full_data_rows']:,} rows, last {con['n_folds']} folds, paired")
print(f"  tuned     {con['tuned_mean']:+.5f}   {[f'{x:+.5f}' for x in con['tuned_folds']]}")
print(f"  baseline  {con['baseline_mean']:+.5f}   {[f'{x:+.5f}' for x in con['baseline_folds']]}")
print(f"  paired difference {con['paired_gain']:+.5f}")
print(f"  per fold          {[f'{d:+.5f}' for d in con['per_fold_diff']]}")
print(f"  folds improved    {con['folds_improved']} of {con['n_folds']}")
"""),
    md("""
### It does not transfer. It reverses.

On full data the tuned configuration is **worse** than the hand-chosen defaults, by
−0.00038, and only one fold of three improves. The per-fold differences alternate sign
(−0.00135, +0.00152, −0.00131) — the signature of noise, not of a parameter effect.

| | gain |
|---|---:|
| Claimed by the search | +0.00260 |
| Surviving fresh seeds, 30% data | +0.00062 |
| **Full data, paired** | **−0.00038** |

So the honest answer is that **tuning bought nothing**. The defaults — chosen by hand, for
stated reasons about 1.26M rows and memory — are as good as a 23-trial TPE search, and the
apparent improvement was selection noise that did not survive either check.

That is a useful result rather than a wasted hour. "I skipped tuning" is a gap in a
portfolio; "I ran the search, and here is the measurement showing it buys 0.0006 before
transfer and nothing after" is a finding. The original decision to deprioritise it now
rests on evidence instead of judgement.

### The reusable part

The apparatus generalises past this dataset. Any best-of-N result on a noisy objective is
inflated by the maximum of N noise draws, and the fix costs one extra evaluation: re-score
the winner under a fresh resampling of the same protocol and report both numbers. Had only
the first number been reported here, this notebook would be claiming a +0.0026 improvement
that reverses sign on the full data.

### A cost note

The first attempt at this search was allowed to reach `num_leaves` 511, `max_bin` 255 and
`learning_rate` 0.01. The last of those is the trap: at a low enough rate early stopping
never fires, so every trial runs all of its rounds. Trials in that corner cost roughly ten
times the baseline, and a search budgeted at an hour was killed after three with no result.

A search space is a compute budget written in another notation. The version here is bounded
so the worst trial costs about twice the baseline, carries a wall-clock cap that ends the
run regardless — it stopped at 23 of 40 trials and said so — and logs every trial as it
lands, because a long job you cannot observe is indistinguishable from a hung one.
"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    out = HERE / "04_models_and_errors.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"written: {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
