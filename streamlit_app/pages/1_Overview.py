"""MSCapital - Overview.

The landing page leads with three numbers rather than one. A single "best score" would be
the hold-out, and the hold-out is the number this project spent five experiments learning
not to trust: it sits at the 83rd percentile of period difficulty. Showing the internal
estimate, the lucky read and the externally graded result side by side is the honest
summary, and it is also the more interesting one.
"""
import sys
from pathlib import Path

# Put the repository root on sys.path before importing anything from it.
#
# `python -m streamlit` silently adds the working directory; a bare `streamlit run` - which
# is what Streamlit Community Cloud executes - adds the MAIN SCRIPT'S directory instead. So
# `streamlit_app/` lands on the path and the repository root does not, and every
# `from streamlit_app.lib import ...` below fails with ModuleNotFoundError. It works locally
# and breaks on deploy, which is the worst place to find out.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from streamlit_app.lib import (
    feature_columns,
    histogram,
    load_csv,
    load_features,
    load_json,
    load_results_table,
    missing,
    page_header,
)

st.set_page_config(page_title="MSCapital | Overview", layout="wide")
page_header(
    "MSCapital - Market Intelligence",
    "Short-horizon return prediction from 60 s of order/trade flow and 600 s of book history",
)

# ---------------------------------------------------------------- the three numbers
res = load_results_table()
hold = load_json("holdout_metrics.json")
period = load_json("period_difficulty_meta.json")

cv = float(res.loc[res.model == "ensemble", "cosine_mean"].iloc[0]) if res is not None else None
ho = (hold or {}).get("scores", {}).get("cosine")

# Two models stand behind these three numbers, and the page says so rather than letting
# them read as one. The hold-out can only be measured on a model that did not train on
# months 65-70, which the single LightGBM did not; the ensemble deliberately trains
# through month 67, buying four more months and forfeiting any hold-out score. Both were
# submitted - 0.128 and 0.129 - so the comparison is external and settled.
c1, c2, c3 = st.columns(3)
c1.metric("Walk-forward CV", f"{cv:+.5f}" if cv else "-",
          help="Ensemble, averaged over 5 periods with an embargo between them. "
               "The honest internal estimate.")
c2.metric("Hold-out (months 65-70)", f"{ho:+.5f}" if ho else "-",
          delta=f"{ho - cv:+.5f} vs CV" if (ho and cv) else None,
          help="Single LightGBM, months 0-63 - the only model with these six months "
               "untouched, which is what makes the number measurable. Read once. It is "
               "also an unusually favourable period.")
lb = load_json("leaderboard.json") or {}
lb_score = lb.get("our_score", 0.129)
c3.metric("Leaderboard", f"+{lb_score:.5f}",
          delta=f"{lb_score - cv:+.5f} vs CV" if cv else None, delta_color="inverse",
          help=(f"Ensemble on months 0-67 - the externally graded number. Rank "
                f"{lb.get('our_rank', '-')} of {lb.get('n_teams', '-')}, median "
                f"{lb.get('median', 0):.3f}, captured {lb.get('captured', '-')}. The "
                "LightGBM above scored 0.128 on the same test set, so the four extra "
                "months and the blend are worth +0.001."))

def _ordinal(n: int) -> str:
    """83rd, not 83th. 11-13 are the exceptions that catch a naive suffix rule."""
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


if period:
    pct = round(period.get("holdout_percentile", 0) * 100)
    st.info(
        f"**The hold-out is listed second on purpose.** Holding the model fixed and scoring "
        f"every period in turn, difficulty swings from 0.117 to 0.148 - and months 65-70 sit "
        f"at the **{_ordinal(pct)} percentile**. De-biasing for that gives "
        f"**{period['debiased']:+.5f}**, within 0.00004 of the walk-forward mean computed a "
        f"completely different way. Purity was never the binding constraint; period "
        f"difficulty was."
    )

st.divider()

# ---------------------------------------------------------------- model comparison
left, right = st.columns([3, 2])
with left:
    st.subheader("Model comparison")
    if res is None:
        missing("Walk-forward results", "python -m src.models.train")
    else:
        show = res[["model", "cosine_mean", "cosine_std", "cosine_min", "cosine_max"]]
        # No background_gradient: pandas routes it through matplotlib, which is 40 MB of
        # dependency for one column of colour and is not in requirements.txt. It worked
        # here only because the notebooks had already installed matplotlib locally.
        st.dataframe(
            show.style.format(dict.fromkeys(
                [c for c in show.columns if c != "model"], "{:+.5f}")),
            width="stretch", hide_index=True,
        )
        st.caption(
            "`zero` and `mean` are controls, not candidates. `mean` scores NEGATIVE "
            "because cosine is not shift-invariant - a constant bias actively hurts. The "
            "two tree models differ by less than a fifth of the fold-to-fold noise "
            "(0.0041), so they are statistically indistinguishable and stability decides."
        )

with right:
    st.subheader("The noise floor")
    st.markdown(
        """
| | |
|---|---:|
| Fold-to-fold std | **0.0041** |
| Period-to-period std | **0.0091** |
| Gain from tuning | −0.0004 |
| Gain from the ensemble | +0.0022 |
| Gain from sequence shape | +0.0006 |

Every measured improvement in this project is smaller than the noise between periods.
That is the single most useful thing it learned, and it is why nothing here is reported
without an interval.
"""
    )

st.divider()

# ---------------------------------------------------------------- market state
cols = feature_columns()
st.caption(
    f"{(len(cols) - 3) if cols else 0} feature columns "
    "(264 of them distinct - see Model Performance)"
)

df = load_features(
    n_rows=20_000,
    columns=[
        "sample_id", "month", "target",
        "mkt_mid_last", "mkt_rel_spread_last", "mkt_depth_imb1_last", "mkt_mid_std_60s",
        "ord_ofi_60s", "txn_volume_imbalance_60s", "txn_intensity_60s",
    ],
)
if df is None:
    missing("Feature set", "python -m src.features.assemble")
    st.stop()

st.subheader(f"Market state ({len(df):,} samples across {df['month'].nunique()} months)")
# Two rows of three rather than one row of six. At six, each column is narrow enough that
# Streamlit ellipsises the metric VALUE - "12.1 bps" rendered as "12.1 ...", which is the
# one part of a metric that must never be abbreviated.
book = st.columns(3)
book[0].metric("Relative spread", f"{df['mkt_rel_spread_last'].mean() * 1e4:.1f} bps",
               help="Last book snapshot, averaged over the sample.")
book[1].metric("Depth imbalance", f"{df['mkt_depth_imb1_last'].mean():+.3f}",
               help="Level-1 bid vs ask volume. Positive means the bid side is heavier.")
book[2].metric("Mid volatility (60s)", f"{df['mkt_mid_std_60s'].mean() * 1e4:.1f} bps")

flow = st.columns(3)
flow[0].metric("Order flow imbalance", f"{df['ord_ofi_60s'].mean():+.3f}",
               help="Signed order arrivals less cancellations over the 60 s window.")
flow[1].metric("Trade intensity", f"{df['txn_intensity_60s'].mean():.2f} /s")
flow[2].metric("Target std", f"{df['target'].std() * 1e4:.1f} bps",
               help="Spread of the value being predicted - the scale every score is "
                    "read against.")

left, right = st.columns(2)
with left:
    st.subheader("Target distribution")
    st.bar_chart(histogram(df["target"].clip(-0.01, 0.01) * 1e4, bins=60,
                           label="target (bps)"))
    st.caption("Median is exactly 0 (5.5% exact zeros) - a tick-size artefact.")
with right:
    st.subheader("Target volatility by month")
    # From the full 1.26M-row table, not from the sample beside it. The sample carries
    # ~70 rows per month, which is too few for a monthly standard deviation - and this
    # chart used to be drawn from it, back when the sample was the first 5,000 rows and
    # so held exactly ONE month. The caption below is a claim about all 71 months, and
    # the chart has to be able to support it.
    by_month = load_csv("target_by_month.csv")
    if by_month is None:
        missing("Monthly target statistics", "python -m src.data.export_results")
    else:
        st.line_chart(by_month.set_index("month")["std"])
        swing = by_month["std"].max() / by_month["std"].min()
        st.caption(
            f"Monthly std runs {by_month['std'].min():.5f} to {by_month['std'].max():.5f} "
            f"- a {swing:.2f}x swing across the 71 months. Regime shift, and the reason "
            "fold-to-fold variation is the number to read scores against."
        )

st.divider()

# ---------------------------------------------------------------- what was found
st.subheader("What the data turned out to be")
findings = load_csv("feature_redundancy.csv")
f1, f2 = st.columns(2)
with f1:
    st.markdown(
        """
**`price = 0` is a sentinel, not a price.** Treating it as a price makes the mean relative
spread **negative** (−0.0064). Excluding empty levels flips it to **+0.0013**.

**The market window is 600 s, not 60 s.** Order and transaction span 60 s; market spans ten
times that. Assuming one window for all three silently truncates 90% of the book history.

**`side` and `order_action` are undocumented** and were recovered by measurement:
NEW(128.1M) ≈ CANCEL(42.0M) + TRADE(104.0M) closes the balance.
"""
    )
with f2:
    st.markdown(
        """
**There is no symbol column.** Every sample is self-contained, so cross-sample history is
impossible and "rolling window" means nested windows *inside* one sample.

**Each file is a single Arrow record batch.** 221.7M rows that cannot be streamed - solved
by column-group projection, peak RAM 7.92 GB instead of 11.53.
"""
    )
    if findings is not None:
        st.metric("Feature pairs correlating above 0.999", len(findings.query("abs_corr > 0.999")))
        st.caption("292 columns, 264 distinct. `make feature-audit` reproduces it.")

st.divider()

# ---------------------------------------------------------------- how it is built
# The diagram is generated from results/ by src/data/build_diagram.py, so the feature
# counts printed on it are the ones measured above rather than a picture drawn once and
# left to describe whatever the project used to be.
st.subheader("How it is built")
_diagram = Path(__file__).resolve().parents[2] / "docs" / "architecture.svg"
if _diagram.exists():
    st.image(str(_diagram), width="stretch")
else:
    missing("Architecture diagram", "make diagram")
