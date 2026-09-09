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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

c1, c2, c3 = st.columns(3)
c1.metric("Walk-forward CV", f"{cv:+.5f}" if cv else "-",
          help="Averaged over 5 periods. The honest internal estimate.")
c2.metric("Hold-out (months 65-70)", f"{ho:+.5f}" if ho else "-",
          delta=f"{ho - cv:+.5f} vs CV" if (ho and cv) else None,
          help="Measured once on untouched data - but an unusually favourable period.")
c3.metric("Leaderboard", "+0.12900", delta=f"{0.129 - cv:+.5f} vs CV" if cv else None,
          delta_color="inverse", help="The only externally graded number. 187 teams, "
                                      "median 0.138.")

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
        st.dataframe(
            show.style.format({c: "{:+.5f}" for c in show.columns if c != "model"})
                .background_gradient(subset=["cosine_mean"], cmap="Greens"),
            use_container_width=True, hide_index=True,
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

st.subheader(f"Market state ({len(df):,} samples)")
k = st.columns(6)
k[0].metric("Spread (rel)", f"{df['mkt_rel_spread_last'].mean() * 1e4:.1f} bps")
k[1].metric("Depth imbalance", f"{df['mkt_depth_imb1_last'].mean():+.3f}")
k[2].metric("Volatility (60s)", f"{df['mkt_mid_std_60s'].mean() * 1e4:.1f} bps")
k[3].metric("Order flow imbalance", f"{df['ord_ofi_60s'].mean():+.3f}")
k[4].metric("Trade intensity", f"{df['txn_intensity_60s'].mean():.2f}/s")
k[5].metric("Target std", f"{df['target'].std() * 1e4:.1f} bps")

left, right = st.columns(2)
with left:
    st.subheader("Target distribution")
    st.bar_chart(histogram(df["target"].clip(-0.01, 0.01) * 1e4, bins=60,
                           label="target (bps)"))
    st.caption("Median is exactly 0 (5.5% exact zeros) - a tick-size artefact.")
with right:
    st.subheader("Target volatility by month")
    st.line_chart(df.groupby("month")["target"].std())
    st.caption("Across full train the monthly std swings by 2.69x - regime shift.")

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
