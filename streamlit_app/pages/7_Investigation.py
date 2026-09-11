"""Why the leaderboard disagreed - the investigation, on one page.

This is the project's strongest material and it was not in the dashboard at all. A
forecast was recorded before submitting, it was wrong, and six hypotheses were then tested
against the gap. Five were eliminated. The page shows the eliminations, not just the
survivor, because the eliminations are what took the work.
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

import pandas as pd
import streamlit as st

from streamlit_app.lib import load_csv, load_json, missing, page_header

st.set_page_config(page_title="MSCapital · Investigation", page_icon="📈", layout="wide")
page_header(
    "Why the leaderboard disagreed",
    "One forecast, recorded before the answer. Six hypotheses. Four falsified.",
)

st.markdown(
    """
The hold-out said **+0.15171**. Before submitting, a falsifiable forecast was recorded:
reweighting by the test set's spread mix gives **≈ 0.143**, with the explicit caveat that
anything materially below would need a different explanation.

**The leaderboard said 0.128.** The forecast was wrong - right direction, magnitude off by
three. What follows is what was done about it.
"""
)

st.divider()

# ------------------------------------------------------------------ the scoreboard
st.subheader("Six hypotheses")
st.dataframe(
    pd.DataFrame([
        {"Hypothesis": "The hold-out was a lucky period",
         "Method": "fix the model, score every period in turn",
         "Verdict": "CONFIRMED - explains 46%"},
        {"Hypothesis": "Spread-regime mix shift",
         "Method": "reweight the hold-out by the test spread mix",
         "Verdict": "real, corrected DOWN to ~14%"},
        {"Hypothesis": "High-drift features hurt under shift",
         "Method": "prune them; vary the train-to-eval gap over 32 months",
         "Verdict": "falsified at two thresholds"},
        {"Hypothesis": "Skill decays with elapsed time",
         "Method": "same experiment, control arm",
         "Verdict": "falsified - the slope is POSITIVE"},
        {"Hypothesis": "The test set is categorically different",
         "Method": "adversarial validation, calibrated against within-train distance",
         "Verdict": "falsified - it is a continuation"},
        {"Hypothesis": "Sequence order carries missing signal",
         "Method": "18 hand-built path statistics added on top, paired",
         "Verdict": "those statistics falsified - +0.0006, CI spans zero"},
    ]),
    width="stretch", hide_index=True,
    # Explicit widths, because the verdict is the column a reader actually scans and it
    # was the one being ellipsised - "CI spans zer" is worse than no table.
    column_config={
        "Hypothesis": st.column_config.TextColumn(width="medium"),
        "Method": st.column_config.TextColumn(width="large"),
        "Verdict": st.column_config.TextColumn(width="medium"),
    },
)

st.divider()

# ------------------------------------------------------------------ period luck
st.subheader("The one that survived: the hold-out was a lucky draw")
per = load_csv("period_difficulty.csv")
pm = load_json("period_difficulty_meta.json")
if per is None or pm is None:
    missing("Period difficulty", "make period-diff")
else:
    a, b = st.columns([3, 2])
    with a:
        st.bar_chart(per.set_index("block")["cosine"])
        st.caption(
            "One model, trained once on months 0-34, scored on every later period. "
            "Training set, features, seeds and rounds are identical across blocks, so the "
            "spread is period difficulty and nothing else."
        )
    with b:
        st.metric("Hold-out months 65-70", f"{pm['holdout_cosine']:+.5f}")
        st.metric("Median of other blocks", f"{pm['typical_cosine']:+.5f}")
        st.metric("Hold-out percentile", f"{pm['holdout_percentile'] * 100:.0f}%")
        st.success(
            f"De-biased: **{pm['debiased']:+.5f}**\n\n"
            f"Walk-forward mean: **+0.14088**\n\n"
            "Two routes, computed completely differently, agreeing to 0.00004. Nothing was "
            "tuned to make them meet."
        )

st.divider()

# ------------------------------------------------------------------ the metric itself
st.subheader("And the forecast was built wrong")
dec = load_csv("cosine_decomposition.csv")
if dec is None:
    missing("Cosine decomposition", "make cosine-decomp")
else:
    st.markdown(
        "Cosine factors exactly over any partition: "
        "`cos(y,p) = Σ cos_g · w_g` with `w_g = ‖y_g‖‖p_g‖ / (‖y‖‖p‖)`. "
        "Subgroups are weighted by **magnitude**, not row count - and the forecast used "
        "sample shares."
    )
    show = dec[["group", "n", "cosine", "count_weight", "weight"]]
    st.dataframe(
        show.style.format({"cosine": "{:+.5f}", "count_weight": "{:.3f}",
                           "weight": "{:.3f}", "n": "{:,}"}),
        width="stretch", hide_index=True,
    )
    st.warning(
        "Equal-sized quartiles, but the weights the metric applies run **0.209 to 0.312** - "
        "and the heaviest lands on the bucket where the model is *strongest*. Redone "
        "properly the forecast moves to **+0.14844**: **further** from the outcome, not "
        "closer. Fixing the error made the story worse, which is why it is worth reporting."
    )

st.divider()

# ------------------------------------------------- the metric's own geometry
st.subheader("Three free corrections, and why none of them survived")
geo = load_csv("prediction_geometry.csv")
gm = load_json("prediction_geometry_meta.json")
if geo is None or gm is None:
    missing("Prediction geometry", "make pred-geometry")
else:
    def gain_of(variant: str) -> float:
        """Pull one measured gain out of the table by name.

        Read from the exported result rather than written into the prose, so the sentences
        below cannot drift from the experiment that produced them.
        """
        return float(geo.loc[geo["variant"] == variant, "gain"].iloc[0])

    pooled = gm["best_pooled_gain"]
    honest = gm["honest_out_of_sample_gain"]
    shift = gm["shift_per_month"]

    st.markdown(
        "Cosine has structure the training loss cannot see: it is **not shift-invariant**, "
        "it weights rows by **magnitude**, and it is computed **once over the whole test "
        "set** - so relative magnitudes across regimes are part of the score. Each of "
        "those suggests a correction that needs no retraining. All three were measured."
    )

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            geo[["family", "variant", "gain", "out_of_sample"]]
                .style.format({"gain": "{:+.5f}"}),
            width="stretch", hide_index=True,
            column_config={
                "variant": st.column_config.TextColumn(width="large"),
                "out_of_sample": st.column_config.CheckboxColumn(
                    "honest?",
                    help="Unchecked means the variant was chosen on the same rows it is "
                         "scored on."),
            },
        )
    with right:
        st.metric("Best regime gain, pooled", f"{pooled:+.5f}")
        st.metric("Same correction, chosen out of sample", f"{honest:+.5f}")
        st.caption(
            f"De-meaning gains {gain_of('subtract own mean'):+.5f} pooled and improves "
            f"only {shift['months_improved']} of {shift['n_months']} months "
            f"(std {shift['std']:.5f}). An average of coin tosses."
        )

    st.warning(
        "**Every reshaping of the magnitudes loses.** Rank-transforming costs "
        f"{gain_of('rank transform'):+.5f} and keeping only the sign costs "
        f"{gain_of('sign only'):+.5f}, so the predicted magnitudes carry real information "
        "beyond their ordering - the model is already close to the best form cosine can "
        "read it in."
    )
    st.info(
        "**The regime correction is the interesting failure.** It describes something "
        "true: across volatility quartiles the target spans **1.69x** while the model "
        f"spans **1.55x**, so it does under-scale. Fitted in place that is worth "
        f"**{pooled:+.5f}** - comparable to the ensemble, and it would have made a "
        f"respectable sixth forecast. Choosing its one free parameter out of sample "
        f"leaves **{honest:+.5f}**: not smaller, gone. The pooled number was measuring "
        "the freedom to pick alpha."
    )

st.divider()

# ------------------------------------------------------------------ five overshoots
st.subheader("Five forecasts, five overshoots")
st.dataframe(
    pd.DataFrame([
        {"Forecast": "Leaderboard, from the hold-out", "Predicted": "0.143", "Actual": "0.128"},
        {"Forecast": "Spread-mix share of the gap", "Predicted": "26%", "Actual": "~14%"},
        {"Forecast": "Gain from ensemble + more data", "Predicted": "+0.0047",
         "Actual": "+0.0010"},
        {"Forecast": "Gain from sequence shape", "Predicted": "clears 0.0041",
         "Actual": "+0.0006"},
        {"Forecast": "Gain from aligning loss with metric", "Predicted": "small but positive",
         "Actual": "−0.0064"},
    ]),
    width="stretch", hide_index=True,
    column_config={
        "Forecast": st.column_config.TextColumn(width="large"),
        "Predicted": st.column_config.TextColumn(width="small"),
        "Actual": st.column_config.TextColumn(width="small"),
    },
)

st.info(
    "**The fifth verdict is narrower than it looks.** The experiment added 18 hand-built "
    "path statistics and they bought +0.0006. But the nine novel ones score **0.0057 "
    "standing alone** - the constant-mean predictor scores 0.0059 - so the summary carries "
    "almost nothing, and the test cannot separate *the path holds no signal* from "
    "*eighteen numbers were the wrong way to look at it*. Telling those apart needs a "
    "learned representation over the ~176 snapshots, which is not built here."
)

st.markdown(
    """
The ensemble row is the best-evidenced of the five. Its forecast was not only recorded in
a notebook: **the same number, as an absolute score, was typed into the Kaggle submission
description at upload time** - `0.128 + 0.0047 = 0.1327`, written as "Predicted
0.132-0.133" - before any score came back, in a field that cannot be edited afterwards.
It came back 0.129.

Different reasoning each time, the same direction of error every time - which points at one
cause rather than five mistakes. **Effects of order 0.002–0.005, measured on internal
splits, sit at this problem's resolution limit.**

The rule that survives: below roughly the fold-to-fold std, treat an internal gain as
evidence about *which* model to prefer, never as a quantity that will reach a leaderboard.

Six hypotheses tested, one confirmed at 46%. What is left is **+0.0128** of gap with no
identified cause - and how impressive that is depends entirely on what you compare it to.
Against the spread of a single period (period-to-period std **0.0091**) it is **1.4σ**, so
an unusually hard test period could account for a real part of it. Against the precision of
the difficulty estimate itself (0.0026) it is 4.9σ from zero, but that is the error on an
average, not the range one period can show.

The first comparison is the one that answers "could this just be bad luck?", so that is the
one to quote: the residual is large enough to keep looking for, and not large enough to
rule luck out. Saying so is more useful than a tidy story.
"""
)

# The standing is read from a captured file rather than written into the prose above.
# The figures quoted here were once 187 teams and a 0.138 median; by the time anyone
# checked, the public leaderboard held 204 teams and a 0.137 median. The story did not
# change and every number in it had. Prose cannot be re-measured; a snapshot can.
_lb = load_json("leaderboard.json")
if _lb:
    st.info(
        f"**Where it stands.** {_lb['n_teams']} teams, median {_lb['median']:.3f}, this "
        f"model {_lb['our_score']:.3f} — **rank {_lb['our_rank']}**, with "
        f"{_lb['teams_tied_with_us']} teams tied on the same score. The best is "
        f"{_lb['best']:.3f} and the upper quartile begins at {_lb['p75']:.3f}, so the "
        "problem is *not* at its noise ceiling: other people extract signal this pipeline "
        f"does not.\n\nPublic leaderboard, captured {_lb['captured']}."
    )
else:
    missing("Leaderboard standing", "results/leaderboard.json")
