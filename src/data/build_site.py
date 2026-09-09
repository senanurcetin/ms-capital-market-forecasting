"""Build a self-contained static site from the exported results.

WHY STATIC

The dashboard is a Streamlit app, and Streamlit is a long-lived server holding a WebSocket
per viewer. Vercel runs serverless functions, so it cannot host one - that is a platform
fact rather than a configuration problem.

But nothing on these pages actually needs a server. The numbers change only when the
pipeline is re-run, and between runs they are constant. A static page serves them faster,
never cold-starts, never sleeps, and costs nothing indefinitely - which is a better fit for
a portfolio link than a container that spins down after a week of no traffic.

What it gives up is live prediction, which genuinely needs a process. The Streamlit app
keeps that; this is the always-on companion, not a replacement.

NO BUILD STEP, NO DEPENDENCIES

Charts are emitted as inline SVG computed here in Python. The page loads no JavaScript and
no CSS from anywhere, so there is nothing to install, nothing to version, and nothing that
can break when a CDN changes. It is one file.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SITE = ROOT / "site"
REPO = "https://github.com/senanurcetin/ms-capital-market-forecasting"

INK = "#e8eaed"
MUTED = "#9aa0a6"
BG = "#0e1117"
PANEL = "#161a23"
ACCENT = "#4c9aff"
WARN = "#f2994a"
GOOD = "#3ecf8e"


def _read_csv(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def _read_json(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ----------------------------------------------------------------- inline SVG charts

def _tick(v: float) -> str:
    """Axis label that survives small magnitudes.

    A fixed 3-decimal format prints every SHAP value as "0.000" - the importances are of
    order 1e-5 - which makes the axis decorative rather than informative.
    """
    if v == 0:
        return "0"
    return f"{v:.3g}" if abs(v) < 0.01 or abs(v) >= 1000 else f"{v:.3f}"


def bar_svg(labels: list[str], values: list[float], *, width: int = 720, height: int = 240,
            colour: str = ACCENT, highlight: int | None = None) -> str:
    """A horizontal-axis bar chart, drawn directly.

    Bars are scaled from zero rather than from the minimum. Starting an axis at the
    smallest value exaggerates small differences, which is exactly the mistake this
    project spent five experiments not making.
    """
    pad_l, pad_b, pad_t = 46, 34, 10
    plot_w, plot_h = width - pad_l - 10, height - pad_b - pad_t
    top = max(values) * 1.1 if values else 1
    bw = plot_w / max(len(values), 1)
    parts = []
    for i, (lab, v) in enumerate(zip(labels, values, strict=False)):
        h = max(0.0, v / top) * plot_h
        x = pad_l + i * bw + bw * 0.15
        y = pad_t + plot_h - h
        fill = WARN if i == highlight else colour
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw * 0.7:.1f}" height="{h:.1f}" '
            f'fill="{fill}" rx="2"><title>{html.escape(str(lab))}: {v:.5f}</title></rect>'
        )
        parts.append(
            f'<text x="{x + bw * 0.35:.1f}" y="{height - 12}" fill="{MUTED}" '
            f'font-size="10" text-anchor="middle">{html.escape(str(lab))}</text>'
        )
    for frac in (0, 0.5, 1.0):
        y = pad_t + plot_h - frac * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - 10}" y2="{y:.1f}" '
                     f'stroke="#2a2f3a" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 3:.1f}" fill="{MUTED}" font-size="10" '
                     f'text-anchor="end">{_tick(top * frac)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">'
            + "".join(parts) + "</svg>")


def line_svg(values: list[float], *, width: int = 720, height: int = 200,
             colour: str = GOOD, max_points: int = 400) -> str:
    """A single path. Used for the equity curve, which is a shape rather than a table.

    Thinned to at most `max_points`, because the chart is 720 px wide and drawing 2,000
    points into it writes more coordinates than there are pixels to show them - the shape
    is identical and the file is several times smaller.
    """
    if not values:
        return ""
    if len(values) > max_points:
        step = len(values) / max_points
        values = [values[min(int(i * step), len(values) - 1)] for i in range(max_points)]
    pad_l, pad_b, pad_t = 46, 24, 10
    plot_w, plot_h = width - pad_l - 10, height - pad_b - pad_t
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    pts = [
        f"{pad_l + i / (len(values) - 1) * plot_w:.0f},"
        f"{pad_t + plot_h - (v - lo) / span * plot_h:.0f}"
        for i, v in enumerate(values)
    ]
    grid = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h - f * plot_h:.1f}" x2="{width - 10}" '
        f'y2="{pad_t + plot_h - f * plot_h:.1f}" stroke="#2a2f3a"/>'
        f'<text x="{pad_l - 6}" y="{pad_t + plot_h - f * plot_h + 3:.1f}" fill="{MUTED}" '
        f'font-size="10" text-anchor="end">{lo + f * span:.2f}</text>'
        for f in (0, 0.5, 1.0)
    )
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">{grid}'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colour}" '
            f'stroke-width="1.6"/></svg>')


# ----------------------------------------------------------------- page pieces

# Truncation loses exactly the part that distinguishes these names: cut
# "ord_new_count_imbalance_30s" to twelve characters and three different features all
# read "new_count_im". Abbreviating the common words keeps the window suffix.
_ABBREV = {
    "imbalance": "imb", "count": "cnt", "return": "ret", "volume": "vol",
    "microprice": "micro", "depth": "dep", "aggressor": "aggr", "transaction": "txn",
    "spread": "sprd", "snapshot": "snap",
}


def _short(name: str, limit: int = 15) -> str:
    for prefix in ("mkt_", "ord_", "txn_"):
        name = name.removeprefix(prefix)
    for long, brief in _ABBREV.items():
        name = name.replace(long, brief)
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _metric(label: str, value: str, note: str = "", colour: str = INK) -> str:
    return (f'<div class="metric"><div class="label">{html.escape(label)}</div>'
            f'<div class="value" style="color:{colour}">{html.escape(value)}</div>'
            f'<div class="note">{note}</div></div>')


def _table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append(f"<td>{html.escape(fmt[c].format(v) if c in fmt else str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def build() -> Path:
    res = _read_csv("walkforward_summary.csv")
    hold = _read_json("holdout_metrics.json") or {}
    period_meta = _read_json("period_difficulty_meta.json") or {}
    period = _read_csv("period_difficulty.csv")
    equity = _read_csv("backtest_equity.csv")
    shap = _read_csv("shap_global.csv")
    dec = _read_csv("cosine_decomposition.csv")

    cv = float(res.loc[res.model == "ensemble", "cosine_mean"].iloc[0]) if res is not None else 0
    ho = hold.get("scores", {}).get("cosine", 0)

    hyps = pd.DataFrame([
        ("The hold-out was a lucky period", "fix the model, score every period in turn",
         "CONFIRMED — explains 46%"),
        ("Spread-regime mix shift", "reweight the hold-out by the test spread mix",
         "real, corrected DOWN to ~14%"),
        ("High-drift features hurt under shift", "prune them; vary the train→eval gap",
         "falsified at two thresholds"),
        ("Skill decays with elapsed time", "same experiment, control arm",
         "falsified — the slope is POSITIVE"),
        ("The test set is categorically different", "adversarial validation, calibrated",
         "falsified — it is a continuation"),
        ("Sequence order carries missing signal", "18 path statistics added on top, paired",
         "falsified — +0.0006, CI spans zero"),
    ], columns=["Hypothesis", "Method", "Verdict"])

    forecasts = pd.DataFrame([
        ("Leaderboard, from the hold-out", "0.143", "0.128"),
        ("Spread-mix share of the gap", "26%", "~14%"),
        ("Gain from ensemble + more data", "+0.0047", "+0.0010"),
        ("Gain from sequence shape", "clears 0.0041", "+0.0006"),
        ("Gain from aligning loss with metric", "small but positive", "−0.0064"),
    ], columns=["Forecast", "Predicted", "Actual"])

    parts: list[str] = []
    parts.append(f"""
<header>
  <h1>MSCapital — Real Financial Market Forecasting</h1>
  <p class="sub">Short-horizon return prediction from market microstructure.
     804.5M raw rows reduced to a scored model on a 16 GB laptop.</p>
  <p class="links"><a href="{REPO}">Source on GitHub</a> ·
     <a href="{REPO}/blob/main/notebooks/05_why_the_leaderboard_disagreed.ipynb">The investigation</a></p>
</header>""")

    parts.append(f"""
<section>
  <h2>Three numbers, not one</h2>
  <div class="metrics">
    {_metric("Walk-forward CV", f"{cv:+.5f}", "averaged over 5 periods — the honest estimate", ACCENT)}
    {_metric("Hold-out (65–70)", f"{ho:+.5f}", "measured once — but a <em>lucky</em> period", WARN)}
    {_metric("Leaderboard", "+0.12900", "the only externally graded number", INK)}
  </div>
  <p class="callout">The hold-out is listed second on purpose. Holding the model fixed and
  scoring every period in turn, difficulty swings from 0.117 to 0.148 — and months 65–70 sit
  at the <strong>83rd percentile</strong>. De-biasing gives
  <strong>{period_meta.get('debiased', 0):+.5f}</strong>, within 0.00004 of the walk-forward
  mean computed a completely different way. Purity was never the binding constraint; period
  difficulty was.</p>
</section>""")

    if period is not None:
        hl = int(period.cosine.idxmax())
        parts.append(f"""
<section>
  <h2>Period difficulty — one model, every period</h2>
  {bar_svg(period.block.tolist(), period.cosine.tolist(), highlight=hl)}
  <p class="cap">Trained once on months 0–34, then scored on every later block. Training
  set, features, seeds and rounds are identical, so the spread is period difficulty and
  nothing else.</p>
</section>""")

    if res is not None:
        parts.append(f"""
<section>
  <h2>Model comparison</h2>
  {_table(res[["model", "cosine_mean", "cosine_std", "cosine_min", "cosine_max"]],
          {c: "{:+.5f}" for c in ("cosine_mean", "cosine_std", "cosine_min", "cosine_max")})}
  <p class="cap"><code>zero</code> and <code>mean</code> are controls. <code>mean</code>
  scores NEGATIVE because cosine is not shift-invariant — a constant bias actively hurts.
  The two tree models differ by less than a fifth of the fold-to-fold noise (0.0041).</p>
</section>""")

    parts.append(f"""
<section>
  <h2>Why the leaderboard disagreed</h2>
  <p>The hold-out said +0.15171. Before submitting, a falsifiable forecast was recorded:
  <strong>≈ 0.143</strong>. The leaderboard said <strong>0.128</strong>. Six hypotheses
  were then tested against the gap.</p>
  {_table(hyps)}
</section>

<section>
  <h2>Five forecasts, five overshoots</h2>
  {_table(forecasts)}
  <p class="cap">Different reasoning each time, the same direction of error every time —
  which points at one cause rather than five mistakes. Effects of order 0.002–0.005,
  measured on internal splits, sit at this problem's resolution limit: fold-to-fold std is
  0.0041, period-to-period std 0.0091.</p>
</section>""")

    if dec is not None:
        parts.append(f"""
<section>
  <h2>And the forecast was built wrong</h2>
  <p>Cosine factors exactly over any partition:
  <code>cos(y,p) = Σ cos_g · w_g</code> with
  <code>w_g = ‖y_g‖‖p_g‖ / (‖y‖‖p‖)</code> — subgroups are weighted by
  <strong>magnitude</strong>, not row count. The forecast used sample shares.</p>
  {_table(dec.loc[dec.group.notna(), ["group", "n", "cosine", "count_weight", "weight"]],
          {"cosine": "{:+.5f}", "count_weight": "{:.3f}", "weight": "{:.3f}", "n": "{:,}"})}
  <p class="cap">Samples with no measurable spread form a fifth bucket, left out here
  because it is a data condition rather than a liquidity regime. Equal-sized quartiles, but
  the weights the metric applies run 0.209–0.312,
  and the heaviest lands where the model is <em>strongest</em>. Redone properly the forecast
  moves to +0.14844 — <strong>further</strong> from the outcome. Fixing the error made the
  story worse, which is why it is worth reporting.</p>
</section>""")

    if equity is not None and "equity" in equity.columns:
        parts.append(f"""
<section>
  <h2>Backtest — hold-out only</h2>
  {line_svg(equity["equity"].tolist())}
  <p class="cap">20% traded at 1 bps: +8.0 bps per trade, win rate 56.5%, per-trade Sharpe
  0.168, max drawdown −9.2%. This measures ranking power, <strong>not</strong> a strategy.</p>
</section>""")

    if shap is not None:
        col = "mean_abs_shap" if "mean_abs_shap" in shap.columns else shap.columns[1]
        top = shap.nlargest(12, col)
        parts.append(f"""
<section>
  <h2>What the model leans on</h2>
  {bar_svg([_short(f) for f in top.iloc[:, 0]], top[col].tolist())}
  <p class="cap">TreeSHAP over the hold-out. The strongest signals are imbalance features,
  which is what microstructure theory predicts.</p>
</section>""")

    parts.append(f"""
<section>
  <h2>Where this stands</h2>
  <p>187 teams, median 0.138, this model 0.129 — below typical, so the problem is
  <em>not</em> at its noise ceiling. Six hypotheses tested, one confirmed at 46%, and the
  rest of the gap is real (5σ above the test set's own period noise) and still
  unidentified. Saying so is more useful than a tidy story.</p>
  <p class="cap">This page is static and shows results only. The interactive dashboard —
  live prediction, microstructure explorer, drift analysis — is a Streamlit app in the
  repository: <code>make streamlit</code>.</p>
  <p class="links"><a href="{REPO}">github.com/senanurcetin/ms-capital-market-forecasting</a></p>
</section>
<footer>For research and model evaluation. Not investment advice.
MIT licensed; competition data is not redistributed.</footer>""")

    css = f"""
*{{box-sizing:border-box}}
body{{margin:0;background:{BG};color:{INK};
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  padding:40px 20px}}
main{{max-width:860px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}}
h2{{font-size:17px;margin:0 0 14px;color:{ACCENT};font-weight:600}}
.sub{{color:{MUTED};margin:0 0 10px;max-width:60ch}}
.links a{{color:{ACCENT};text-decoration:none}}
.links a:hover{{text-decoration:underline}}
header{{border-bottom:1px solid #2a2f3a;padding-bottom:22px;margin-bottom:8px}}
section{{background:{PANEL};border:1px solid #232833;border-radius:10px;
  padding:22px;margin:18px 0}}
.metrics{{display:flex;gap:14px;flex-wrap:wrap}}
.metric{{flex:1;min-width:180px;background:{BG};border:1px solid #232833;
  border-radius:8px;padding:14px}}
.metric .label{{color:{MUTED};font-size:12px;text-transform:uppercase;
  letter-spacing:.06em}}
.metric .value{{font-size:26px;font-weight:600;margin:4px 0;
  font-variant-numeric:tabular-nums}}
.metric .note{{color:{MUTED};font-size:12px}}
.callout{{background:{BG};border-left:3px solid {ACCENT};padding:12px 14px;
  margin:16px 0 0;border-radius:0 6px 6px 0;color:#c9cdd3;font-size:14px}}
.cap{{color:{MUTED};font-size:13px;margin:12px 0 0}}
table{{width:100%;border-collapse:collapse;font-size:13px;
  font-variant-numeric:tabular-nums}}
th{{text-align:left;color:{MUTED};font-weight:600;border-bottom:1px solid #2a2f3a;
  padding:8px 10px;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
td{{padding:8px 10px;border-bottom:1px solid #1d222c}}
tr:last-child td{{border-bottom:none}}
code{{background:{BG};padding:1px 5px;border-radius:4px;font-size:.9em;color:#c9cdd3}}
footer{{color:{MUTED};font-size:12px;text-align:center;padding:24px 0 0}}
@media(max-width:640px){{body{{padding:24px 12px}}.metric{{min-width:100%}}
  table{{font-size:12px}}}}
"""
    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MSCapital — Real Financial Market Forecasting</title>
<meta name="description" content="Short-horizon return prediction from market
microstructure: 804.5M rows, six data findings, and an investigation into why the
leaderboard disagreed with the hold-out.">
<style>{css}</style>
</head><body><main>{"".join(parts)}</main></body></html>"""

    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "index.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote %s (%.0f KB)", out, out.stat().st_size / 1024)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the static results site")
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()


if __name__ == "__main__":
    main()
