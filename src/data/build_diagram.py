"""Draw the architecture diagram, with its numbers read from the pipeline's own output.

A diagram is the thing most likely to go quietly wrong in a repository: it is drawn once,
usually in a tool that is not the codebase, and from then on it describes whatever the
project used to be. This one is generated, so the feature count on it is the feature count
in `results/`, and a stale figure becomes a failing test rather than a slide nobody
questions.

The fixed numbers - row counts, file sizes, peak RAM - are constants here rather than
lookups because they describe the raw competition data and the machine it was processed
on. They were measured once, they cannot change, and their provenance is in the README.

Output is a single SVG with no external fonts and no script, so it renders in a GitHub
README, in the Streamlit app, and in a projector's browser without a network round trip.
It is drawn on a light panel deliberately: GitHub serves README images through <img>, and
an image that adapts to the reader's theme is not worth the two-file <picture> dance when
a light card reads correctly against both.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "architecture.svg"

# Measured once, from the raw competition files. See README "Dataset at a glance".
RAW_ROWS = "804.5M rows"
RAW_SIZE = "9.26 GiB"
PEAK_RAM = "7.92 GB peak"
NAIVE_RAM = "11.53 GB needed"
SAMPLES = "1,257,637"

INK = "#1B2430"
MUTED = "#5B6979"
LINE = "#C3CCD8"
PANEL = "#FFFFFF"
TINT = "#F1F5FA"
ACCENT = "#0F8C7A"
WARN = "#B4632A"


def _facts() -> dict:
    """The few numbers the diagram shares with the rest of the project."""
    audit = json.loads((RESULTS / "feature_audit.json").read_text(encoding="utf-8"))
    months = json.loads((RESULTS / "period_difficulty_meta.json").read_text(
        encoding="utf-8"))
    return {
        "n_features": audit["n_features"],
        "n_effective": audit["n_effective"],
        "holdout": months.get("holdout_months", [65, 70]),
    }


def _box(x: int, y: int, w: int, h: int, title: str, lines: list[str],
         *, tint: bool = False, accent: str | None = None) -> str:
    fill = TINT if tint else PANEL
    stroke = accent or LINE
    body = "".join(
        f'<text x="{x + 14}" y="{y + 44 + i * 17}" class="s">{t}</text>'
        for i, t in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>'
        f'<text x="{x + 14}" y="{y + 24}" class="t">{title}</text>{body}'
    )


def _line(x1: int, y1: int, x2: int, y2: int) -> str:
    """A plain connector, no head - used for the bus the feature families join on."""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" '
            f'stroke-width="1.5"/>')


def _arrow(x1: int, y1: int, x2: int, y2: int, label: str = "") -> str:
    mid = f'<text x="{(x1 + x2) / 2 + 8}" y="{(y1 + y2) / 2 - 6}" class="a">{label}</text>'
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" '
        f'stroke-width="1.5" marker-end="url(#h)"/>{mid if label else ""}'
    )


def build() -> Path:
    f = _facts()
    lo, hi = f["holdout"]

    parts = [
        f'<rect width="1180" height="890" fill="{PANEL}"/>',
        '<text x="32" y="44" class="h">MSCapital — pipeline architecture</text>',
        '<text x="32" y="68" class="s">Short-horizon return forecasting from market '
        'microstructure. Every box is code in this repository.</text>',

        # ---------------------------------------------------------------- ingest
        '<text x="32" y="108" class="lbl">INGEST</text>',
        _box(32, 120, 250, 108, "Kaggle competition data",
             [RAW_ROWS + " · " + RAW_SIZE, "8 Arrow files, one record", "batch each"]),
        _arrow(282, 174, 330, 174),
        _box(330, 120, 250, 108, "Column-group converter",
             ["src/data/ingestion.py", PEAK_RAM + " vs " + NAIVE_RAM,
              "on a 16 GB machine"], accent=WARN),
        _arrow(580, 174, 628, 174),
        _box(628, 120, 250, 108, "Parquet shards → BigQuery",
             ["src/data/bq_loader.py", "8 parallel workers", "row counts verified exactly"]),
        _arrow(878, 174, 926, 174),
        _box(926, 120, 222, 108, "Staging",
             ["partitioned by month,", "clustered by sample_id", "min(seconds) = 0 → no",
              "look-ahead"], tint=True),

        # ---------------------------------------------------------------- features
        # The three families are computed in PARALLEL and joined, so they converge on a
        # bus rather than chaining. An earlier version drew all four boxes pointing
        # downwards, which read as four sequential stages and misdescribed the one design
        # decision this layer actually rests on: aggregate per sample_id first, then join,
        # because a row-level join of 221.7M x 170.1M rows is meaningless and explosive.
        '<text x="32" y="276" class="lbl">FEATURES — aggregate per sample_id, then join</text>',
        _box(32, 288, 250, 96, "Market · 114",
             ["book shape, microprice,", "realised volatility"], tint=True),
        _box(330, 288, 250, 96, "Order · 81",
             ["order-flow imbalance,", "cancellation rate"], tint=True),
        _box(628, 288, 250, 96, "Transaction · 52",
             ["signed volume, Kyle's λ,", "Amihud illiquidity"], tint=True),
        _line(157, 384, 157, 408), _line(455, 384, 455, 408),
        _line(753, 384, 753, 408), _line(157, 408, 753, 408),
        _arrow(455, 408, 455, 436),
        _box(205, 436, 500, 72, f"Assembled feature table · {f['n_features']} columns",
             [f"{SAMPLES} rows, one per sample · {f['n_effective']} distinct after the "
              "redundancy audit"], accent=ACCENT),
        _arrow(455, 508, 455, 544),

        # ---------------------------------------------------------------- validation
        '<text x="32" y="572" class="lbl">VALIDATION</text>',
        _box(32, 584, 548, 116, "Walk-forward, expanding window, 1-month embargo",
             ["5 folds over months 0–64. The embargo cuts the overlap between",
              "consecutive 60 s windows, which a random split would leak across.",
              f"Hold-out months {lo}–{hi} read ONCE. Primary metric: cosine similarity."],
             accent=ACCENT),
        _arrow(580, 642, 628, 642),
        _box(628, 584, 520, 116, "Models",
             ["Ridge (median imputation) · LightGBM · XGBoost",
              "Blend weights fitted by cosine-optimal NNLS.",
              "Selection by cosine, never by the training loss."]),

        # ---------------------------------------------------------------- serving
        '<text x="32" y="740" class="lbl">SERVING</text>',
        _box(32, 752, 355, 96, "Artefact",
             ["src/inference/ — loads without", "the training package"], accent=ACCENT),
        _arrow(387, 800, 435, 800),
        _box(435, 752, 340, 96, "FastAPI",
             ["/health /model-info /predict", "777 MB image"]),
        _arrow(775, 800, 823, 800),
        _box(823, 752, 325, 96, "Streamlit dashboard",
             ["7 pages, published with the", "results bundled — no pipeline needed"]),

        '<text x="32" y="874" class="a">Generated by src/data/build_diagram.py — the '
        'feature counts are read from results/, so this cannot drift from the pipeline.'
        '</text>',
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 890" width="1180" height="890" role="img" aria-label="MSCapital pipeline architecture">
<defs>
  <marker id="h" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/>
  </marker>
</defs>
<style>
  text {{ font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  .h {{ font-size: 26px; font-weight: 700; fill: {INK}; }}
  .t {{ font-size: 15px; font-weight: 600; fill: {INK}; }}
  .s {{ font-size: 13px; fill: {MUTED}; }}
  .a {{ font-size: 12px; fill: {MUTED}; }}
  .lbl {{ font-size: 12px; font-weight: 700; fill: {ACCENT}; letter-spacing: 1.2px; }}
</style>
{"".join(parts)}
</svg>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    log.info("wrote %s (%.1f KB)", OUT, OUT.stat().st_size / 1024)
    return OUT


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Build the architecture diagram").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()


if __name__ == "__main__":
    main()
