"""The forecast tally appears on three surfaces, and they have to agree.

"Five forecasts, five overshoots" is the project's central claim, so the number in the
heading is load-bearing: it is the evidence that a pattern exists rather than a few
scattered misses. Three places state it - the README, the dashboard's investigation page,
and the static site - and each keeps its own copy of the rows.

They drifted twice, in opposite directions:

  * the README heading said "Three forecasts, three overshoots" above five rows, and closed
    with "rather than three mistakes". Rows had been added over time; the heading had not
    followed.
  * a sixth row was then added - the ensemble's submission description, "Predicted
    0.132-0.133" against an actual 0.129 - which is the SAME forecast as the row already
    there, expressed as an absolute score instead of a gain: 0.128 + 0.0047 = 0.1327, and
    0.129 - 0.128 = +0.0010. Counting it twice inflates the very pattern the table exists
    to establish.

Both are silent failures. Nothing errors; the claim just stops matching its evidence.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PAGE = ROOT / "streamlit_app" / "pages" / "7_Investigation.py"
SITE_BUILDER = ROOT / "src" / "data" / "build_site.py"

WORDS = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def heading_counts(text: str) -> list[int]:
    """Every "<word> forecasts, <word> overshoots" claim, as numbers."""
    return [
        WORDS[m.group(1).lower()]
        for m in re.finditer(r"(\w+) forecasts?, \w+ overshoots?", text, re.I)
        if m.group(1).lower() in WORDS
    ]


def readme_rows() -> int:
    """Rows of the markdown table that follows the forecast heading."""
    body = README.read_text(encoding="utf-8")
    start = re.search(r"### \w+ forecasts?, \w+ overshoots?", body, re.I)
    assert start, "the README no longer has a forecast heading"
    rows = 0
    for line in body[start.end():].splitlines():
        if line.startswith("|"):
            if not re.match(r"^\|[\s:|-]+\|$", line):        # skip the separator rule
                rows += 1
        elif rows:
            break
    return rows - 1                                          # the header row


def _literal_rows(path: Path, kind) -> int:
    """Count entries in the first list-of-`kind` literal that mentions a forecast."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.List) and node.elts
                and isinstance(node.elts[0], kind)):
            continue
        dumped = ast.dump(node)
        if "Forecast" in dumped or "Leaderboard, from the hold-out" in dumped:
            return len(node.elts)
    raise AssertionError(f"no forecast table found in {path.name}")


def test_every_surface_lists_the_same_number_of_forecasts():
    """One table, three copies. A row added to one and not the others is invisible."""
    counts = {
        "README": readme_rows(),
        "dashboard": _literal_rows(PAGE, ast.Dict),
        "static site": _literal_rows(SITE_BUILDER, ast.Tuple),
    }
    assert len(set(counts.values())) == 1, f"the forecast tables disagree: {counts}"


def test_the_headings_match_the_rows_they_introduce():
    """"Five forecasts" over six rows is a claim its own table contradicts.

    The notebook is deliberately excluded: it states a running tally at several points in
    the narrative ("three forecasts" partway through), which is chronology rather than
    drift.
    """
    rows = readme_rows()
    for name, text in (("README", README.read_text(encoding="utf-8")),
                       ("dashboard", PAGE.read_text(encoding="utf-8")),
                       ("static site", SITE_BUILDER.read_text(encoding="utf-8"))):
        for claimed in heading_counts(text):
            assert claimed == rows, (
                f"{name} claims {claimed} forecasts but the table carries {rows}"
            )


def test_no_forecast_is_counted_twice():
    """The ensemble forecast exists as a gain (+0.0047) and, in the submission
    description, as an absolute score (0.132-0.133). They are one prediction.

    Pinned by arithmetic rather than by a banned string, so the equivalence stays visible:
    0.128 + 0.0047 rounds to 0.133, and the gap between the tabled gain and the leaderboard
    move is exactly the same number.
    """
    baseline, predicted_gain, realised = 0.128, 0.0047, 0.129
    assert round(baseline + predicted_gain, 3) == 0.133
    assert round(realised - baseline, 4) == 0.0010

    body = README.read_text(encoding="utf-8")
    table = body[re.search(r"### \w+ forecasts?", body, re.I).end():]
    table = table[:table.index("\n\n", table.index("|"))]
    assert "0.132" not in table, (
        "the ensemble forecast appears twice - once as a gain and once as a score"
    )
