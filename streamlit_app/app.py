"""MSCapital dashboard - the router.

This file used to BE the overview, which is why the sidebar's first entry read "app":
Streamlit's automatic `pages/` navigation labels the main script from its filename, and
the filename is fixed by the deployment. Declaring the pages explicitly fixes the label
and buys three things the automatic version cannot do - grouping, icons, and an order that
is not encoded in numeric filename prefixes.

The grouping is the argument of the project in miniature: what the data turned out to be,
what the model does, and then why the leaderboard disagreed with the hold-out. A reader
who follows the sidebar top to bottom gets the story in the order it happened.

Each page is still a self-contained script and still runs on its own - `streamlit run
streamlit_app/pages/7_Investigation.py` works, which is what `scripts/check_deploy.py`
relies on to render every page without a browser.
"""
import sys
from pathlib import Path

# Put the repository root on sys.path before importing anything from it.
#
# `python -m streamlit` silently adds the working directory; a bare `streamlit run` - which
# is what Streamlit Community Cloud executes - adds the MAIN SCRIPT'S directory instead. So
# `streamlit_app/` lands on the path and the repository root does not, and every
# `from streamlit_app.lib import ...` in the pages below fails with ModuleNotFoundError. It
# works locally and breaks on deploy, which is the worst place to find out.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

PAGES = Path(__file__).parent / "pages"


def page(filename: str, title: str, icon: str, *, default: bool = False):
    """Absolute paths, because the working directory is not ours to assume.

    Streamlit Community Cloud runs `streamlit run streamlit_app/app.py` from the
    repository root; a local run may start anywhere. A relative page path resolves
    against the working directory and would break in one of those two.
    """
    return st.Page(PAGES / filename, title=title, icon=icon, default=default)


st.navigation({
    "Summary": [
        page("1_Overview.py", "Overview", ":material/insights:", default=True),
    ],
    "The data": [
        page("2_Microstructure.py", "Microstructure", ":material/candlestick_chart:"),
    ],
    "The model": [
        page("4_Model_Performance.py", "Model performance", ":material/query_stats:"),
        page("5_Explainability.py", "Explainability", ":material/lightbulb:"),
        page("3_Predictions.py", "Prediction", ":material/bolt:"),
        page("6_Backtesting.py", "Backtesting", ":material/timeline:"),
    ],
    "The investigation": [
        page("7_Investigation.py", "Why the leaderboard disagreed", ":material/search:"),
    ],
}).run()
