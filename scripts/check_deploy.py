"""Run every dashboard page the way the deployment runs it, and fail on any exception.

This exists because of a defect that got all the way to production. `background_gradient`
reaches for matplotlib inside pandas at RENDER time. Nothing imported matplotlib, so no
import check could see it; it was installed locally from the pipeline work, so no local
run could see it either. Streamlit Community Cloud installs requirements.txt into an empty
interpreter, and every page raised ImportError there while the full test suite passed
here.

The lesson is narrow and worth stating: a clean-CLONE test verifies the repository, not
the environment. Two things had to be checked separately - that the code is all committed,
and that the code runs against the dependency set the deployment actually installs.

`AppTest` executes a page in-process, the same script run Streamlit performs, so a failure
at render time surfaces as an exception rather than as a screenshot nobody looks at. Run
under `make deploy-check`, which builds a virtualenv from requirements.txt ALONE - the CI
job is the one that matters, because a developer machine always has more installed than
the deployment does.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "streamlit_app"

# Generous: cold pages load a model bundle and a 5,000-row parquet sample.
TIMEOUT_S = 120


def main() -> int:
    pages = [APP / "app.py",
             *sorted(p for p in APP.glob("pages/*.py") if p.name != "__init__.py")]
    failures = []

    for page in pages:
        name = page.relative_to(ROOT).as_posix()
        try:
            at = AppTest.from_file(str(page), default_timeout=TIMEOUT_S).run()
        except Exception as exc:                       # noqa: BLE001 - report, don't raise
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {name}\n      {type(exc).__name__}: {exc}")
            continue

        if at.exception:
            # st.exception blocks are what a user sees on the deployed page: the traceback
            # rendered INTO the app instead of the content.
            detail = "; ".join(str(e.value) for e in at.exception)
            failures.append((name, detail))
            print(f"FAIL  {name}\n      {detail}")
        else:
            # An error banner is not an exception but still means the page is not working.
            errors = [e.value for e in at.error]
            flag = f"  ({len(errors)} st.error)" if errors else ""
            print(f"ok    {name}{flag}")

    print(f"\n{len(pages) - len(failures)}/{len(pages)} pages rendered.")
    if failures:
        print("\nThese would be broken on the deployment:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
