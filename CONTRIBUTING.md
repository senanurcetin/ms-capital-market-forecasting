# Contributing

This is a research project whose value is that its claims are checkable, so the rules below are
about keeping them that way.

## Setup

```bash
pip install -r requirements-dev.txt
pre-commit install        # optional: ruff + mypy + file hygiene on every commit
make check                # lint + typecheck + tests
make deploy-check         # the dashboard on its runtime dependencies ALONE (slow, ~2 min)
make demo                 # the whole pipeline on synthetic data, ~15 s
```

`make check` needs no credentials, no BigQuery and no competition data. If a change needs any of
those to be tested, the change is probably testing the wrong thing.

## What a change has to do

- **Pass CI**: `ruff`, `mypy` (the file set is in `pyproject.toml`), the tests with the coverage
  floor, the dashboard on the runtime requirements, the Docker build, and the demo artefact served
  by the API image. Never skip, disable or quarantine a test to get green.
- **Test what it claims.** The test suite injects the failures it says it prevents; a new guard
  needs a test that fails without it. Where practical, check that the test fails against the old
  behaviour before trusting it.
- **Keep numbers generated, not typed.** A figure in the README or `MODEL_CARD.md` has to match
  `results/`, and `tests/test_documented_numbers.py` / `tests/test_model_card.py` enforce it. If a
  re-run moves a number, update the prose in the same change.
- **State what was not measured.** An experiment that has not been run on the real data says so
  (see `make prune`). A decision rule is written down *before* the run, not fitted afterwards.
- **No secrets, no data.** `.gitignore` excludes keys and the competition data; do not add either.
  `results/` carries derived aggregates only.

## Dependencies

Every requirements file is pinned exactly. Dependabot proposes updates, grouped (dev tools,
dashboard, serving, cloud clients), and CI decides. The numerics stack - numpy, pandas, scipy,
scikit-learn, lightgbm, xgboost, pyarrow, joblib, shap, optuna - is excluded on purpose:
`results/` and the shipped artefact were produced with those exact versions, and a bump can change
predictions or fail to load a saved booster. To move one, bump the pin yourself, re-run
`make ship`, and compare the scores before and after.

## Refreshing results

`make export-results` rebuilds `results/` from a live pipeline run, and the leaderboard snapshot
(`results/leaderboard.json`) is captured, dated and read from by every standing quoted in the
repository. Do not hand-edit either.

## Style

`ruff` is the formatter and linter (config in `pyproject.toml`). Comments explain *why*, usually a
failure that actually happened. Match the surrounding code.

## Reporting a security problem

Do not open a public issue; see [SECURITY.md](SECURITY.md).
