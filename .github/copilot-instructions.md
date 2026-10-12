# GitHub Copilot repository instructions — MS Capital Market Forecasting

## What this repo evaluates
A financial time-series model evaluated on a cosine-similarity objective with expanding walk-forward validation. Leakage and evaluation integrity matter more than producing favorable backtest or holdout scores. Never imply investment performance from test results.

## Relevant modules
- `src/evaluation/metrics.py`: cosine similarity, Pearson, MAE/RMSE, directional accuracy, LightGBM/XGBoost evaluation.
- `src/evaluation/temporal_validation.py`: chronological expanding folds, embargo between train and validation, untouched holdout months 65–70.
- `src/evaluation/backtesting.py`: trading/evaluation behavior and costs.
- `tests/test_metrics.py`, `tests/test_temporal_validation.py`: existing deterministic tests.
- `pyproject.toml`: Ruff/pytest/myPy conventions and checks.
- `.github/agents/quant-validation.agent.md`: focused, manually selected validation agent.

## Development rules
- Read implementation and adjacent tests before code edits; preserve configured folds and avoid changing benchmarks simply to make scores look better.
- Use small synthetic/seeded NumPy and pandas data. Do not access BigQuery, broker services, credentials, real market data, or long model-training jobs unless explicitly required.
- Test positive scaling invariance of cosine, directional zero handling, arithmetic metrics, and embargo/holdout integrity. Detect row/order leakage; never random-split overlapping time windows.
- To verify changes run `python -m pytest -q tests/<target_test.py>` and `ruff check tests/<target_test.py>`; lint groups are strict, so respect configured import spacing.
- Existing `pyproject.toml` is tool configuration rather than a project package manifest. Do not add a `[project]` block without an explicit need; Streamlit deployment must not confuse manifests.
- Focus on small changes. Leave README, CI workflows, generated model artifacts and parallel Developer Program work untouched unless directly requested.
- In PR summaries state the exact tests actually executed, true/false claims, affected data windows and unresolved assumptions. No auto-merge.
