---
description: Read-only source-backed acceptance for Quant Bugfix agent
---

# Quant Bugfix — custom agent acceptance

Read-only source acceptance. Explicit agent selection can be through GitHub Copilot CLI `--agent=quant-bugfix` or a supported IDE agent selector. The external harness, not this answer, must record requested agent ID and CLI exit code. Do not label NOT RUN simply because the IDE was not used. If the actually loaded runtime profile is not independently confirmed, label agent identity UNVERIFIED separately from source review.

Read these checked-in files, recording any that are absent:
- `src/config.py`
- `src/evaluation/temporal_validation.py`
- `src/evaluation/backtesting.py`
- `src/evaluation/metrics.py`
- `tests/test_backtesting.py`
- `tests/test_metric_boundary_contracts.py`
- `tests/test_split_order.py`

Answer these four checks based ONLY on inspected source:
1. Identify a specific evaluator or backtest boundary where ties, nonfinite predictions, threshold/quantile selection or shape mismatch deserve a deterministic synthetic negative test.
2. Explain how final holdout months and embargo ordering must remain untouched by a proposed repair, citing temporal validation contracts.
3. Propose a minimal failing-test-first source fix for the chosen condition, including expected invariant, but do not implement it in this acceptance probe.
4. Identify which existing tests are relevant and why neither a proposed test nor inspected source establishes real-market profitability or a successful training run.

For every check, return PASS/FAIL/UNVERIFIED and concrete source-path/function/test evidence. Clearly label source reasoning versus unexecuted tests, CI, datasets, deployment or live systems. No edits, code execution, shell commands, CI/Actions/CircleCI/Vercel, installs, external APIs, private data or secrets. No production code edit, live trading, training, BigQuery or paid services.
