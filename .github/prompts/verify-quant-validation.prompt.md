---
description: Read-only source-backed acceptance for Quant Validation agent
---

# Quant Validation — custom agent acceptance

Read-only source acceptance. Explicit agent selection can be through GitHub Copilot CLI `--agent=quant-validation` or a supported IDE agent selector. The external harness, not this answer, must record requested agent ID and CLI exit code. Do not label NOT RUN simply because the IDE was not used. If the actually loaded runtime profile is not independently confirmed, label agent identity UNVERIFIED separately from source review.

Read these checked-in files, recording any that are absent:
- `README.md`
- `src/config.py`
- `src/evaluation/temporal_validation.py`
- `src/evaluation/metrics.py`
- `src/evaluation/backtesting.py`
- `tests/test_temporal_validation.py`
- `tests/test_metrics.py`
- `tests/test_backtesting.py`

Answer these four checks based ONLY on inspected source:
1. Trace chronological train/embargo/validation and protected final holdout months in temporal validation; explain what assert_no_overlap actually guarantees.
2. Review cosine similarity, MAE/RMSE, directional accuracy and zero-label handling. Propose a small independently computed synthetic test for a boundary case.
3. Explain risk of ties, lookahead, transaction costs and train/test contamination in signal generation/backtests; identify code/test protections and unverified gaps.
4. Distinguish offline source/test expectations from investment performance, live broker trades, BigQuery datasets or fresh market results.

For every check, return PASS/FAIL/UNVERIFIED and concrete source-path/function/test evidence. Clearly label source reasoning versus unexecuted tests, CI, datasets, deployment or live systems. No edits, code execution, shell commands, CI/Actions/CircleCI/Vercel, installs, external APIs, private data or secrets. No orders, portfolio recommendations, training, live market data, or benchmark score fabrication.
