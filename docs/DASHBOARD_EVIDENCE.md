# MSCapital Streamlit dashboard — visual evidence and model-version boundary

**Review date:** 12 October 2026. This document records *what the repository can substantiate* about its portfolio dashboard and what requires a separate browser/inference check. It does not certify the live deployment.

## Verified repository evidence

- The committed dashboard image is [`docs/assets/mscapital-research-dashboard.png`](assets/mscapital-research-dashboard.png) (97,227 bytes in the GitHub repository listing). The [README](../README.md) describes it as a **12 October 2026 capture rendered from Streamlit source** for portfolio documentation, not an independently verified screenshot of the publicly deployed URL.
- The README describes a **seven-page Streamlit** research app; **one overview image cannot validate all seven pages**, their interactions, numerical values or the deployment's currently running commit.
- The dashboard reads derived results, model artifacts and an evenly sampled **4,970-row** feature subset from [`results/`](../results/), *not* live financial prices. Competition raw data is not redistributed.
- The [model card](../MODEL_CARD.md) distinguishes the **default served LightGBM v3** model from the **v4 blend submitted to the leaderboard**. A screenshot of the predictions page is not evidence of serving the leaderboard ensemble.

## Keep these scores separate

| Experiment / interface | Score or model | Interpretation |
| --- | --- | --- |
| Internal walk-forward CV, five folds | Ensemble cosine **+0.14088** | Cross-period internal estimate |
| Untouched hold-out, months 65–70 | LightGBM cosine **+0.15171** | Favorable period; not ensemble performance |
| Submitted leaderboard result | v4 ensemble cosine **+0.12900** | External competition measurement, below recorded median |
| Interactive demo inference | Default LightGBM v3 per current model card | Inference demo; **not** the scored leaderboard ensemble |

These are **historical research results** documented by the [README](../README.md#headline-result), [model card](../MODEL_CARD.md), and [leaderboard record](../results/leaderboard.json), not investment-performance claims. The dashboard screenshot itself does not verify their numerical reproduction.

## Acceptance checklist before claiming a refreshed dashboard or Social Preview

1. Run the repository's documented local checks (`make check`) against the current code and verify results/data contracts. Do not change or re-label historical metrics simply because a backtesting fix lands.
2. Start Streamlit using `make streamlit`; inspect the seven page routes and match text, charts and tooltips against the current versioned `results/` sources.
3. Confirm the public Streamlit deployment actually serves the intended commit and verify the offline prediction fallback versus the FastAPI path where applicable.
4. Explicitly identify whether a given chart concerns a CV fold, hold-out, leaderboard entry, synthetic `make demo` run or a derived aggregate. Never promote the synthetic demo's planted signal as a model result.
5. Check display legibility on desktop/mobile, empty states, broken media, and access to original notebook/model-card evidence.
6. Capture a **fresh, authentic** dashboard screenshot with its date and code revision, remove client/personal data, and only then replace the README image or approve a new GitHub Social Preview.
7. Keep the non-investment-advice notice visible. Do not imply that rank correlation/backtesting proves trading profitability after costs, liquidity and execution constraints.

**Current image release status:** historical *source-rendered* portfolio capture retained. Seven-page live visual/regression review is **not** established by the repository screenshot alone. No new image, backtest or deployment is produced by this documentation change.
