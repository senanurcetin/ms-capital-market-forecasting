# Reviewer brief — MS Capital Market Forecasting

**Role:** Quantitative ML, analytics engineering and model risk validation. **Stage:** completed research system, not a live investment or trading service.

## Decision and evidence

The project converts very large market microstructure data into a reproducible short-horizon forecasting pipeline, with Arrow column projection, BigQuery feature engineering, and LightGBM/XGBoost evaluation.

| Assessment question | Evidence |
|---|---|
| Does it beat its internal validation protocol? | Five walk-forward/embargo folds: mean cosine **+0.14088**; [validation design](../README.md#validation-walk-forward-with-an-embargo) |
| Does the holdout generalize to a competition ranking? | Holdout **+0.15171**, shipped leaderboard **+0.12900**; [result analysis](../README.md#headline-result) |
| What contradicts the best-case story? | Public snapshot: **141st/204 teams**, model below snapshot median; [leaderboard evidence](../results/leaderboard.json) |
| Where are reproducibility and limitations documented? | [Model card](../MODEL_CARD.md), [analysis notebook](../notebooks/05_why_the_leaderboard_disagreed.ipynb), [README](../README.md#where-this-stands) |
| Can a reviewer see the interface? | [Captured dashboard](assets/mscapital-research-dashboard.png) and [live Streamlit app](https://ms-capital-market-forecasting-mfy6rngulq4fpaovzrhntf.streamlit.app/) |

## Engineering decisions

- Time-aware validation with an embargo rather than random splitting of overlapping windows.
- Memory-conscious data processing to handle **804.5 million source rows** within resource constraints.
- Tracking model versions, data contracts, drift and feature redundancy rather than reporting a single score.

## Limits and attribution

The holdout period was relatively favorable; the public leaderboard exposes a generalization gap. Competition data is **not redistributed**, and this is neither investment advice nor an evaluated trading strategy. Refer to [data and software notices](../NOTICE.md).

**Start here:** [Full case study and commands](../README.md).
