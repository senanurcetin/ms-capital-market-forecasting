# Reviewer brief — MS Capital Market Forecasting

> **Validation correction (20 October 2026):** the historical ensemble CV fitted and scored blend weights on the same fold. Its +0.14088 is an optimistic fitted-blend diagnostic, not an unbiased external-performance estimate. New training calibrates weights on earlier folds only. Historical results are retained. A separate [fixed-fit full-data revalidation](REVALIDATION.md) completed on 21 October 2026; its model set and training budget differ from the historical blend.


**Role:** Quantitative ML, analytics engineering and model risk validation. **Stage:** completed research system, not a live investment or trading service.

## Decision and evidence

The project converts very large market microstructure data into a reproducible short-horizon forecasting pipeline, with Arrow column projection, BigQuery feature engineering, and LightGBM/XGBoost evaluation.

| Assessment question | Evidence |
|---|---|
| What does corrected retrospective validation show? | On four shared scored periods: LightGBM **+0.138046**, blend **+0.138299**; small exploratory gain, not established superiority. [Fixed-fit protocol and uncertainty](REVALIDATION.md) |
| What happened to the historical CV headline? | **+0.14088** is a same-fold fitted-blend diagnostic, not an unbiased estimate. [Historical analysis](RESEARCH.md#headline-result) |
| Does the holdout generalize to a competition ranking? | LightGBM v3 holdout **+0.15171**; separate v4 ensemble leaderboard **+0.12900**; [result analysis](../README.md#headline-result) |
| What contradicts the best-case story? | Public snapshot dated **10 September 2026**: **141st/204 teams**, model below snapshot median; [leaderboard evidence](../results/leaderboard.json) |
| Where are reproducibility and limitations documented? | [Model card](../MODEL_CARD.md), [analysis notebook](../notebooks/05_why_the_leaderboard_disagreed.ipynb), [README](../README.md#where-this-stands) |
| Can a reviewer see the interface? | [Source-rendered historical capture](assets/mscapital-research-dashboard.png), [published Streamlit app](https://ms-capital-market-forecasting-mfy6rngulq4fpaovzrhntf.streamlit.app/), and [visual verification boundaries](DASHBOARD_EVIDENCE.md) |

**Model serving distinction:** the [model card](../MODEL_CARD.md) documents LightGBM v3 as the default inference model, while the shipped leaderboard score belongs to a separate v4 ensemble. The screenshot alone does not validate either live deployment or model-version equivalence.

## Engineering decisions

- Time-aware validation with an embargo rather than random splitting of overlapping windows.
- Memory-conscious data processing to handle **804.5 million source rows** within resource constraints.
- Tracking model versions, data contracts, drift and feature redundancy rather than reporting a single score.

## Limits and attribution

The holdout period was relatively favorable; the public leaderboard exposes a generalization gap. Competition data is **not redistributed**, and this is neither investment advice nor an evaluated trading strategy. Refer to [data and software notices](../NOTICE.md).

**Start here:** [Full case study and commands](../README.md).
