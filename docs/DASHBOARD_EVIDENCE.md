# Dashboard evidence and model boundaries

**Source status reviewed:** 22 October 2026. **Screenshot captured:** 12 October 2026.
This document distinguishes the current source, historical image and unverified live
deployment. [Latest recorded checks](VERIFICATION.md).

## What is established

- The [committed screenshot](assets/mscapital-research-dashboard.png) is a historical
  source-rendered capture of the earlier seven-page layout. It does not validate the
  current interface, interactions or publicly deployed commit.
- Current source contains a router and **eight research pages**. Ordinary viewing reads
  derived reports, the bundled model and a bounded 4,970-row feature sample from
  [results/](../results/); it does not fetch live financial prices or query the cloud.
- The Evidence page separates historical metrics, the warehouse snapshot, completed
  Kaggle publication readback and fixed-fit full-data revalidation. The static report
  compares the new models on shared periods.
- The default prediction artifact is **LightGBM v3**. The **v4 ensemble** submitted to
  Kaggle is separate; showing the prediction UI does not prove that it serves v4.
- The current interface uses the official Drive NEXUS 2.0 logos and palette.
  [Brand identity and asset provenance](BRAND_IDENTITY.md).

## Keep measurement scopes separate

| Measurement | Result | Interpretation |
|---|---:|---|
| Historical five-fold ensemble | +0.14088 | Same-fold fitted-blend diagnostic; optimistic, not an unbiased estimate |
| Historical LightGBM holdout, months 65–70 | +0.15171 | Favorable period; not ensemble performance |
| Captured external leaderboard | +0.12900 | Submitted v4 ensemble; historical standing |
| New fixed-fit LightGBM, shared folds 2–5 | +0.138046 | Retrospective full-data revalidation |
| New prior-fold blend, shared folds 2–5 | +0.138299 | Same shared periods; small gain is sensitive to resampling assumptions |

[Model card](../MODEL_CARD.md) · [Revalidation and uncertainty](REVALIDATION.md) ·
[Historical leaderboard capture](../results/leaderboard.json).
Neither a screenshot nor headless rendering proves numerical reproduction or trading profitability.

## Visual acceptance boundary

The browser bridge could not attach during the 22 October pass. No refreshed live
desktop/mobile screenshot or public-deployment certification is claimed. Runtime-only
Streamlit rendering and Docker/API checks passed, as recorded in [verification](VERIFICATION.md).

Before replacing the screenshot or certifying a public revision:

1. Identify the deployed commit and inspect all eight routes against their result sources.
2. Exercise local prediction and the configured API path separately.
3. Check desktop/mobile legibility, theme behavior, charts, media, empty states and downloads.
4. Capture a genuine dated image with its code revision and no personal data.

Retain the research-only notice and identify whether each view concerns historical CV,
holdout, leaderboard, synthetic data or the separate retrospective revalidation.
