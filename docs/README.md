# MSCapital research documentation

NEXUS applied data science portfolio. This is a non-commercial research project with
historical evaluation evidence, not a subscription product or a market data service.

| Reader task | Authoritative document |
|---|---|
| Assess the work in two minutes | [Reviewer brief](hiring-summary.md) |
| Understand the question, findings and failed hypotheses | [Research case study](../README.md#headline-result) |
| Reproduce from a clean clone | [Reproducibility](REPRODUCIBILITY.md) |
| Trace Kaggle → BigQuery → model → dashboard | [Integration and cost boundaries](INTEGRATIONS.md) |
| Run experiments or manage model releases | [Operations reference](OPERATIONS.md) |
| Understand the served model and evaluation windows | [Model card](../MODEL_CARD.md) |
| Review interface and screenshot evidence | [Dashboard evidence](DASHBOARD_EVIDENCE.md) |
| Reuse the approved logo and palette | [NEXUS identity](BRAND_IDENTITY.md) |
| Review checks and live integration limitations | [Verification record](VERIFICATION.md) |
| Inspect validation corrections and uncertainty | [Validation protocol](VALIDATION.md) |
| Inspect the completed fixed-fit full-data run | [Revalidation results and immutable run evidence](REVALIDATION.md) |
| Read the complete historical study | [Research narrative](RESEARCH.md) |
| Contribute, report vulnerabilities or check attribution | [Contributing](../CONTRIBUTING.md), [security](../SECURITY.md), [notices](../NOTICE.md) |

## Repository ownership

| Location | Purpose |
|---|---|
| `configs/`, `sql/` | Runtime configuration, staging queries and data contracts |
| `src/data/`, `src/features/` | Ingestion, bounded BigQuery clients, feature engineering and exports |
| `src/models/`, `src/evaluation/` | Model experiments, temporal validation and evidence audit |
| `src/inference/`, `api/` | Model registry, prediction and API guards |
| `streamlit_app/` | Eight research pages; local outputs preferred to bundled results |
| `results/` | Historical measurements, derived reports, a bounded feature sample and the inference bundle |
| `notebooks/` | Five executed narratives and their reproducible builders |
| `kaggle/` | Versioned sources and metadata of the two corrected public notebooks |
| `site/` | Generated, self-contained HTML research report |
| `docs/assets/` | Official brand assets and clearly dated visual evidence |
| `tests/`, `scripts/`, `.github/` | Regression checks, deployment smoke checks and CI |

Raw data, credentials, model-training scratch space and tool caches belong outside Git.
Executed notebooks and failed-experiment results are research evidence, not cleanup targets.
Do not delete them merely because the final model does not use their candidate features.
