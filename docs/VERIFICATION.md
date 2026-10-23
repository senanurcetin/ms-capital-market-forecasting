# Verification status — 22 October 2026

[Documentation index](README.md)

Implementation baseline: [PR #81](https://github.com/senanurcetin/ms-capital-market-forecasting/pull/81),
commit `935d7e8`. Its [CI run](https://github.com/senanurcetin/ms-capital-market-forecasting/actions/runs/39716676929)
passed all four jobs. These records establish the checks below; they do not certify
which commit the public Streamlit deployment currently serves.

| Check | Latest recorded evidence |
|---|---|
| Offline regression suite | Passed; coverage **73.49%**, above the configured 68% floor; optional skips are recorded in the CI log |
| Ruff and mypy | Passed across the configured application, pipeline, dashboard and scripts |
| Runtime-only Streamlit rendering | Router and all eight research pages passed, including the new publication and full-data evidence |
| Docker | API and dashboard images built successfully |
| API artifact compatibility | Synthetic model trained, loaded and served through the API image |
| Published evidence | Historical headline metrics reconcile; full-data summary, CSV, run hashes and shared-fold comparisons are checked separately |
| Official NEXUS identity | Three official SVG hashes are covered by regression checks; their source is documented in [brand provenance](BRAND_IDENTITY.md) |
| Kaggle | Both corrected public notebooks completed execution and matched the versioned cell sources; [readback snapshot](../results/kaggle_publications.json) |
| BigQuery | 22 tables inspected and nine metadata contracts passed on 20 October; [dated warehouse report](../results/warehouse_audit.json) |
| Full-data revalidation | Completed on 21 October: 1,257,637 source rows, 292 features, no sampling; [protocol, results and hashes](REVALIDATION.md) |
| Live desktop/mobile visual review | Not certified: the browser bridge could not attach on 22 October |

## Scope and remaining limits

The fixed-fit run passed no scored validation data to model fitting and calibrated
blend weights only on earlier folds. It did not remeasure the holdout, promote a model
or create a competition submission. Previously explored periods remain retrospective
evidence. The temporary BigQuery repository secret was removed after the completed run.

Synthetic demo scores are not competition measurements. Metadata row counts do not
establish uniqueness or null rates; the separate REST export verified every expected
sample key before the full-data run. Hashes identify bytes, not independent provenance.

The existing screenshot is a dated 12 October source-rendered capture. No refreshed
live/mobile screenshot or deployed-commit equivalence is claimed. Streamlit AppTest
checks rendering; it does not replace browser interaction or visual inspection.
See [dashboard evidence](DASHBOARD_EVIDENCE.md).

Earlier local checks on 19 October had lower coverage and no working Docker/cloud
access. They are superseded by the dated CI and integration records above; historical
connection recovery details remain in [integrations](INTEGRATIONS.md).
