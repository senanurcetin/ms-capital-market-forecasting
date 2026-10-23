# Security policy

## Scope

This is a research project. The serving API lives in `api/`; the published Streamlit
dashboard displays derived results and supports inference. The repository does not
configure a project-operated public API endpoint, so reports should identify the code,
defaults and actual exposure being assessed. Optional MLflow exposure is described below.

## Reporting

Please report a vulnerability **privately**, using GitHub's *Report a vulnerability* button on the
repository's Security tab. If that is not available, open an issue that says only that you have a
security report and asks for a private channel - **do not put exploit details in a public issue**.

You can expect an acknowledgement, and a fix or an explanation of why it is not one. There is no
bounty.

## Hardening you should apply when you run the API

The defaults favour a zero-config local run; a deployment should change them:

| Setting | Default | Set it when |
|---|---|---|
| `MSCAPITAL_API_KEY` | unset - model endpoints are open | the API is reachable by anyone you do not trust |
| `MSCAPITAL_ADMIN_TOKEN` | unset - `/reload` answers 403 | you need to swap the model without a restart |
| `MSCAPITAL_RATE_LIMIT_PER_MIN` | 0 - off | a client address identifies one caller (not behind a shared proxy) |
| `MSCAPITAL_MAX_BODY_BYTES` / `MSCAPITAL_MAX_BATCH_ROWS` | 20 MiB / 10,000 | you want a tighter cost bound per request |

The MLflow server in `docker-compose.yml` has no authentication, so it is published on `127.0.0.1`
only; set `MSCAPITAL_MLFLOW_BIND=0.0.0.0` to expose it on purpose, and put something in front of it
if you do.

Terminate TLS in front of the API: it does not do so itself, and the credentials are sent as
headers. The container runs as a non-root user. `/health` and `/metrics` are unauthenticated by
design. `/health` reports readiness and a load-error detail when degraded; `/metrics`
reports aggregate operational measurements. Neither returns submitted prediction rows.

## Published images

The `api` and `app` images are public on GitHub Container Registry
(`ghcr.io/senanurcetin/ms-capital-market-forecasting/api` and `.../app`). Releases use both
prefixed and plain version tags, for example `v1.2.3` and `1.2.3`; these are naming examples,
not claims that this particular release exists. The workflow publishes commits on `main`.

The **API image** contains serving code and configuration. It starts without a model and
reports `degraded` until you mount an artifact. The **dashboard image** also contains the
bundled derived reports, bounded feature sample, LightGBM artifact, brand assets and theme
configuration for offline viewing. Neither image should contain raw competition files or
credential files. Supply runtime credentials externally when an integration requires them.

## Not a security boundary

The model is a regression on public-style market features with a planted-signal demo. Its outputs
are for research and are **not investment advice**.
