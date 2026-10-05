# Security policy

## Scope

This is a research project, not a service. The one component that listens on a network is the
serving API (`api/`), and the published dashboard is read-only over derived results. **There is no
live deployment of the API run by the project**, so a report is about the code and its defaults.

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

Terminate TLS in front of the API: it does not do so itself, and the credentials are sent as
headers. The container runs as a non-root user. `/health` and `/metrics` are unauthenticated by
design and carry counts only, never request contents.

## Not a security boundary

The model is a regression on public-style market features with a planted-signal demo. Its outputs
are for research and are **not investment advice**.
