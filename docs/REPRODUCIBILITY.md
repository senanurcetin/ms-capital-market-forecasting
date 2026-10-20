# Reproduce and review

[Documentation index](README.md)

## Clean Python 3.11 environment

```bash
git clone https://github.com/senanurcetin/ms-capital-market-forecasting.git
cd ms-capital-market-forecasting
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Preserve the pinned dependency sets. `pyproject.toml` configures tools; it deliberately
does not declare a second package installation path for Streamlit Cloud.

## Review without credentials or competition data

```bash
python -m src.evaluation.evidence --output .artifacts/evidence-audit.json
python -m src.evaluation.uncertainty
python -m src.data.build_site
python -m streamlit run streamlit_app/app.py
```

Open `site/index.html` for the offline report. The app reads committed results and a
small feature sample; it does not query BigQuery or Kaggle during ordinary viewing.
The evidence report includes SHA-256 hashes and reconciles the ensemble mean across
CSV, JSON and its fold records. A successful audit verifies consistency, not provenance
from an independent rerun. No wall-clock timestamp is fabricated from file modification times.

## Verify changes

```bash
python -m ruff check src/ api/ streamlit_app/ tests/ scripts/
python -m mypy
python -m pytest tests/ -q --maxfail=1 --cov --cov-report=term
python scripts/check_deploy.py
```

The last command executes the router and every page using Streamlit AppTest, including
the evidence page. It is a render check, not browser interaction or live-deployment proof.
CI separately installs only `requirements.txt` for the runtime render check.

## Synthetic end-to-end demonstration

```bash
python -m src.demo --samples 2500 --root .demo
```

The seeded demo trains its own model and writes to `.demo/`; its planted signal is not
a Kaggle result. Do not export synthetic outputs over the historical `results/` bundle.

For full-data experiments, credentials, release management and CLI options see
[operations](OPERATIONS.md) and [integration boundaries](INTEGRATIONS.md).

Training CLI outputs now include a `.run.json` sidecar. See the
[validation correction and experiment record contract](VALIDATION.md).
