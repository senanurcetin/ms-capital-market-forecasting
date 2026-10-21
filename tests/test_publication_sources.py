"""Published research must preserve corrected method boundaries and executable cells."""
import ast
import copy
import json
import re
from pathlib import Path

from scripts.check_kaggle_publications import source_hash

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_sources_are_valid_and_keep_correction_visible():
    files = sorted((ROOT / "kaggle").glob("*/*.ipynb"))
    assert len(files) == 2
    for path in files:
        nb = json.loads(path.read_text(encoding="utf-8"))
        first_cells = "\n".join("".join(c["source"]) for c in nb["cells"][:2])
        assert "Validation correction" in first_cells
        assert "same fold targets" in first_cells or "same fold" in first_cells.lower()
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        meta = json.loads((path.parent / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] is False
        assert meta["enable_internet"] is False
        assert meta["is_private"] is False


def test_source_hash_ignores_execution_but_detects_changed_claims():
    nb = {"cells": [{"cell_type": "code", "source": ["x = 1\n"], "outputs": []},
                    {"cell_type": "markdown", "source": ["Historical diagnostic"]}]}
    rendered = copy.deepcopy(nb)
    rendered["cells"][0]["outputs"] = [{"output_type": "stream", "text": "1"}]
    assert source_hash(nb) == source_hash(rendered)
    rendered["cells"][1]["source"] = ["Unbiased performance"]
    assert source_hash(nb) != source_hash(rendered)


def test_active_surfaces_do_not_claim_independent_agreement():
    for filename in ("README.md", "src/data/build_site.py", "streamlit_app/pages/1_Overview.py",
                     "streamlit_app/pages/7_Investigation.py"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "computed a completely different way" not in text
        assert "agreeing to 0.00004" not in text
        normalised = re.sub(r"\s+", " ", text)
        assert "does not establish independent validation" in normalised or "does not validate either estimate independently" in normalised
