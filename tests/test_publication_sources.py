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


def test_bundled_publication_readback_matches_the_versioned_sources():
    report = json.loads((ROOT / "results/kaggle_publications.json").read_text())
    assert report["status"] == "pass"
    assert len(report["publications"]) == 2
    for row in report["publications"]:
        metadata = ROOT / "kaggle" / row["id"].split("/")[-1] / "kernel-metadata.json"
        meta = json.loads(metadata.read_text())
        notebook = json.loads((metadata.parent / meta["code_file"]).read_text())
        assert row["source_sha256"] == source_hash(notebook)
        assert row["latest_run_status"] == "COMPLETE"
        assert row["source_matches"] is True


def test_active_surfaces_do_not_claim_independent_agreement():
    for filename in ("README.md", "src/data/build_site.py", "streamlit_app/pages/1_Overview.py",
                     "streamlit_app/pages/7_Investigation.py"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "computed a completely different way" not in text
        assert "agreeing to 0.00004" not in text
        normalised = re.sub(r"\s+", " ", text)
        assert "does not establish independent validation" in normalised or "does not validate either estimate independently" in normalised


def test_official_nexus_identity_assets_are_preserved():
    expected = {
        "nexus-official-transparent.svg": "143788fc114348b03bcef5285c134d63135bd7dbd908380e973596f743c9f45b",
        "nexus-official-mark.svg": "62b6f2a3b306b4c969098f92dfb578ef7ca20373b202872782c70ec8e1dd1dbe",
        "nexus-official-midnight.svg": "c31c9df29384ea3d75587a757a7b28386e66e41c1b52ce072887df1cbf00d6bd",
    }
    import hashlib

    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "docs/assets" / name).read_bytes()).hexdigest() == digest
