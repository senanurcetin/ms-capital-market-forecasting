"""Promote, roll back, list: the registry must never leave the served model unusable."""
import json

import pytest
from src.inference.predictor import Predictor, load_bundle, make_provenance, save_bundle
from src.inference.registry import (
    RegistryError,
    activate,
    active_release,
    list_releases,
    promote,
    rollback,
)

from tests.test_api import FEATURES, DummyModel


def _build(path, version, **kw):
    return save_bundle(path, model=DummyModel(), kind="sklearn", features=FEATURES,
                       name="dummy", version=version, **kw)


def _served_version(root):
    return json.loads((root / "current" / "model_meta.json").read_text())["version"]


def test_promote_makes_the_model_servable_from_current(tmp_path):
    root = tmp_path / "models"
    release = promote(_build(tmp_path / "b1", "v1"), root)
    assert release.startswith("dummy-v1-") and active_release(root) == release
    served = Predictor.from_dir(root / "current")          # exactly what the API loads
    assert served.predict([dict.fromkeys(FEATURES, 1.0)]).shape == (1,)


def test_a_second_promotion_replaces_current_but_keeps_the_first_release(tmp_path):
    root = tmp_path / "models"
    r1 = promote(_build(tmp_path / "b1", "v1"), root)
    r2 = promote(_build(tmp_path / "b2", "v2"), root)
    assert _served_version(root) == "v2" and active_release(root) == r2
    assert {r["release"] for r in list_releases(root)} == {r1, r2}
    assert [r["active"] for r in list_releases(root)].count(True) == 1


def test_rollback_restores_the_previous_release(tmp_path):
    root = tmp_path / "models"
    r1 = promote(_build(tmp_path / "b1", "v1"), root)
    promote(_build(tmp_path / "b2", "v2"), root)
    assert rollback(root) == r1
    assert _served_version(root) == "v1" and active_release(root) == r1


def test_rolling_back_twice_walks_back_instead_of_bouncing(tmp_path):
    root = tmp_path / "models"
    r1 = promote(_build(tmp_path / "b1", "v1"), root)
    r2 = promote(_build(tmp_path / "b2", "v2"), root)
    promote(_build(tmp_path / "b3", "v3"), root)
    assert rollback(root) == r2 and _served_version(root) == "v2"
    assert rollback(root) == r1 and _served_version(root) == "v1"   # not back to v3
    with pytest.raises(RegistryError, match="no earlier release"):
        rollback(root)


def test_a_release_rolled_away_from_is_not_returned_to_by_a_later_rollback(tmp_path):
    root = tmp_path / "models"
    promote(_build(tmp_path / "b1", "v1"), root)
    r2 = promote(_build(tmp_path / "b2", "v2"), root)
    promote(_build(tmp_path / "b3", "v3"), root)        # turns out to be bad
    rollback(root)                                       # back to v2
    promote(_build(tmp_path / "b4", "v4"), root)
    assert rollback(root) == r2                          # v2, not the abandoned v3


def test_activate_is_a_deliberate_jump_to_any_release(tmp_path):
    root = tmp_path / "models"
    r1 = promote(_build(tmp_path / "b1", "v1"), root)
    promote(_build(tmp_path / "b2", "v2"), root)
    assert activate(r1, root) == r1 and _served_version(root) == "v1"
    assert [r["active"] for r in list_releases(root)].count(True) == 1


def test_rollback_with_nothing_to_return_to_is_an_error_not_a_noop(tmp_path):
    root = tmp_path / "models"
    with pytest.raises(RegistryError, match="nothing has been promoted"):
        rollback(root)
    promote(_build(tmp_path / "b1", "v1"), root)
    with pytest.raises(RegistryError, match="no earlier release"):
        rollback(root)


def test_a_broken_artefact_is_refused_and_current_is_untouched(tmp_path):
    root = tmp_path / "models"
    promote(_build(tmp_path / "b1", "v1"), root)
    broken = tmp_path / "broken"
    broken.mkdir()
    with pytest.raises(RegistryError, match="refusing to promote"):
        promote(broken, root)
    assert _served_version(root) == "v1" and len(list_releases(root)) == 1


def test_activating_an_unknown_release_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="no such release"):
        activate("nope", tmp_path / "models")


def test_a_release_is_a_copy_so_later_edits_to_the_source_do_not_change_it(tmp_path):
    root = tmp_path / "models"
    src = _build(tmp_path / "b1", "v1")
    rid = promote(src, root)
    (src / "model_meta.json").write_text(json.dumps({"tampered": True}))
    assert load_bundle(root / "releases" / rid).version == "v1"


def test_failed_swap_restores_the_previous_current(tmp_path, monkeypatch):
    """If the second rename fails the served model must be the old one, not nothing."""
    import os

    root = tmp_path / "models"
    promote(_build(tmp_path / "b1", "v1"), root)
    real = os.rename

    def flaky(src, dst):
        if str(dst).endswith("current") and str(src).endswith("current.new"):
            raise OSError("disk says no")
        return real(src, dst)

    monkeypatch.setattr(os, "rename", flaky)
    with pytest.raises(OSError):
        promote(_build(tmp_path / "b2", "v2"), root)
    monkeypatch.setattr(os, "rename", real)
    assert _served_version(root) == "v1"


# --- provenance ------------------------------------------------------------------------------

def test_provenance_is_written_loaded_and_reported(tmp_path):
    prov = make_provenance(features=FEATURES, n_train_rows=1234, train_months=(0, 63))
    d = _build(tmp_path / "b", "v1", provenance=prov)
    bundle = load_bundle(d)
    assert bundle.provenance["n_train_rows"] == 1234
    assert bundle.provenance["train_months"] == [0, 63]
    assert Predictor(bundle).info()["provenance"] == bundle.provenance


def test_provenance_hash_depends_on_feature_order():
    a = make_provenance(features=["x", "y"], n_train_rows=1, train_months=(0, 1))
    b = make_provenance(features=["y", "x"], n_train_rows=1, train_months=(0, 1))
    assert a["features_sha256"] != b["features_sha256"]


def test_artefacts_without_provenance_still_load(tmp_path):
    assert load_bundle(_build(tmp_path / "b", "v1")).provenance == {}


def test_cli_promote_list_rollback(tmp_path, capsys):
    from src.inference.registry import main

    root = str(tmp_path / "models")
    main(["--root", root, "promote", str(_build(tmp_path / "b1", "v1"))])
    main(["--root", root, "promote", str(_build(tmp_path / "b2", "v2"))])
    main(["--root", root, "list"])
    listing = capsys.readouterr().out
    assert listing.count("*") == 1 and "dummy-v2-" in listing.split("*")[1]
    main(["--root", root, "rollback"])
    assert "rolled back to dummy-v1-" in capsys.readouterr().out


def test_cli_reports_errors_as_a_clean_exit_not_a_traceback(tmp_path):
    from src.inference.registry import main

    with pytest.raises(SystemExit, match="nothing has been promoted"):
        main(["--root", str(tmp_path / "models"), "rollback"])
