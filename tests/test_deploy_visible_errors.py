"""A visible Streamlit error must fail the deployment smoke gate."""

from pathlib import Path
from types import SimpleNamespace

from scripts import check_deploy


class FakeAppTest:
    results = {}

    @classmethod
    def from_file(cls, path, **_kwargs):
        response = cls.results[Path(path).name]
        return SimpleNamespace(run=lambda: response)


def test_visible_error_banner_is_a_failed_page(monkeypatch, tmp_path, capsys):
    (tmp_path / "app.py").write_text("# test router\n", encoding="utf-8")
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "1_Overview.py").write_text("# test overview\n", encoding="utf-8")
    FakeAppTest.results = {
        "app.py": SimpleNamespace(exception=[], error=[]),
        "1_Overview.py": SimpleNamespace(
            exception=[], error=[SimpleNamespace(value="Missing source data")]
        ),
    }
    monkeypatch.setattr(check_deploy, "APP", tmp_path)
    monkeypatch.setattr(check_deploy, "ROOT", tmp_path.parent)
    monkeypatch.setattr(check_deploy, "AppTest", FakeAppTest)
    assert check_deploy.main() == 1
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "1_Overview.py" in output
    assert "Missing source data" in output
    assert "1/2 pages rendered" in output


def test_clean_pages_remain_green(monkeypatch, tmp_path, capsys):
    (tmp_path / "app.py").write_text("# test router\n", encoding="utf-8")
    FakeAppTest.results = {"app.py": SimpleNamespace(exception=[], error=[])}
    monkeypatch.setattr(check_deploy, "APP", tmp_path)
    monkeypatch.setattr(check_deploy, "ROOT", tmp_path.parent)
    monkeypatch.setattr(check_deploy, "AppTest", FakeAppTest)
    assert check_deploy.main() == 0
    assert "1/1 pages rendered" in capsys.readouterr().out
