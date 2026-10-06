"""scripts/changelog_section.py: the release notes the release workflow publishes."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "changelog_section.py"
_spec = importlib.util.spec_from_file_location("changelog_section", _PATH)
cs = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["changelog_section"] = cs
_spec.loader.exec_module(cs)  # type: ignore[union-attr]

TEXT = """# Changelog

## [Unreleased]

## [1.1.0] - 2026-10-06

### Added
- a new thing

### Changed
- a changed thing

## [1.0.0] - 2026-10-05

### Added
- the first thing

[Unreleased]: https://example.com/compare/v1.1.0...HEAD
[1.1.0]: https://example.com/compare/v1.0.0...v1.1.0
[1.0.0]: https://example.com/releases/tag/v1.0.0
"""


def test_a_middle_section_stops_at_the_next_heading():
    body = cs.section(TEXT, "1.1.0")
    assert body.startswith("### Added") and "a changed thing" in body
    assert "first thing" not in body and "## [1.0.0]" not in body


def test_the_last_section_stops_before_the_link_definitions():
    body = cs.section(TEXT, "1.0.0")
    assert body == "### Added\n- the first thing"


def test_an_empty_or_missing_section_is_empty():
    assert cs.section(TEXT, "Unreleased") == ""
    assert cs.section(TEXT, "9.9.9") == ""


def test_a_version_is_matched_exactly_not_by_prefix():
    text = "## [1.10.0] - x\n- ten\n\n## [1.1.0] - y\n- one\n"
    assert cs.section(text, "1.1.0") == "- one"


def test_main_fails_loudly_for_a_version_with_no_notes(capsys):
    assert cs.main(["9.9.9"]) == 1
    assert "no (or an empty) section" in capsys.readouterr().err


def test_main_accepts_a_leading_v_and_prints_the_real_changelog_section(capsys):
    assert cs.main(["v1.1.0"]) == 0
    out = capsys.readouterr().out
    assert "### Added" in out and "RidgeModel" in out


def test_main_rejects_a_missing_argument(capsys):
    assert cs.main([]) == 2


@pytest.mark.parametrize("version", ["1.1.0", "1.0.0"])
def test_every_released_version_in_the_changelog_has_notes(version):
    assert cs.section(cs.CHANGELOG.read_text(encoding="utf-8"), version), version
