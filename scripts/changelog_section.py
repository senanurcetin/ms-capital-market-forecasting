"""Print one version's section of CHANGELOG.md: the release notes for that version.

    python scripts/changelog_section.py 1.1.0      # the body under "## [1.1.0] - ..."

Used by .github/workflows/release.yml to fill in the GitHub release. Exits 1 when the version has no
section or the section is empty, so a release cannot go out with blank notes by accident.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def section(text: str, version: str) -> str:
    """The body of `## [version]`, from the line after its heading to the next `## [` heading."""
    heading = re.compile(rf"^## \[{re.escape(version)}\](?:\s|$)", re.MULTILINE)
    start = heading.search(text)
    if start is None:
        return ""
    rest = text[start.end():].split("\n", 1)[1] if "\n" in text[start.end():] else ""
    # a section ends at the next heading or at the link definitions ("[1.1.0]: https://...") that
    # close the file
    nxt = re.search(r"^(?:## \[|\[[^\]]+\]: )", rest, re.MULTILINE)
    return (rest[: nxt.start()] if nxt else rest).strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: changelog_section.py <version, without the leading v>", file=sys.stderr)
        return 2
    body = section(CHANGELOG.read_text(encoding="utf-8"), args[0].removeprefix("v"))
    if not body:
        print(f"no (or an empty) section for {args[0]} in {CHANGELOG.name}", file=sys.stderr)
        return 1
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
