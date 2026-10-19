"""Check relative Markdown document links after reorganizing the research documentation."""
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors = []
    documents = [*ROOT.glob("*.md"), *sorted((ROOT / "docs").rglob("*.md"))]
    for document in documents:
        content = re.sub(r"```.*?```", "", document.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", content):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / unquote(target)).exists():
                errors.append(f"{document.relative_to(ROOT)}: missing {target}")
    for error in errors:
        print(error)
    print(f"Documentation links: {len(documents)} documents, {len(errors)} missing targets")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
