"""Opt-in readback of owned Kaggle publications; never pushes or runs notebooks."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def source_hash(notebook: dict) -> str:
    """Ignore execution outputs and IDs; compare ordered cell types and source text."""
    source = [{"cell_type": cell["cell_type"], "source": "".join(cell["source"])}
              for cell in notebook["cells"]]
    return hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".artifacts/kaggle-publications.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    api = KaggleApi()
    api.authenticate()
    checks = []
    for metadata in sorted((root / "kaggle").glob("*/kernel-metadata.json")):
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if not meta["id"].startswith("senanuretin/ms-capital-"):
            raise ValueError("Only the configured owner publications may be checked")
        local = json.loads((metadata.parent / meta["code_file"]).read_text(encoding="utf-8"))
        try:
            with tempfile.TemporaryDirectory(prefix="mscapital-kaggle-") as temporary:
                api.kernels_pull(meta["id"], path=temporary, metadata=True, quiet=True)
                remote_meta = json.loads((Path(temporary) / "kernel-metadata.json").read_text())
                remote = json.loads((Path(temporary) / remote_meta["code_file"]).read_text(encoding="utf-8"))
            status = api.kernels_status(meta["id"])
            if not isinstance(status, dict):
                status = json.loads(str(status))
            matches = source_hash(local) == source_hash(remote)
            state = status.get("status")
            check = {"id": meta["id"], "url": f"https://www.kaggle.com/code/{meta['id']}",
                     "status": "pass" if matches and state == "COMPLETE" else "fail",
                     "latest_run_status": state, "source_matches": matches,
                     "source_sha256": source_hash(local), "public": not remote_meta["is_private"],
                     "scope": "Re-execution of historical result analysis; no full model retraining or submission"}
        except Exception as exc:
            check = {"id": meta["id"], "status": "unavailable", "reason": type(exc).__name__}
        checks.append(check)
        print(json.dumps(check), flush=True)
    report = {"checked_at": datetime.now(UTC).isoformat(), "publications": checks,
              "status": "pass" if checks and all(c["status"] == "pass" for c in checks) else "fail"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
