"""A small file-based model registry: versioned releases, promote, rollback.

Until now a model's "version" was a string inside model_meta.json and the served artefact was
whatever happened to sit in `models/current`. Replacing it overwrote the previous one, so there
was nothing to go back to and no record of what had been live.

LAYOUT, under the registry root (by default <data_root>/models, which makes `current` exactly
the directory the API already loads):

    releases/<name>-<version>-<UTC stamp>/    one immutable copy per promotion
    current/                                  a copy of the active release; what the API serves
    history.json                              every promote / rollback, oldest first

WHY COPIES, NOT SYMLINKS: the project is developed on Windows, where creating a symlink needs
elevated rights, and a Docker bind mount follows a directory but not always a link out of it.

THE SWAP is two renames (current -> current.old, current.new -> current), not one atomic
operation, so there is a window of microseconds in which `current` does not exist; an API that
loads during it reports "degraded" and a /reload fixes it. If the second rename fails the first
is undone. Promotion validates the artefact by loading it first, so a broken build is refused
rather than served.

Nothing here talks to a running API: after promote or rollback, call POST /reload.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path

from src.inference.predictor import METADATA_FILE, ModelNotLoadedError, load_bundle

HISTORY_FILE = "history.json"
RELEASES_DIR = "releases"
CURRENT_DIR = "current"


class RegistryError(RuntimeError):
    """A promotion or rollback that cannot be carried out."""


def default_root() -> Path:
    return Path(os.environ.get("MSCAPITAL_DATA_ROOT", "C:/mscapital_data")) / "models"


def _history(root: Path) -> list[dict]:
    path = root / HISTORY_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write_history(root: Path, history: list[dict]) -> None:
    tmp = root / (HISTORY_FILE + ".tmp")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(tmp, root / HISTORY_FILE)


def active_release(root: Path) -> str | None:
    """The release `current` was last set to, or None if nothing has been promoted."""
    history = _history(root)
    return history[-1]["release"] if history else None


def _swap_in(root: Path, release: Path) -> None:
    """Make `current` a copy of `release`, restoring the old one if anything fails."""
    new, cur, old = root / "current.new", root / CURRENT_DIR, root / "current.old"
    for leftover in (new, old):
        shutil.rmtree(leftover, ignore_errors=True)
    shutil.copytree(release, new)
    had_current = cur.exists()
    try:
        if had_current:
            os.rename(cur, old)
        os.rename(new, cur)
    except OSError:
        if had_current and old.exists() and not cur.exists():
            os.rename(old, cur)
        shutil.rmtree(new, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)


def promote(model_dir: str | Path, root: str | Path | None = None, *, note: str = "") -> str:
    """Store `model_dir` as a new release and make it the served model. Returns its id."""
    root = Path(root) if root else default_root()
    model_dir = Path(model_dir)
    try:
        bundle = load_bundle(model_dir)      # refuse anything the API could not serve
    except ModelNotLoadedError as exc:
        raise RegistryError(f"refusing to promote {model_dir}: {exc}") from exc
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    release_id = f"{bundle.name}-{bundle.version}-{stamp}"
    release = root / RELEASES_DIR / release_id
    release.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, release)
    _swap_in(root, release)
    history = _history(root)
    history.append({"release": release_id, "action": "promote", "at": stamp, "note": note})
    _write_history(root, history)
    return release_id


def activate(release_id: str, root: str | Path | None = None, *, action: str = "activate",
             note: str = "", rejected: str | None = None) -> str:
    """Make an existing release the served model."""
    root = Path(root) if root else default_root()
    release = root / RELEASES_DIR / release_id
    if not (release / METADATA_FILE).exists():
        raise RegistryError(f"no such release: {release_id}")
    _swap_in(root, release)
    history = _history(root)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    entry = {"release": release_id, "action": action, "at": stamp, "note": note}
    if rejected:
        entry["rejected"] = rejected
    history.append(entry)
    _write_history(root, history)
    return release_id


def rollback(root: str | Path | None = None) -> str:
    """Go back to the release that was served before the active one.

    A release that has been rolled away from is treated as rejected and skipped from then on,
    so rolling back twice walks back through the releases (r3 -> r2 -> r1) instead of bouncing
    between the same two, and promoting a new release afterwards does not "roll back" onto
    the one that was just abandoned. A deliberate jump to any release is activate().
    """
    root = Path(root) if root else default_root()
    history = _history(root)
    if not history:
        raise RegistryError("nothing has been promoted yet")
    current = history[-1]["release"]
    rejected = {e["rejected"] for e in history if e.get("rejected")} | {current}
    for entry in reversed(history[:-1]):
        rid = entry["release"]
        if rid not in rejected and (root / RELEASES_DIR / rid / METADATA_FILE).exists():
            return activate(rid, root, action="rollback", note=f"from {current}",
                            rejected=current)
    raise RegistryError(f"no earlier release to return to from {current}")


def list_releases(root: str | Path | None = None) -> list[dict]:
    """Every release on disk, newest first, with the active one marked."""
    root = Path(root) if root else default_root()
    active = active_release(root)
    out = []
    releases = root / RELEASES_DIR
    for d in sorted(releases.iterdir(), reverse=True) if releases.exists() else []:
        meta_path = d / METADATA_FILE
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out.append({"release": d.name, "name": meta.get("name"), "version": meta.get("version"),
                    "trained_at": meta.get("trained_at"), "active": d.name == active})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Promote, roll back and list model releases")
    ap.add_argument("--root", help="registry root (default: <MSCAPITAL_DATA_ROOT>/models)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote", help="store a model directory as a release and serve it")
    p.add_argument("model_dir")
    p.add_argument("--note", default="")
    sub.add_parser("rollback", help="serve the previous release again")
    a = sub.add_parser("activate", help="serve a named release")
    a.add_argument("release")
    sub.add_parser("list", help="show releases, newest first")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "promote":
            print(f"promoted {promote(args.model_dir, args.root, note=args.note)}")
        elif args.cmd == "rollback":
            print(f"rolled back to {rollback(args.root)}")
        elif args.cmd == "activate":
            print(f"activated {activate(args.release, args.root)}")
        else:
            for r in list_releases(args.root):
                print(f"{'*' if r['active'] else ' '} {r['release']}  trained {r['trained_at']}")
            return
    except RegistryError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print("now POST /reload on the running API to pick it up")


if __name__ == "__main__":
    main()
