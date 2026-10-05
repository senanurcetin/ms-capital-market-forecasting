"""Smoke-test a RUNNING API over HTTP, the way a client would use it.

    python scripts/e2e_smoke.py http://localhost:8000 [--api-key KEY]

It checks the contract end to end rather than any one function: the service is healthy and has a
model, /v1/features lists the inputs, a row built from those names scores, a bad row is rejected
rather than silently filled, the credential (when one is set) is enforced, and /metrics counted
the work. Standard library only, so it runs from CI or a laptop against a container or a
local uvicorn without installing anything.

Exits non-zero with a message on the first failed check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(base: str, method: str, path: str, body: dict | None = None,
         key: str | None = None) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(f"{base}{path}", method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def check(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        sys.exit(f"smoke test failed: {what}")


def wait_healthy(base: str, seconds: int = 60) -> None:
    deadline, last = time.time() + seconds, "no response"
    while time.time() < deadline:
        try:
            status, body = call(base, "GET", "/health")
            last = body
            if status == 200 and json.loads(body).get("model_loaded"):
                return
        except (urllib.error.URLError, OSError) as exc:
            last = str(exc)
        time.sleep(2)
    sys.exit(f"API did not become healthy with a model within {seconds}s: {last}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base_url")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--wait", type=int, default=60, help="seconds to wait for a loaded model")
    args = ap.parse_args(argv)
    base, key = args.base_url.rstrip("/"), args.api_key

    wait_healthy(base, args.wait)
    check(True, "healthy, with a model loaded")

    status, body = call(base, "GET", "/v1/features", key=key)
    check(status == 200, "GET /v1/features answers")
    names = json.loads(body)["features"]
    check(len(names) > 0 and len(set(names)) == len(names), f"{len(names)} distinct feature names")

    row = dict.fromkeys(names, 0.0)
    status, body = call(base, "POST", "/v1/predict", {"features": row}, key)
    check(status == 200, "POST /v1/predict scores a complete row")
    out = json.loads(body)
    check(isinstance(out["predicted_return"], float) and out["direction"] in
          {"UP", "DOWN", "FLAT"}, f"response is well-formed ({out['direction']})")

    status, body = call(base, "POST", "/v1/batch-predict", {"rows": [row, row, row]}, key)
    check(status == 200 and json.loads(body)["n"] == 3, "POST /v1/batch-predict scores 3 rows")

    partial = dict(list(row.items())[:-1])
    status, _ = call(base, "POST", "/v1/predict", {"features": partial}, key)
    check(status == 422, "a row with a feature missing is rejected, not filled in")

    status, body = call(base, "GET", "/v1/model-info", key=key)
    check(status == 200 and "provenance" in json.loads(body), "/v1/model-info reports provenance")

    if key:
        status, _ = call(base, "POST", "/v1/predict", {"features": row})
        check(status == 401, "the same request without the key is refused")
        status, _ = call(base, "GET", "/health")
        check(status == 200, "/health stays open without the key")

    status, text = call(base, "GET", "/metrics")
    check(status == 200 and "mscapital_rows_scored_total 4" in text,
          "/metrics counted the 4 rows scored")
    print("smoke test passed")


if __name__ == "__main__":
    main()
