"""
Live-proxy lint for provider URL path-segment encoding.

LiteLLM's provider transformations interpolate user-controlled
identifiers (``file_id``, ``batch_id``, ``video_id``, ``container_id``,
model name, ...) into outbound upstream URLs. If the identifier is not
percent-encoded, characters like ``/``, ``%2F``, ``?``, and ``#`` do not
stay inside a single path segment once FastAPI and httpx finish
normalizing the URL, and the request lands on a different upstream
endpoint than the route intended.

This script authenticates to a running LiteLLM proxy with a normal
virtual key and fires a set of inputs containing ``../../`` sequences
through each affected endpoint. For every probe it reports:

  * FAIL         -- the upstream response indicates the input was
                    decoded back into path separators and the request
                    reached a different endpoint (e.g. 200 with a
                    batches-shaped body from a file-retrieve call).
  * PASS         -- the proxy / upstream returned a 4xx consistent with
                    the percent-encoded form never resolving to a real
                    endpoint, i.e. encoding held.
  * INCONCLUSIVE -- 5xx, auth failures, or 200s that did not match any
                    shape heuristic.
  * ERROR        -- network / config issue talking to the proxy.

Usage
-----
    # Requires a LiteLLM proxy running at $PROXY_BASE with a virtual key
    # that has access to anthropic / openai models. The key does NOT
    # need admin privileges.
    export PROXY_BASE=http://localhost:4000
    export PROXY_KEY=sk-virtual-xxxxxxxx

    python scripts/lint_url_encoding.py

    # Or target a single probe:
    python scripts/lint_url_encoding.py --only anthropic_files_retrieve

Run it twice: once before the encoding change to establish a baseline
(some probes are expected to FAIL), then again after the change (every
probe should report PASS). The script exits non-zero if any probe
reports FAIL.

Notes
-----
* Inputs are GET-only (or POST with an empty body) and cannot write or
  mutate state upstream.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple
from urllib.parse import quote

import httpx


DEFAULT_INPUT = "..%2F..%2Fv1%2Fmessages%2Fbatches"
"""Sample 2-level ``../../v1/messages/batches`` input.

FastAPI's ``:path`` converter decodes ``%2F`` to ``/``; for a route
rooted at ``/v1/<resource>/<id>`` the template URL becomes
``/v1/<resource>/../../v1/messages/batches``, which httpx normalizes to
``/v1/messages/batches`` before dispatch -- that is what we're checking
for.
"""


# --- body shape heuristics -------------------------------------------------


def _looks_like_anthropic_batch(body: Any) -> bool:
    """Anthropic message-batches response shape (list or single)."""
    if not isinstance(body, dict):
        return False
    if isinstance(body.get("data"), list) and any(
        isinstance(item, dict) and item.get("type") == "message_batch"
        for item in body["data"]
    ):
        return True
    if body.get("type") == "message_batch" or "processing_status" in body:
        return True
    return False


def _looks_like_anthropic_files_list(body: Any) -> bool:
    """Anthropic files-list shape."""
    if not isinstance(body, dict):
        return False
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    return isinstance(first, dict) and first.get("type") == "file"


def _looks_like_openai_models_list(body: Any) -> bool:
    """OpenAI ``/v1/models`` response shape.

    Strict: ``data`` must exist, be non-empty, and first item must be
    ``object=="model"``. Avoids false positives from any other
    ``{object: "list"}`` response (files, batches, etc.).
    """
    if not isinstance(body, dict):
        return False
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    return isinstance(first, dict) and first.get("object") == "model"


# --- probe definitions -----------------------------------------------------


@dataclass
class Probe:
    name: str
    method: str
    path: str
    # (status_code, parsed_body, headers) -> True iff the request landed
    # on a different endpoint than the route intended.
    is_decoded: Callable[[int, Any, httpx.Headers], bool]
    description: str
    body: Any = None  # optional JSON body for POST probes


PROBES: List[Probe] = [
    # --- Anthropic files -> would resolve to /v1/messages/batches ------------
    Probe(
        name="anthropic_files_retrieve",
        method="GET",
        # FastAPI decodes %2F into /; template becomes
        # /v1/files/../../v1/messages/batches which collapses to
        # /v1/messages/batches when normalized.
        path=f"/anthropic/v1/files/{DEFAULT_INPUT}",
        is_decoded=lambda sc, body, _h: sc == 200 and _looks_like_anthropic_batch(body),
        description=(
            "Retrieve-file handler builds /v1/files/<id>; if the input "
            "collapses to /v1/messages/batches, the response looks like "
            "a batches list instead of a file."
        ),
    ),
    # --- OpenAI files -> would resolve to /v1/models -------------------------
    Probe(
        name="openai_files_retrieve",
        method="GET",
        path=f"/v1/files/{quote('../../v1/models', safe='')}",
        is_decoded=lambda sc, body, _h: sc == 200 and _looks_like_openai_models_list(body),
        description=(
            "Route handler builds /v1/files/<id>; the input collapses to "
            "/v1/models on the upstream."
        ),
    ),
    # --- OpenAI containers -> would resolve to /v1/models --------------------
    Probe(
        name="openai_container_retrieve",
        method="GET",
        path=f"/v1/containers/{quote('../../v1/models', safe='')}",
        is_decoded=lambda sc, body, _h: sc == 200 and _looks_like_openai_models_list(body),
        description="Container retrieve -> /v1/models.",
    ),
    # --- Fine-tuning jobs -> would resolve to /v1/models ---------------------
    # Route prefix is /v1/fine_tuning/jobs/<id>, which has THREE segments
    # above <id>, so we need three '../' to land on /v1/models.
    Probe(
        name="openai_finetuning_retrieve",
        method="GET",
        path=f"/v1/fine_tuning/jobs/{quote('../../../v1/models', safe='')}",
        is_decoded=lambda sc, body, _h: sc == 200 and _looks_like_openai_models_list(body),
        description=(
            "Fine-tuning retrieve -> /v1/models (needs 3x ../ to climb "
            "out of /v1/fine_tuning/jobs/)."
        ),
    ),
    # --- Anthropic batches -> would resolve to /v1/files (different family) --
    # From /v1/messages/batches/<id> we need ../../files to reach /v1/files
    # (not 'messages/batches/..' which would still be within batches).
    Probe(
        name="anthropic_batches_retrieve",
        method="GET",
        path=f"/anthropic/v1/batches/{quote('../../files', safe='')}",
        is_decoded=lambda sc, body, _h: sc == 200 and _looks_like_anthropic_files_list(body),
        description=(
            "Batch retrieve -> /v1/files list (a different endpoint "
            "family, so a files-shaped 200 is unambiguous evidence)."
        ),
    ),
]


# --- runner ----------------------------------------------------------------


VERDICT_COLORS = {
    "FAIL": "\x1b[31m",          # red
    "PASS": "\x1b[32m",          # green
    "INCONCLUSIVE": "\x1b[33m",  # yellow
    "ERROR": "\x1b[33m",         # yellow
}


def _format_verdict(verdict: str, use_color: bool) -> str:
    label = f"{verdict:12s}"
    if not use_color:
        return label
    return f"{VERDICT_COLORS.get(verdict, '')}{label}\x1b[0m"


def run_probe(
    client: httpx.Client,
    base: str,
    key: str,
    probe: Probe,
    timeout: float,
) -> Tuple[str, str]:
    url = f"{base.rstrip('/')}{probe.path}"
    headers = {"Authorization": f"Bearer {key}"}

    try:
        if probe.method == "GET":
            response = client.get(url, headers=headers, timeout=timeout)
        elif probe.method == "POST":
            response = client.post(
                url, headers=headers, json=probe.body or {}, timeout=timeout
            )
        else:
            return "ERROR", f"unsupported method {probe.method}"
    except httpx.HTTPError as exc:
        return "ERROR", f"{type(exc).__name__}: {exc}"

    try:
        body: Any = response.json()
    except ValueError:
        body = response.text

    try:
        decoded = probe.is_decoded(response.status_code, body, response.headers)
    except Exception as exc:
        return "ERROR", f"classifier crashed: {exc}"

    # Build a short summary for the output line.
    summary = f"HTTP {response.status_code}"
    if isinstance(body, dict):
        keys = list(body.keys())[:5]
        summary += f" body_keys={keys}"
    elif isinstance(body, str) and body:
        summary += f" body_prefix={body[:80]!r}"

    if decoded:
        verdict = "FAIL"
    elif response.status_code in (400, 404, 422):
        # The endpoint rejected the percent-encoded form: encoding held.
        verdict = "PASS"
    else:
        # Auth failures (401/403), rate limits (429), 5xx, or a 2xx
        # that didn't match any shape heuristic all leave us unable to
        # decide from this probe alone.
        verdict = "INCONCLUSIVE"

    return verdict, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Live-proxy lint for LiteLLM provider URL path-segment "
            "encoding. Fires inputs containing ../../ at affected "
            "endpoints and classifies each response as "
            "FAIL / PASS / INCONCLUSIVE / ERROR."
        ),
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base",
        default=os.environ.get("PROXY_BASE", "http://localhost:4000"),
        help="LiteLLM proxy base URL (env PROXY_BASE).",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("PROXY_KEY"),
        help="Virtual key for the proxy (env PROXY_KEY). A normal "
        "non-admin key is expected.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Run a single probe by name (see PROBES list).",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full response body and request path for each probe.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors (default: auto-detect TTY; NO_COLOR also honored).",
    )
    args = parser.parse_args()

    if not args.key:
        print("error: --key or PROXY_KEY is required", file=sys.stderr)
        return 2

    selected = PROBES
    if args.only:
        selected = [p for p in PROBES if p.name == args.only]
        if not selected:
            print(f"error: no probe named {args.only!r}", file=sys.stderr)
            print("available:", ", ".join(p.name for p in PROBES))
            return 2

    use_color = (
        sys.stdout.isatty()
        and not args.no_color
        and os.environ.get("NO_COLOR") is None
    )

    counts = {"FAIL": 0, "PASS": 0, "INCONCLUSIVE": 0, "ERROR": 0}

    with httpx.Client(follow_redirects=False) as client:
        print(f"[lint-url-encoding] target={args.base}  probes={len(selected)}")
        print(f"[lint-url-encoding] input={DEFAULT_INPUT}")
        print()
        for probe in selected:
            verdict, summary = run_probe(
                client, args.base, args.key, probe, args.timeout
            )
            marker = _format_verdict(verdict, use_color)
            print(f"  {marker}  {probe.name:32s}  {summary}")
            if args.verbose:
                # Re-issue the same call to capture the raw body for
                # display; cheap, read-only, aids diagnosis.
                url = f"{args.base.rstrip('/')}{probe.path}"
                print(f"           desc: {probe.description}")
                print(f"           path: {probe.path}")
                print(f"           url:  {url}")
                try:
                    response = (
                        client.get(
                            url,
                            headers={"Authorization": f"Bearer {args.key}"},
                            timeout=args.timeout,
                        )
                        if probe.method == "GET"
                        else client.post(
                            url,
                            headers={"Authorization": f"Bearer {args.key}"},
                            json=probe.body or {},
                            timeout=args.timeout,
                        )
                    )
                    text = response.text
                    print(f"           body: {text[:400]!r}")
                except Exception as exc:
                    print(f"           body: <re-fetch failed: {exc}>")
            counts[verdict] = counts.get(verdict, 0) + 1

    print()
    print(
        f"[lint-url-encoding] summary: "
        f"{counts['FAIL']} FAIL  "
        f"{counts['PASS']} PASS  "
        f"{counts['INCONCLUSIVE']} INCONCLUSIVE  "
        f"{counts['ERROR']} ERROR"
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
