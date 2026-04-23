"""Shared URL-encoding helpers for provider transformations.

Contract:

* ``encode_path_segment`` raises ``ValueError`` on ``None`` / ``""``.
  A single path segment must always be present; returning ``""`` would
  quietly collapse ``/{id}`` to ``/`` and potentially reach a list
  endpoint instead of the intended ``/{id}`` target.
* ``encode_url_path`` accepts ``None`` / ``""`` → ``""``. It is called
  from paths that legitimately pass an empty placeholder.
* Whole-segment ``.`` or ``..`` → ``ValueError`` (not a legitimate
  identifier, and ``quote`` leaves ``.`` alone on its own).
* ``%`` is always re-encoded to ``%25`` so percent-encoded input cannot
  round-trip through a second decode.
"""

from typing import Optional
from urllib.parse import quote


def _reject_dot_segment(segment: str, full: str) -> None:
    if segment in ("..", "."):
        raise ValueError(f"Illegal path segment in identifier: {full!r}")


def encode_path_segment(segment: Optional[str]) -> str:
    """Percent-encode a single URL path segment.

    Encodes every reserved character (including ``/``, ``?``, ``#``, ``%``)
    so the value is interpolated as one atomic path component. Raises
    ``ValueError`` on empty / ``None`` input (a missing identifier would
    silently collapse ``/{id}`` to ``/``) and on whole-segment ``.`` / ``..``.
    """
    if segment is None or segment == "":
        raise ValueError("identifier is required, got empty or None")
    str_segment = str(segment)
    _reject_dot_segment(str_segment, str_segment)
    return quote(str_segment, safe="")


def encode_url_path(path: Optional[str]) -> str:
    """Percent-encode a multi-segment URL path.

    Preserves ``/`` between segments so legitimate provider identifiers
    like Cloudflare's ``@cf/meta/llama-3.1-8b-instruct`` or Bytez's
    ``google/gemma-3-4b-it`` still work, and leaves ``@`` unencoded (a
    valid ``pchar`` per RFC 3986). Rejects any ``..`` / ``.`` / empty
    segment, and percent-encodes ``?``, ``#``, ``%``, ``:``.

    ``:`` is encoded so inputs like ``evil.com:80/x`` cannot be
    interpreted as an authority by a downstream ``urljoin``.
    """
    if path is None or path == "":
        return ""
    str_path = str(path)
    for segment in str_path.split("/"):
        if segment == "":
            raise ValueError(
                f"Empty path segment in identifier: {str_path!r}"
            )
        _reject_dot_segment(segment, str_path)
    return quote(str_path, safe="/@")
