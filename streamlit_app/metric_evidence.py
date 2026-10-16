"""Render recorded research scores without manufacturing missing measurements."""

from math import isfinite
from numbers import Real


def optional_score(value: object) -> float | None:
    """Allow measured finite numeric scores, including valid zero and negative scores."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if isfinite(number) else None


def score_text(value: float | None) -> str:
    """Show a visibly unavailable value when its source is absent or invalid."""
    return f"{value:+.5f}" if value is not None else "—"
