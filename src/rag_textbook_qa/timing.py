"""Read optional elapsed times without inventing measurements for bad metadata."""

from __future__ import annotations

import math
from typing import Any


def finite_seconds(value: Any) -> float | None:
    """Return a finite nonnegative duration, or None for an unknown measurement."""
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None
