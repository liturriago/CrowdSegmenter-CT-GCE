"""String and duration formatting helpers."""

from __future__ import annotations


def format_time(seconds: float) -> str:
    """Format elapsed seconds as ``MM:SS``."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
