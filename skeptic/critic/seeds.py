"""Seed-majority vote. ER-1's rule: a flag holds when it holds in at least half the seeds."""

from __future__ import annotations

import math
from collections.abc import Sequence


def votes_required(n: int, rule: str = "ceil_half") -> int:
    if n < 1:
        raise ValueError("need at least one seed")
    if rule != "ceil_half":
        raise ValueError(f"unknown seed_majority rule: {rule}")
    return math.ceil(n / 2)


def majority(flags: Sequence[bool]) -> bool:
    """True when at least ceil(n / 2) entries are true."""
    flags = list(flags)
    if not flags:
        raise ValueError("majority of an empty vote")
    return sum(bool(flag) for flag in flags) >= votes_required(len(flags))
