"""Local bookkeeping for bounded agent runs. This module never contacts a service."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class BudgetExhausted(RuntimeError):
    """A persisted run limit was reached before another API call."""


class ConfigurationError(ValueError):
    """Only local, credential-free messages may use this exception type."""


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class RunBudget:
    """Charge before calling a provider; failed/interrupted calls keep their reservation.

    Token reservations use UTF-8 bytes plus output allowance and framing headroom.
    This deliberately overestimates normal text inputs. Successful calls settle to
    reported usage. Call/output/search limits are hard bounds, not dollar limits.
    """

    def __init__(self, path: Path, *, max_calls: int, max_searches: int,
                 max_tokens: int, max_output_tokens: int):
        if min(max_calls, max_tokens, max_output_tokens) <= 0 or max_searches < 0:
            raise ValueError("Run limits must be positive (searches may be zero)")
        self.path = path
        self.limits = dict(max_calls=max_calls, max_searches=max_searches,
                           max_tokens=max_tokens, max_output_tokens=max_output_tokens)
        self.used = dict(calls=0, searches=0, charged_tokens=0, input_tokens=0, output_tokens=0)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["limits"] != self.limits:
                raise ConfigurationError("Resume must use the original run limits; choose a new run directory to change them")
            self.used = saved["used"]
            if set(self.used) != {"calls", "searches", "charged_tokens", "input_tokens", "output_tokens"} \
                    or any(type(v) is not int or v < 0 for v in self.used.values()):
                raise ValueError("Invalid saved usage")

    def save(self):
        atomic_json(self.path, {"limits": self.limits, "used": self.used})

    def search(self):
        if self.used["searches"] >= self.limits["max_searches"]:
            raise BudgetExhausted("Exa search limit reached")
        self.used["searches"] += 1
        self.save()

    def llm(self, provider):
        def call(system: str, user: str):
            reservation = len((system + user).encode("utf-8")) + self.limits["max_output_tokens"] + 1024
            if self.used["calls"] >= self.limits["max_calls"]:
                raise BudgetExhausted("LLM call limit reached")
            if self.used["charged_tokens"] + reservation > self.limits["max_tokens"]:
                raise BudgetExhausted("Token allowance too small for the next request")
            self.used["calls"] += 1
            self.used["charged_tokens"] += reservation
            self.save()
            text, usage = provider(system, user)
            counts = {key: usage.get(key) for key in ("input_tokens", "output_tokens")}
            if all(type(v) is int and v >= 0 for v in counts.values()):
                self.used["charged_tokens"] += sum(counts.values()) - reservation
                for key, value in counts.items():
                    self.used[key] += value
                self.save()
            return text, usage
        return call
