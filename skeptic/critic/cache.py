"""Bootstrap cache. Keyed by array bytes, not by score names."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

from skeptic.critic.er1 import ER1_COMMIT
from skeptic.critic.types import ScoreSet


class StatsCache(Protocol):
    def get(self, key: str) -> dict | None: ...

    def put(self, key: str, value: dict) -> None: ...


def stats_key(candidate: ScoreSet, comparator: ScoreSet, cell: str, replicates: int, seed: int) -> str:
    raw = (
        f"{candidate.content_hash()}|{comparator.content_hash()}|{cell}"
        f"|B={replicates}|seed={seed}|{ER1_COMMIT}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class MemoryStatsCache:
    def __init__(self) -> None:
        self._values: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        return self._values.get(key)

    def put(self, key: str, value: dict) -> None:
        self._values[key] = value


class JsonDirStatsCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
