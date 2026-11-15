"""Frozen records the critic passes between gates and writes into a decision row."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from skeptic.critic.er1 import ER1_COMMIT, auroc_scores


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    INERT = "INERT"
    SCORE_MATCHED = "SCORE-MATCHED"
    NO_EFFECT = "NO EFFECT"
    UNDERPOWERED = "UNDERPOWERED"
    MISCALIBRATES = "MISCALIBRATES"
    BUG = "BUG"


def json_safe(value):
    """Python objects json.dumps can emit with allow_nan=False."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("decision row contains a non-finite number")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"not JSON-safe: {type(value).__name__}")


def _as_1d(array, what: str) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    if out.ndim != 1:
        raise ValueError(f"{what} must be 1-D")
    if out.size == 0 or not np.isfinite(out).all():
        raise ValueError(f"{what} must be non-empty and finite")
    return out


@dataclass(frozen=True, eq=False)
class ScoreSet:
    name: str
    arm: str
    val_id: tuple[np.ndarray, ...]
    val_oos: tuple[np.ndarray, ...]
    status: str = "ok"
    parents: tuple[str, ...] = ()
    ece: tuple[float, ...] | None = None
    family: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parents", tuple(self.parents))
        if self.ece is not None:
            object.__setattr__(self, "ece", tuple(float(x) for x in self.ece))
        if self.status != "ok":
            return
        if len(self.val_id) != len(self.val_oos) or len(self.val_id) == 0:
            raise ValueError("val_id and val_oos need the same positive number of seeds")
        id_rows = [_as_1d(array, f"{self.name} val_id") for array in self.val_id]
        oos_rows = [_as_1d(array, f"{self.name} val_oos") for array in self.val_oos]
        if len({row.shape[0] for row in id_rows}) != 1 or len({row.shape[0] for row in oos_rows}) != 1:
            raise ValueError("score lengths differ across seeds")
        object.__setattr__(self, "val_id", tuple(id_rows))
        object.__setattr__(self, "val_oos", tuple(oos_rows))

    def seed_aurocs(self) -> np.ndarray:
        cached = self.__dict__.get("_seed_aurocs")
        if cached is not None:
            return cached
        values = np.array(
            [auroc_scores(ident, oos) for ident, oos in zip(self.val_id, self.val_oos)],
            dtype=np.float64,
        )
        object.__setattr__(self, "_seed_aurocs", values)
        return values

    def mean_auroc(self) -> float:
        return float(self.seed_aurocs().mean())

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        for array in (*self.val_id, *self.val_oos):
            block = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(np.asarray(block.shape, dtype=np.int64).tobytes())
            digest.update(block.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class BaselineSet:
    cell: str
    scores: Mapping[str, ScoreSet]
    g2_names: tuple[str, ...]
    plain_ece: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", dict(self.scores))
        object.__setattr__(self, "g2_names", tuple(self.g2_names))
        object.__setattr__(
            self,
            "plain_ece",
            {arm: tuple(float(x) for x in values) for arm, values in self.plain_ece.items()},
        )

    def resolve_parent(self, parent: str, arm: str) -> ScoreSet:
        return self.scores[f"{parent}_{arm}"]


@dataclass(frozen=True, eq=False)
class GateResult:
    gate: str
    outcome: str
    detail: dict


@dataclass(frozen=True, eq=False)
class Decision:
    sha: str
    cell: str
    incumbent: str
    comparator: str | None
    delta_val: float | None
    ci: tuple[float, float] | None
    p: float | None
    alpha_t: float | None
    fdr_t: int | None
    gates: dict[str, str]
    verdict: Verdict
    mdd: float | None
    detail: dict

    def to_row(self, idx: int | None = None) -> dict:
        row = {
            "sha": self.sha,
            "cell": self.cell,
            "idx": idx,
            "incumbent": self.incumbent,
            "comparator": self.comparator,
            "delta_val": self.delta_val,
            "ci": None if self.ci is None else [self.ci[0], self.ci[1]],
            "p": self.p,
            "alpha_t": self.alpha_t,
            "fdr_t": self.fdr_t,
            "gates": dict(self.gates),
            "verdict": self.verdict.value,
            "mdd": self.mdd,
            "detail": self.detail,
            "er1_commit": ER1_COMMIT,
        }
        return json_safe(row)

    def to_json(self) -> str:
        return json.dumps(self.to_row(), sort_keys=True, allow_nan=False)
