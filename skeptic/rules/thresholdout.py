"""E5 R1: Thresholdout replay with explicitly supplied registered parameters.

The noise source is injected by the experiment plan. This module intentionally
chooses no distribution, threshold, sigma, or seed while Ryu's preregistration
is pending.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from skeptic.rules.common import ScoreArtifact, auroc

NoisePair = Callable[[int], tuple[float, float]]


@dataclass(frozen=True)
class Split:
    open_in: np.ndarray
    open_oos: np.ndarray
    guard_in: np.ndarray
    guard_oos: np.ndarray


def stratified_split(n_in: int, n_total: int, seed: int) -> Split:
    """Stable 50/50 split within each class; the same indices serve every rule query."""
    if type(seed) is not int or seed < 0 or type(n_in) is not int or type(n_total) is not int \
            or n_in < 2 or n_total - n_in < 2:
        raise ValueError("R1 split needs two or more ID and OOS items")

    def halves(indices: range) -> tuple[np.ndarray, np.ndarray]:
        ordered = sorted(indices, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).digest())
        pivot = len(ordered) // 2
        return np.asarray(sorted(ordered[:pivot]), dtype=np.int64), \
            np.asarray(sorted(ordered[pivot:]), dtype=np.int64)

    open_in, guard_in = halves(range(n_in))
    open_oos, guard_oos = halves(range(n_in, n_total))
    return Split(open_in, open_oos, guard_in, guard_oos)


def _gain(candidate: ScoreArtifact, incumbent: ScoreArtifact,
          inside: np.ndarray, outside: np.ndarray) -> float:
    return float(np.mean([auroc(c[inside], c[outside]) - auroc(b[inside], b[outside])
                          for c, b in zip(candidate.scores, incumbent.scores)]))


@dataclass
class Thresholdout:
    incumbent_sha: str
    incumbent: ScoreArtifact
    threshold: float
    split: Split
    noise: NoisePair

    @classmethod
    def from_baseline(cls, sha: str, artifact: ScoreArtifact, *, threshold: float,
                      split_seed: int, noise: NoisePair) -> "Thresholdout":
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("R1 threshold must be a registered nonnegative number")
        if not callable(noise):
            raise ValueError("R1 requires a registered noise source")
        return cls(sha, artifact, float(threshold),
                   stratified_split(artifact.n_in, artifact.scores.shape[1], split_seed), noise)

    def step(self, row: dict, artifact: ScoreArtifact | None, position: int) -> dict:
        before = self.incumbent_sha
        decision = {"rule": "R1", "sha": row["sha"], "incumbent": before,
                    "comparator": before, "delta_open": None, "delta_guard": None,
                    "released_gain": None, "used_guard": False,
                    "verdict": "REJECT", "reason": row.get("status", "missing_status")}
        if row.get("status") != "ok":
            return decision
        if artifact is None or artifact.scores.shape != self.incumbent.scores.shape \
                or artifact.n_in != self.incumbent.n_in:
            raise ValueError("R1 candidate and incumbent scores must align item by item")
        delta_open = _gain(artifact, self.incumbent, self.split.open_in, self.split.open_oos)
        delta_guard = _gain(artifact, self.incumbent, self.split.guard_in, self.split.guard_oos)
        threshold_noise, answer_noise = self.noise(position)
        if any(type(x) not in (int, float) or not math.isfinite(x)
               for x in (threshold_noise, answer_noise)):
            raise ValueError("R1 noise source returned non-finite values")
        used_guard = abs(delta_open - delta_guard) > self.threshold + threshold_noise
        released = delta_guard + answer_noise if used_guard else delta_open
        decision.update(delta_open=round(delta_open, 9), delta_guard=round(delta_guard, 9),
                        released_gain=round(released, 9), used_guard=used_guard,
                        reason="nonpositive_gain")
        if released > 0:
            decision["verdict"], decision["reason"] = "ACCEPT", "positive_gain"
            self.incumbent_sha, self.incumbent = row["sha"], artifact
        return decision
