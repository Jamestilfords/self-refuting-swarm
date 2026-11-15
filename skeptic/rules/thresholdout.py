"""E5 R1: Thresholdout replay with registered Laplace parameters."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from skeptic.rules.common import CELLS, ScoreArtifact, auroc


class NoiseSource(Protocol):
    def threshold(self) -> float: ...
    def query(self) -> float: ...
    def answer(self) -> float: ...


@dataclass
class LaplaceNoise:
    rng: np.random.Generator
    sigma: float

    def threshold(self) -> float:
        return float(self.rng.laplace(scale=2 * self.sigma))

    def query(self) -> float:
        return float(self.rng.laplace(scale=4 * self.sigma))

    def answer(self) -> float:
        return float(self.rng.laplace(scale=self.sigma))


def laplace_noise(cell: str, ordering: int, *, sigma: float, seed: int) -> LaplaceNoise:
    """Independent PCG64 stream from the registered seed, cell, and ordering."""
    if cell not in CELLS or type(ordering) is not int or ordering < 0 \
            or type(seed) is not int or seed < 0 \
            or type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid registered R1 noise parameters")
    cell_index = sorted(CELLS).index(cell)
    rng = np.random.default_rng(np.random.SeedSequence([seed, cell_index, ordering]))
    return LaplaceNoise(rng, float(sigma))


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


def split_aurocs(artifact: ScoreArtifact, split: Split) -> tuple[np.ndarray, np.ndarray]:
    """Cache the two five-seed AUROC vectors for a fixed registered split."""
    return (np.asarray([auroc(seed[split.open_in], seed[split.open_oos]) for seed in artifact.scores]),
            np.asarray([auroc(seed[split.guard_in], seed[split.guard_oos]) for seed in artifact.scores]))


@dataclass
class Thresholdout:
    incumbent_sha: str
    incumbent: ScoreArtifact
    threshold: float
    split: Split
    noise: NoiseSource
    threshold_noise: float
    split_scores: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None

    @classmethod
    def from_baseline(cls, sha: str, artifact: ScoreArtifact, *, threshold: float,
                      split_seed: int, noise: NoiseSource,
                      split_scores: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None
                      ) -> "Thresholdout":
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("R1 threshold must be a registered nonnegative number")
        if any(not callable(getattr(noise, name, None)) for name in ("threshold", "query", "answer")):
            raise ValueError("R1 requires a registered noise source")
        threshold_noise = noise.threshold()
        if not math.isfinite(threshold_noise):
            raise ValueError("R1 threshold noise must be finite")
        return cls(sha, artifact, float(threshold),
                   stratified_split(artifact.n_in, artifact.scores.shape[1], split_seed),
                   noise, threshold_noise, split_scores)

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
        if self.split_scores is None:
            delta_open = _gain(artifact, self.incumbent, self.split.open_in, self.split.open_oos)
            delta_guard = _gain(artifact, self.incumbent, self.split.guard_in, self.split.guard_oos)
        else:
            candidate_open, candidate_guard = self.split_scores[row["sha"]]
            incumbent_open, incumbent_guard = self.split_scores[before]
            delta_open = float(np.mean(candidate_open - incumbent_open))
            delta_guard = float(np.mean(candidate_guard - incumbent_guard))
        query_noise = self.noise.query()
        if not math.isfinite(query_noise):
            raise ValueError("R1 query noise must be finite")
        used_guard = abs(delta_open - delta_guard) > self.threshold + self.threshold_noise + query_noise
        if used_guard:
            answer_noise = self.noise.answer()
            next_threshold_noise = self.noise.threshold()
            if not math.isfinite(answer_noise) or not math.isfinite(next_threshold_noise):
                raise ValueError("R1 answer or threshold noise must be finite")
            released = delta_guard + answer_noise
            self.threshold_noise = next_threshold_noise
        else:
            released = delta_open
        decision.update(delta_open=round(delta_open, 9), delta_guard=round(delta_guard, 9),
                        released_gain=round(released, 9), used_guard=used_guard,
                        reason="nonpositive_gain")
        if released > 0:
            decision["verdict"], decision["reason"] = "ACCEPT", "positive_gain"
            self.incumbent_sha, self.incumbent = row["sha"], artifact
        return decision
