"""Validated per-item score artifacts shared by scoring and replay."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
CELLS = {"clinc150/bert": "clinc150_bert", "clinc150/roberta": "clinc150_roberta",
         "banking77_oos/bert": "banking77_bert", "banking77_oos/roberta": "banking77_roberta"}
_SHA = re.compile(r"[0-9a-f]{12}\Z")

@dataclass(frozen=True)
class ScoreArtifact:
    scores: np.ndarray
    n_in: int
    pred: np.ndarray | None = None
    conf: np.ndarray | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, np.float64)
        if type(self.n_in) is not int or scores.ndim != 2 or scores.shape[0] != 5 or scores.shape[1] < 4 \
                or not np.isfinite(scores).all() or not 0 < self.n_in < scores.shape[1]:
            raise ValueError("Expected five finite, aligned seed score arrays")
        object.__setattr__(self, "scores", scores)
        if (self.pred is None) != (self.conf is None):
            raise ValueError("Predictions and confidences must appear together")
        if self.pred is not None:
            pred, conf = np.asarray(self.pred), np.asarray(self.conf, np.float64)
            if pred.shape != scores.shape or conf.shape != scores.shape \
                    or not np.issubdtype(pred.dtype, np.integer) or (pred < 0).any() or not np.isfinite(conf).all() \
                    or (conf < 0).any() or (conf > 1).any():
                raise ValueError("Invalid probability-derived arrays")
            object.__setattr__(self, "pred", pred)
            object.__setattr__(self, "conf", conf)

def load_scores(open_root: str | Path, cell: str, sha: str,
                *, source: str) -> ScoreArtifact:
    if cell not in CELLS or not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ValueError("Unknown cell or proposal hash")
    if source not in ("legacy", "isolated"):
        raise ValueError("Score source must be explicit: legacy or isolated")
    directory = "scores" if source == "legacy" else "scores_isolated"
    path = Path(open_root) / directory / CELLS[cell] / f"{sha}.npz"
    if any(p.is_symlink() for p in (Path(open_root) / directory, path.parent, path)):
        raise ValueError("Score artifact paths cannot be symlinks")
    with np.load(path, allow_pickle=False) as cache:
        if not {"scores", "n_in"}.issubset(cache.files) \
                or set(cache.files) - {"scores", "n_in", "runtime_s", "pred", "conf",
                                            "scorer", "proposal_sha"}:
            raise ValueError("Invalid score artifact keys")
        if source == "isolated" and ("scorer" not in cache.files or "proposal_sha" not in cache.files
                                      or str(cache["scorer"]) != "isolated-e3-v1"
                                      or str(cache["proposal_sha"]) != sha):
            raise ValueError("Isolated score artifact lacks verified provenance")
        n_in = np.asarray(cache["n_in"])
        if n_in.shape != () or not np.issubdtype(n_in.dtype, np.integer):
            raise ValueError("Invalid score artifact split size")
        if "runtime_s" in cache.files:
            runtime = np.asarray(cache["runtime_s"])
            if runtime.shape != () or not np.isfinite(runtime) or runtime < 0:
                raise ValueError("Invalid score runtime")
        return ScoreArtifact(cache["scores"], int(n_in),
                             cache["pred"] if "pred" in cache.files else None,
                             cache["conf"] if "conf" in cache.files else None)

def auroc(inside: np.ndarray, outside: np.ndarray) -> float:
    inside, outside = np.asarray(inside, np.float64), np.asarray(outside, np.float64)
    if inside.ndim != 1 or outside.ndim != 1 or not len(inside) or not len(outside):
        raise ValueError("AUROC requires nonempty one-dimensional splits")
    scores = np.concatenate((inside, outside))
    if not np.isfinite(scores).all():
        raise ValueError("AUROC scores must be finite")
    ordered = np.sort(scores)
    left = np.searchsorted(ordered, outside, side="left")
    right = np.searchsorted(ordered, outside, side="right")
    rank_sum = ((left + right + 1) / 2).sum()
    return float((rank_sum - len(outside) * (len(outside) + 1) / 2)
                 / (len(inside) * len(outside)))

def seed_mean_auroc(artifact: ScoreArtifact) -> float:
    return float(np.mean([auroc(row[:artifact.n_in], row[artifact.n_in:])
                          for row in artifact.scores]))
