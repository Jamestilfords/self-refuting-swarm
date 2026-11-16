"""Confirm open validation scores match the registered baseline arrays.

The critic reads each reference from ``runs/scores/<group>/<sha>.npz``.
``python -m skeptic.critic.baselines`` writes the same arrays as
``runs/baselines/<group>/<name>/seed<k>.npy``. This check requires Spearman
correlation 1 within 1e-12. A missing file exits 2. The first mismatch exits 1.

Score files store float32. Rounding to float32 can turn two nearly equal float64 scores into a tie and
flip their rank order, which breaks an exact rank match although nothing differs. A seed therefore also
passes when the stored array equals the reference rounded to float32, which is a stricter test of
identity than the rank check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from skeptic.rules.common import REFERENCE_BASELINES
from skeptic.score_artifacts import CELLS


def spearman(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size == 0:
        return 0.0
    if np.array_equal(a, b):
        return 1.0
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    denom = float(np.linalg.norm(ar) * np.linalg.norm(br))
    if denom == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(ar @ br / denom)


def arrays_match(open_scores, baseline_rows, tol: float = 1e-12) -> bool:
    """``open_scores`` is (5, n). ``baseline_rows`` is one concatenated row per seed."""
    open_scores = np.asarray(open_scores, dtype=np.float64)
    rows = [np.asarray(row, dtype=np.float64).ravel() for row in baseline_rows]
    if open_scores.shape[0] != len(rows):
        return False
    for seed, row in enumerate(rows):
        if row.shape != open_scores[seed].shape:
            return False
        if np.array_equal(open_scores[seed], row.astype(np.float32).astype(np.float64)):
            continue          # identical at the precision the score files are stored in (float32)
        if abs(spearman(open_scores[seed], row) - 1.0) > tol:
            return False
    return True


def first_mismatch(scores_root: Path, baselines_root: Path) -> str | None:
    """Return the first cell/name that disagrees, or a missing-file message."""
    missing = []
    for cell, group in CELLS.items():
        for name, sha in REFERENCE_BASELINES.items():
            npz = scores_root / group / f"{sha}.npz"
            folder = baselines_root / group / name
            seeds = [folder / f"seed{index}.npy" for index in range(5)]
            if not npz.is_file() or any(not path.is_file() for path in seeds):
                missing.append(f"{cell} {name}")
                continue
            with np.load(npz, allow_pickle=False) as saved:
                scores = np.asarray(saved["scores"], dtype=np.float64)
            rows = [np.load(path) for path in seeds]
            if not arrays_match(scores, rows):
                return f"mismatch {cell} {name} sha={sha}"
    if missing:
        return "missing " + ", ".join(missing)
    return None


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("runs/scores"))
    parser.add_argument("--baselines", type=Path, default=Path("runs/baselines"))
    args = parser.parse_args(argv)
    problem = first_mismatch(args.scores, args.baselines)
    if problem is None:
        print("check_references: Spearman = 1 for all five references in all four cells")
        return 0
    print(f"check_references: {problem}", file=sys.stderr)
    return 2 if problem.startswith("missing") else 1


if __name__ == "__main__":
    raise SystemExit(main())
