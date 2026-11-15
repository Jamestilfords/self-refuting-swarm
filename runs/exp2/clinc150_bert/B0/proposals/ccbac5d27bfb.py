import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Training-set empirical CDFs calibrate energy and encoder Mahalanobis distance onto the same OOS-probability scale. A noisy-OR fusion detects queries that are anomalous in either classifier confidence or feature-manifold support without requiring evaluation-set normalization."
SOURCE = "https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0017", "m_0023"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_energy = np.nan_to_num(train_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    train_dist = np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=0.0)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e12, neginf=0.0)

    energy_ref = np.sort(train_energy)
    dist_ref = np.sort(train_dist)
    n = float(len(train_energy))

    energy_q = (np.searchsorted(energy_ref, eval_energy, side="right") + 0.5) / (n + 1.0)
    dist_q = (np.searchsorted(dist_ref, eval_dist, side="right") + 0.5) / (n + 1.0)

    return 1.0 - (1.0 - energy_q) * (1.0 - dist_q)
