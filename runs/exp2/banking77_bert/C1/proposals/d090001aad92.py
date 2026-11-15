import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-space Mahalanobis distance are mapped to train-calibrated empirical tail scales before a probabilistic-OR fusion. This emphasizes examples that are unusually uncertain or unusually far from every in-scope class without allowing either raw scale to dominate."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    energy_ref = np.sort(np.nan_to_num(
        train_energy, nan=0.0, posinf=1e12, neginf=-1e12
    ))
    dist_ref = np.sort(np.nan_to_num(
        train_dist, nan=0.0, posinf=1e12, neginf=-1e12
    ))

    eval_energy = np.nan_to_num(
        eval_energy, nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_dist = np.nan_to_num(
        eval_dist, nan=0.0, posinf=1e12, neginf=-1e12
    )

    e_tail = np.searchsorted(energy_ref, eval_energy, side="right") / float(len(energy_ref))
    d_tail = np.searchsorted(dist_ref, eval_dist, side="right") / float(len(dist_ref))

    scores = e_tail + d_tail - e_tail * d_tail
    return np.asarray(scores, dtype=float)
