import numpy as np
from scipy.special import ndtri

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Logit uncertainty and encoder-space Mahalanobis distance capture complementary OOS signals. Training empirical-CDF calibration maps both signals to a shared normal-score scale, and their joint magnitude emphasizes examples anomalous in either or both views."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def normal_scores(reference, values):
        reference = np.asarray(reference, dtype=float)
        reference = reference[np.isfinite(reference)]
        if reference.size == 0:
            return np.zeros(np.asarray(values).shape, dtype=float)

        reference.sort()
        values = np.nan_to_num(
            np.asarray(values, dtype=float),
            nan=float(reference[-1]),
            posinf=float(reference[-1]),
            neginf=float(reference[0]),
        )
        q = np.searchsorted(reference, values, side="right").astype(float)
        q = (q + 0.5) / (reference.size + 1.0)
        q = np.clip(q, 1e-5, 1.0 - 1e-5)
        return ndtri(q)

    z_energy = normal_scores(train_energy, eval_energy)
    z_dist = normal_scores(train_dist, eval_dist)

    return np.sqrt(0.45 * z_energy * z_energy + 0.55 * z_dist * z_dist)
