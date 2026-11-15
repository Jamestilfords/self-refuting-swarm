import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-feature Mahalanobis distance are mapped to train-fitted upper-tail probabilities before fusion. Their product emphasizes queries that are simultaneously classifier-uncertain and poorly supported in representation space, rather than reproducing an additive ranking."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0048", "m_0054", "m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def upper_tail(reference, values):
        ref = np.sort(np.nan_to_num(reference, nan=0.0, posinf=1e12, neginf=-1e12))
        val = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        n = max(ref.size, 1)
        rank = np.searchsorted(ref, val, side="right")
        return np.clip((rank + 0.5) / (n + 1.0), 1e-6, 1.0 - 1e-6)

    e_tail = upper_tail(train_energy, eval_energy)
    d_tail = upper_tail(train_dist, eval_dist)

    return np.sqrt(e_tail * d_tail)
