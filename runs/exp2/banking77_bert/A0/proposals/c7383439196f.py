import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Logit uncertainty and feature-space atypicality provide complementary OOS evidence. Train-derived empirical tail calibration maps energy and Mahalanobis distance to a shared scale before combining them."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    def tail_scale(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        n = ref.size
        ranks = np.searchsorted(ref, values, side="right")
        p = (ranks + 1.0) / (n + 2.0)
        return -np.log1p(-p)

    e_tail = tail_scale(train_energy, eval_energy)
    d_tail = tail_scale(train_dist, eval_dist)
    return np.sqrt(e_tail * d_tail)
