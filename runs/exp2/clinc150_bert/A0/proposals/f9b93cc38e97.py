import numpy as np

FAMILY = "fusion"
ARM = "edl"
RATIONALE = (
    "Train-calibrated upper-tail probabilities put classifier energy and pooled-space "
    "Mahalanobis novelty on a common scale. Their probabilistic union flags queries "
    "that are either uncertain in logit space or geometrically unsupported."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0063", "m_0079"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def upper_tail(train_values, values):
        ref = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12))
        x = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(ref, x, side="right") / float(ref.size + 1)

    e_tail = upper_tail(train_energy, eval_energy)
    d_tail = upper_tail(train_dist, eval_dist)

    return 1.0 - (1.0 - e_tail) * (1.0 - d_tail)
