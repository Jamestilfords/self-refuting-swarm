import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy captures classifier uncertainty while bottleneck Mahalanobis distance measures "
    "departure from learned intent geometry. Train-fitted empirical CDFs put both signals on "
    "a common robust scale before combining them."
)
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0049"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    def cdf_scale(train_values, values):
        train_values = np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12)
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ordered = np.sort(train_values)
        ranks = np.searchsorted(ordered, values, side="right")
        return (ranks + 0.5) / (ordered.size + 1.0)

    energy_q = cdf_scale(train_energy, eval_energy)
    dist_q = cdf_scale(train_dist, eval_dist)

    return np.asarray(0.5 * energy_q + 0.5 * dist_q, dtype=float)
