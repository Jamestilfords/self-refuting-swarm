import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and feature-space Mahalanobis distance capture complementary confidence and representation-support failures. Train-calibrated empirical percentiles put both signals on a common, distribution-free scale before fusion."
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

    def percentile(train_values, values):
        ref = np.sort(np.nan_to_num(
            train_values, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        query = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        left = np.searchsorted(ref, query, side="left")
        right = np.searchsorted(ref, query, side="right")
        return (left + right) / (2.0 * max(ref.size, 1))

    e_pct = percentile(train_energy, eval_energy)
    d_pct = percentile(train_dist, eval_dist)

    eps = 1e-4
    e_logit = np.log((e_pct + eps) / (1.0 - e_pct + eps))
    d_logit = np.log((d_pct + eps) / (1.0 - d_pct + eps))
    return 0.5 * (e_logit + d_logit)
