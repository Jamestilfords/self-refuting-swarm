import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy-based logit uncertainty and pooled-feature Mahalanobis distance capture complementary failures of in-scope support. Robust train-derived centering and scaling put the two signals on a common bounded scale before combining them."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    ev_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    tr_dist = np.asarray(g.neg_log_density(tr.pooled), dtype=float)
    ev_dist = np.asarray(g.neg_log_density(ev.pooled), dtype=float)

    def robust_unit(train_values, values):
        med = np.median(train_values)
        mad = np.median(np.abs(train_values - med))
        scale = max(1.4826 * mad, np.std(train_values), 1e-8)
        z = np.clip((values - med) / scale, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    energy_u = robust_unit(tr_energy, ev_energy)
    dist_u = robust_unit(tr_dist, ev_dist)
    return energy_u + dist_u
