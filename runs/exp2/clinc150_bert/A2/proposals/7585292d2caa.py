import numpy as np
from scipy.stats import norm

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and bottleneck-space class distance capture complementary classifier uncertainty and geometric novelty. Mapping each score through its training empirical CDF and then to normal quantiles puts their in-scope tail behavior on a common, robust scale before combining them."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0105", "m_0121"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    def normal_tail(values, reference):
        ref = np.sort(np.nan_to_num(reference, nan=0.0, posinf=1e12, neginf=-1e12))
        val = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        n = ref.size
        p = (np.searchsorted(ref, val, side="right") + 0.5) / (n + 1.0)
        p = np.clip(p, 1.0 / (n + 1.0), n / (n + 1.0))
        return norm.ppf(p)

    energy_z = normal_tail(eval_energy, train_energy)
    distance_z = normal_tail(eval_dist, train_dist)
    return np.nan_to_num(0.5 * (energy_z + distance_z), nan=0.0, posinf=12.0, neginf=-12.0)
