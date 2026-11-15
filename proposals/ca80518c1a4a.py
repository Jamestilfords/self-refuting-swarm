import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Low-energy confident queries retain the energy ranking, while queries whose "
    "energy is unusually uncertain receive an additional penalty only when they "
    "also lie beyond the train-supported pooled-feature distance."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0012", "m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.median(train_energy)
    e_hi = np.quantile(train_energy, 0.90)
    e_scale = max(e_hi - e_med, 1e-6)

    d_med = np.median(train_dist)
    d_hi = np.quantile(train_dist, 0.90)
    d_scale = max(d_hi - d_med, 1e-6)

    energy_z = (eval_energy - e_med) / e_scale

    # Confident (below-median energy) items receive no feature correction.
    uncertainty_gate = np.clip((eval_energy - e_med) / e_scale, 0.0, 1.0)
    unsupported_distance = np.maximum((eval_dist - d_med) / d_scale, 0.0)

    return energy_z + uncertainty_gate * unsupported_distance
