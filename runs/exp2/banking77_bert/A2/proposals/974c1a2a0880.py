import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Confident predictions retain an energy-dominated score, while samples with unusually uncertain energy receive an additional encoder-space distance penalty. This conditional fusion targets OOS examples whose classifier uncertainty makes representation novelty most informative."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    e_tr = ctx.h.energy(tr.logits)
    e_ev = ctx.h.energy(ev.logits)
    d_tr = g.min_mahalanobis_sq(tr.pooled)
    d_ev = g.min_mahalanobis_sq(ev.pooled)

    e_med = np.median(e_tr)
    d_med = np.median(d_tr)
    e_scale = np.quantile(e_tr, 0.75) - np.quantile(e_tr, 0.25)
    d_scale = np.quantile(d_tr, 0.75) - np.quantile(d_tr, 0.25)
    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    uncertainty_center = np.quantile(e_tr, 0.75)
    gate = 1.0 / (1.0 + np.exp(-np.clip((e_ev - uncertainty_center) / e_scale, -30.0, 30.0)))

    energy_score = (e_ev - e_med) / e_scale
    distance_score = (d_ev - d_med) / d_scale
    return energy_score + gate * distance_score
