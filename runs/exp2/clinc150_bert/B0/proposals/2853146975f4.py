import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy remains the primary score for confident in-scope-like items, while a smooth "
    "training-calibrated gate increasingly incorporates encoder Mahalanobis distance when energy is uncertain."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    e_tr = ctx.h.energy(tr.logits)
    e_ev = ctx.h.energy(ev.logits)
    d_tr = g.min_mahalanobis_sq(tr.pooled)
    d_ev = g.min_mahalanobis_sq(ev.pooled)

    e_lo, e_hi = np.quantile(e_tr, [0.25, 0.75])
    d_lo, d_hi = np.quantile(d_tr, [0.25, 0.75])
    e_scale = max(float(e_hi - e_lo), 1e-6)
    d_scale = max(float(d_hi - d_lo), 1e-6)

    e_score = (e_ev - e_lo) / e_scale
    d_score = (d_ev - d_lo) / d_scale

    uncertainty_center = float(np.quantile(e_tr, 0.70))
    gate = 1.0 / (1.0 + np.exp(-np.clip((e_ev - uncertainty_center) / e_scale, -40.0, 40.0)))

    return e_score + gate * d_score
