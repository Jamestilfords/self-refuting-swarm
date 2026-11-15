import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "For confident predictions this retains the energy score, while queries below a training-calibrated confidence threshold are ranked by pooled-feature distance instead. This lets feature novelty dominate precisely where logits are ambiguous."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    d_tr = g.min_mahalanobis_sq(tr.pooled)
    d_ev = g.min_mahalanobis_sq(ev.pooled)

    e_tr = ctx.h.energy(tr.logits)
    e_ev = ctx.h.energy(ev.logits)
    msp_tr = ctx.h.msp(tr.logits)
    msp_ev = ctx.h.msp(ev.logits)

    e_center = float(np.median(e_tr))
    e_scale = max(float(np.quantile(e_tr, 0.75) - np.quantile(e_tr, 0.25)), 1e-6)
    d_center = float(np.median(d_tr))
    d_scale = max(float(np.quantile(d_tr, 0.75) - np.quantile(d_tr, 0.25)), 1e-6)

    confidence_cutoff = float(np.quantile(msp_tr, 0.40))
    energy_score = (e_ev - e_center) / e_scale
    distance_score = (d_ev - d_center) / d_scale

    uncertain = msp_ev < confidence_cutoff
    return np.where(uncertain, distance_score, energy_score)
