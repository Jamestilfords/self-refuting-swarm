import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy and pooled-feature Mahalanobis distance are standardized using only "
    "training statistics, then distance is scored as an unusually large residual "
    "given classifier uncertainty. This emphasizes geometrically unsupported "
    "predictions rather than a fixed monotone sum of the two signals."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0117", "m_0118"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_tr = np.nan_to_num(e_tr, nan=0.0, posinf=1e6, neginf=-1e6)
    d_tr = np.nan_to_num(d_tr, nan=0.0, posinf=1e6, neginf=-1e6)
    e_ev = np.nan_to_num(e_ev, nan=0.0, posinf=1e6, neginf=-1e6)
    d_ev = np.nan_to_num(d_ev, nan=0.0, posinf=1e6, neginf=-1e6)

    e_mu = float(np.mean(e_tr))
    d_mu = float(np.mean(d_tr))
    e_sd = max(float(np.std(e_tr)), 1e-6)
    d_sd = max(float(np.std(d_tr)), 1e-6)

    ez_tr = (e_tr - e_mu) / e_sd
    dz_tr = (d_tr - d_mu) / d_sd

    denom = float(np.dot(ez_tr, ez_tr)) + 1e-8
    beta = float(np.dot(ez_tr, dz_tr) / denom)
    residual_tr = dz_tr - beta * ez_tr
    residual_sd = max(float(np.std(residual_tr)), 1e-6)

    ez_ev = (e_ev - e_mu) / e_sd
    dz_ev = (d_ev - d_mu) / d_sd
    residual_ev = (dz_ev - beta * ez_ev) / residual_sd

    scores = ez_ev + np.maximum(residual_ev, 0.0)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
