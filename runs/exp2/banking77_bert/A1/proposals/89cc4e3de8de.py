import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode tightly constrained in-scope structure, so OOS queries should have unusually large trailing-subspace residuals. Combining independently calibrated bottleneck and encoder residual percentiles reduces reliance on a single representation."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    p_feat = ctx.h.partial(tr.feats, tr.labels, top_k=128)
    p_pool = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    tr_feat = np.nan_to_num(p_feat.bottom(tr.feats), nan=0.0, posinf=1e12, neginf=-1e12)
    tr_pool = np.nan_to_num(p_pool.bottom(tr.pooled), nan=0.0, posinf=1e12, neginf=-1e12)
    ev_feat = np.nan_to_num(p_feat.bottom(ev.feats), nan=0.0, posinf=1e12, neginf=-1e12)
    ev_pool = np.nan_to_num(p_pool.bottom(ev.pooled), nan=0.0, posinf=1e12, neginf=-1e12)

    ref_feat = np.sort(tr_feat)
    ref_pool = np.sort(tr_pool)

    q_feat = np.searchsorted(ref_feat, ev_feat, side="right") / float(len(ref_feat))
    q_pool = np.searchsorted(ref_pool, ev_pool, side="right") / float(len(ref_pool))

    return 0.5 * (q_feat + q_pool)
