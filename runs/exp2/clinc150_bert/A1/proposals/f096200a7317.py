import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Bottleneck and pooled embeddings encode complementary notions of local in-scope support. "
    "A query that is isolated in either representation space receives a high train-calibrated novelty score."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0111", "m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = 15

    tr_b = ctx.h.knn(tr.feats, tr.feats, k=k, l2=True)
    tr_p = ctx.h.knn(tr.pooled, tr.pooled, k=k, l2=True)
    ev_b = ctx.h.knn(tr.feats, ev.feats, k=k, l2=True)
    ev_p = ctx.h.knn(tr.pooled, ev.pooled, k=k, l2=True)

    tr_b = np.nan_to_num(tr_b, nan=0.0, posinf=1e12, neginf=0.0)
    tr_p = np.nan_to_num(tr_p, nan=0.0, posinf=1e12, neginf=0.0)
    ev_b = np.nan_to_num(ev_b, nan=0.0, posinf=1e12, neginf=0.0)
    ev_p = np.nan_to_num(ev_p, nan=0.0, posinf=1e12, neginf=0.0)

    sb = np.sort(tr_b)
    sp = np.sort(tr_p)
    pb = np.searchsorted(sb, ev_b, side="right") / float(len(sb) + 1)
    pp = np.searchsorted(sp, ev_p, side="right") / float(len(sp) + 1)

    pb = np.clip(pb, 0.0, 1.0 - 1e-6)
    pp = np.clip(pp, 0.0, 1.0 - 1e-6)
    return 0.5 * (-np.log1p(-pb) - np.log1p(-pp))
