import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that in-scope representations tend to satisfy. "
    "Combining train-calibrated trailing-direction deviations from bottleneck and pooled spaces captures violations at two semantic levels."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pb = ctx.h.partial(tr.feats, tr.labels, top_k=128)
    pp = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    tr_b = np.nan_to_num(np.asarray(pb.bottom(tr.feats)).reshape(-1),
                         nan=0.0, posinf=1e12, neginf=-1e12)
    tr_p = np.nan_to_num(np.asarray(pp.bottom(tr.pooled)).reshape(-1),
                         nan=0.0, posinf=1e12, neginf=-1e12)
    ev_b = np.nan_to_num(np.asarray(pb.bottom(ev.feats)).reshape(-1),
                         nan=0.0, posinf=1e12, neginf=-1e12)
    ev_p = np.nan_to_num(np.asarray(pp.bottom(ev.pooled)).reshape(-1),
                         nan=0.0, posinf=1e12, neginf=-1e12)

    sb = np.sort(tr_b)
    sp = np.sort(tr_p)
    qb = np.searchsorted(sb, ev_b, side="right") / float(len(sb) + 1)
    qp = np.searchsorted(sp, ev_p, side="right") / float(len(sp) + 1)

    return 0.5 * (qb + qp)
