import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known-intent examples are typically distinctly closer to one class prototype than to the next-best prototype. "
    "OOS inputs often remain far from every class and have a weak nearest-versus-second-nearest distance separation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0093"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d_tr = g.class_mahalanobis_sq(tr.pooled)
    d_ev = g.class_mahalanobis_sq(ev.pooled)

    tr_two = np.partition(d_tr, 1, axis=1)[:, :2]
    ev_two = np.partition(d_ev, 1, axis=1)[:, :2]

    tr_near = tr_two[:, 0]
    ev_near = ev_two[:, 0]
    tr_gap = np.maximum(tr_two[:, 1] - tr_two[:, 0], 1e-8)
    ev_gap = np.maximum(ev_two[:, 1] - ev_two[:, 0], 1e-8)

    tr_score = tr_near / np.sqrt(tr_gap)
    ev_score = ev_near / np.sqrt(ev_gap)

    ref = np.sort(tr_score[np.isfinite(tr_score)])
    if ref.size == 0:
        return np.nan_to_num(ev_score, nan=0.0, posinf=1e12, neginf=-1e12)

    ranks = np.searchsorted(ref, np.nan_to_num(
        ev_score, nan=np.inf, posinf=np.inf, neginf=-np.inf
    ), side="right")
    return ranks.astype(float) / float(ref.size)
