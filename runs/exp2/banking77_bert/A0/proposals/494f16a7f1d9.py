import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions encode tight intent-specific constraints, while normalizing trailing residual energy by total partial-space residual reduces sensitivity to overall feature scale. OOS inputs should devote an unusually large fraction of their residual energy to these low-variance directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    tr_tail = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    tr_head = np.asarray(partial.top(tr.pooled), dtype=float).reshape(-1)
    ev_tail = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)
    ev_head = np.asarray(partial.top(ev.pooled), dtype=float).reshape(-1)

    scale = np.median(np.abs(tr_tail) + np.abs(tr_head)) + 1e-12
    train_ratio = tr_tail / (np.abs(tr_tail) + np.abs(tr_head) + scale * 1e-6)
    eval_ratio = ev_tail / (np.abs(ev_tail) + np.abs(ev_head) + scale * 1e-6)

    order = np.sort(train_ratio)
    return np.searchsorted(order, eval_ratio, side="right").astype(float) / (len(order) + 1.0)
