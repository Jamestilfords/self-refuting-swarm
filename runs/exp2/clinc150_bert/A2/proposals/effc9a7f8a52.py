import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The fraction of within-class residual energy lying in trailing covariance directions "
    "emphasizes constraints that in-scope representations satisfy tightly while discounting broad feature magnitude."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    proj = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    train_top = np.asarray(proj.top(tr.pooled), dtype=float)
    train_bottom = np.asarray(proj.bottom(tr.pooled), dtype=float)
    eval_top = np.asarray(proj.top(ev.pooled), dtype=float)
    eval_bottom = np.asarray(proj.bottom(ev.pooled), dtype=float)

    scale = np.median(np.abs(train_top) + np.abs(train_bottom)) + 1e-12
    train_ratio = train_bottom / (np.abs(train_top) + train_bottom + scale)
    eval_ratio = eval_bottom / (np.abs(eval_top) + eval_bottom + scale)

    order = np.sort(train_ratio)
    score = np.searchsorted(order, eval_ratio, side="right") / float(len(order))
    return np.clip(score, 0.0, 1.0)
