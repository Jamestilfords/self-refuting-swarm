import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class covariance directions encode constraints shared by in-scope intents. "
    "OOS queries should violate these trailing-direction constraints more strongly than typical training examples."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    tail = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    raw = np.asarray(tail.bottom(ev.pooled), dtype=float)
    train_raw = np.asarray(tail.bottom(tr.pooled), dtype=float)

    center = np.median(train_raw)
    scale = np.median(np.abs(train_raw - center)) + 1e-8
    return np.log1p(np.maximum(raw - center, 0.0) / scale)
