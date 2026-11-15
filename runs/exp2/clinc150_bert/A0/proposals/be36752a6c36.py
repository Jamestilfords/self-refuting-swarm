import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions encode constraints that supported intents tend to satisfy. "
    "The ratio of trailing to leading residual energy highlights violations while reducing sensitivity to overall representation magnitude."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0113"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    p = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    top = np.asarray(p.top(ev.pooled), dtype=float)
    bottom = np.asarray(p.bottom(ev.pooled), dtype=float)

    if top.ndim > 1:
        top = np.sum(top * top, axis=tuple(range(1, top.ndim)))
    if bottom.ndim > 1:
        bottom = np.sum(bottom * bottom, axis=tuple(range(1, bottom.ndim)))

    return bottom / (np.abs(top) + np.abs(bottom) + 1e-12)
