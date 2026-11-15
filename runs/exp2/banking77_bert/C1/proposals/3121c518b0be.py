import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Distance measured only in trailing within-class covariance directions emphasizes subtle deviations that are suppressed by high-variance semantic directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    model = ctx.h.partial(tr.feats, tr.labels, top_k=128)
    values = model.bottom(ev.feats)
    return np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
