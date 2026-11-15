import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions capture deviations that are suppressed by dominant semantic variation. OOS queries should exhibit larger trailing-subspace residuals than in-scope examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    values = np.asarray(model.bottom(ev.pooled), dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
