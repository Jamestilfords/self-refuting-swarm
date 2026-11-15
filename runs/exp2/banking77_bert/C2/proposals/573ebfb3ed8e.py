import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries can deviate along low-variance within-class directions that are suppressed by standard full-space distances. Trailing-eigenspace residuals emphasize these constrained semantic directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    residual = ctx.h.partial(tr.pooled, tr.labels, top_k=128).bottom(ev.pooled)
    return np.nan_to_num(residual, nan=0.0, posinf=1e12, neginf=-1e12)
