import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Out-of-scope inputs can exhibit anomalous residual variation in low-variance within-class directions that is suppressed by conventional leading-subspace representations. A trailing partial Mahalanobis score emphasizes this signal while fitting the subspace only on in-scope training data."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    trailing = np.asarray(partial.bottom(ev.pooled), dtype=float)
    return np.nan_to_num(trailing, nan=0.0, posinf=1e12, neginf=-1e12)
