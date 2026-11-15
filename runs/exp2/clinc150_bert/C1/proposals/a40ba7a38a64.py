import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs can deviate along low-variance within-class covariance directions that are suppressed by standard leading-subspace representations. The trailing-subspace residual emphasizes these atypical deviations."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_scores = partial.bottom(tr.pooled)
    eval_scores = partial.bottom(ev.pooled)

    center = np.median(train_scores)
    scale = np.median(np.abs(train_scores - center))
    scale = max(float(scale), 1e-8)

    scores = (eval_scores - center) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
