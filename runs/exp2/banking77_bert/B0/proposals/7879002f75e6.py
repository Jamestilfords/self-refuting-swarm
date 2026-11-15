import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries can deviate most strongly along low-variance within-class directions that are suppressed by dominant semantic variation. Trailing-eigenspace residuals therefore expose deviations that a full-space distance can dilute."
SOURCE = "https://cdn.amazon.science/64/48/4d1e599c4472859e27a9f7647786/few-shot-out-of-domain-intent-detection-with-covariance-corrected-mahalanobis-distance.pdf"
PARENTS = []
MEMORY_USED = ["m_0002"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    residual = ctx.h.partial(tr.pooled, tr.labels, top_k=128).bottom(ev.pooled)
    return np.nan_to_num(residual, nan=0.0, posinf=1e12, neginf=-1e12)
