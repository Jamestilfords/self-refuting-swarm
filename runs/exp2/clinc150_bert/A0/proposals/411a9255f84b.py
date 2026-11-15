import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions encode low-variance constraints shared by in-scope intent representations. OOS queries that violate these constrained pooled-feature directions receive larger residual scores."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    scores = np.asarray(model.bottom(ev.pooled), dtype=float).reshape(-1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
