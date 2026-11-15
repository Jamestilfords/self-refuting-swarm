import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions capture constrained in-scope variation, so "
    "OOS utterances should have unusually large residuals along these trailing directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    scores = np.asarray(model.bottom(ev.pooled), dtype=float)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
