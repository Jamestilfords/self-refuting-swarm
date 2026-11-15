import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions capture low-variance constraints of known intents. "
    "Queries that violate these constrained directions should receive larger OOS scores."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_score = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_score = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    center = np.median(train_score)
    scale = np.median(np.abs(train_score - center)) + 1e-8
    out = (eval_score - center) / scale
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
