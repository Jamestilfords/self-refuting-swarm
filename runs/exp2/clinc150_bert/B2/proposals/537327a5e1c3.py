import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries can deviate along low-variance within-class directions that are suppressed by full-space distance measures. A trailing-eigenspace score emphasizes these tightly constrained in-scope directions using only training representations."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    train_score = np.asarray(model.bottom(tr.pooled), dtype=float)
    eval_score = np.asarray(model.bottom(ev.pooled), dtype=float)

    finite_train = train_score[np.isfinite(train_score)]
    if finite_train.size == 0:
        finite_train = np.array([0.0, 1.0])

    center = np.median(finite_train)
    scale = np.percentile(finite_train, 75) - np.percentile(finite_train, 25)
    scale = max(float(scale), 1e-8)

    out = (eval_score - center) / scale
    out = np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)
    return out
