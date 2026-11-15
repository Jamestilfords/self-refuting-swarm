import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "In-scope queries should remain supported across both very local and broader neighborhoods, while OOS queries often become increasingly isolated as neighborhood scale grows. Combining train-calibrated kNN distances at multiple scales reduces sensitivity to accidental single-neighbor matches."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = ctx.h.l2(tr.feats)
    eval_x = ctx.h.l2(ev.feats)

    ks = (5, 15, 40)
    parts = []
    for k in ks:
        train_d = ctx.h.knn(train_x, train_x, k=k + 1, l2=False)
        scale = np.median(train_d[np.isfinite(train_d)])
        scale = max(float(scale), 1e-8)

        eval_d = ctx.h.knn(train_x, eval_x, k=k, l2=False)
        parts.append(np.log1p(np.maximum(eval_d, 0.0) / scale))

    return np.mean(np.stack(parts, axis=1), axis=1)
