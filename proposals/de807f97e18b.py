import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A geometric average of several normalized kNN radii measures support across local, "
    "mesoscopic, and broad neighborhoods, so OOS queries isolated at any stable scale receive a high score."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0030", "m_0016"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)

    radii = []
    for k in (5, 17, 53):
        kk = min(k, max(1, len(train_x) - 1))
        train_r = ctx.h.knn(train_x, train_x, k=kk + 1, l2=False)
        scale = np.median(train_r[np.isfinite(train_r)])
        scale = max(float(scale), 1e-12)
        eval_r = ctx.h.knn(train_x, eval_x, k=kk, l2=False)
        radii.append(np.log1p(np.maximum(eval_r, 0.0) / scale))

    return np.mean(np.stack(radii, axis=1), axis=1)
