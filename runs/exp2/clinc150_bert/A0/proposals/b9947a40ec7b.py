import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query's classifier-predicted intent should also be geometrically close to its "
    "nearest Gaussian intent region. Positive disagreement between the predicted-class distance "
    "and the nearest-class distance flags queries whose logits and representation support conflict."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_mismatch = (
        train_d[np.arange(train_d.shape[0]), train_pred] - np.min(train_d, axis=1)
    )
    eval_mismatch = (
        eval_d[np.arange(eval_d.shape[0]), eval_pred] - np.min(eval_d, axis=1)
    )

    center = np.median(train_mismatch)
    scale = np.median(np.abs(train_mismatch - center))
    scale = max(float(scale), 1e-6)

    return np.log1p(np.maximum(eval_mismatch, 0.0) / scale)
