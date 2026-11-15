import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries may receive a confident predicted label while lying far from that predicted class's feature distribution. Calibrating predicted-class Mahalanobis distance by each class's in-scope training distribution makes distances comparable across classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = train_dist.shape[1]
    center = np.zeros(n_classes, dtype=float)
    scale = np.ones(n_classes, dtype=float)

    for c in range(n_classes):
        own = train_dist[labels == c, c]
        if own.size:
            q25, q50, q75 = np.quantile(own, [0.25, 0.50, 0.75])
            center[c] = q50
            scale[c] = max(q75 - q25, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, n_classes - 1)
    rows = np.arange(eval_dist.shape[0])
    return (eval_dist[rows, pred] - center[pred]) / scale[pred]
