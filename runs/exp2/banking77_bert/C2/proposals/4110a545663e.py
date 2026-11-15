import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs may be close to some class manifold yet far from the class selected by the classifier. Normalizing predicted-class Mahalanobis distance by that class's in-scope distance scale makes this disagreement comparable across classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    n_classes = train_dist.shape[1]
    scales = np.empty(n_classes, dtype=float)
    for c in range(n_classes):
        own = train_dist[tr.labels == c, c]
        if own.size:
            scales[c] = np.quantile(own, 0.90)
        else:
            scales[c] = np.quantile(train_dist[:, c], 0.90)
    scales = np.maximum(scales, 1e-8)

    eval_dist = g.class_mahalanobis_sq(ev.pooled)
    pred = np.argmax(ev.logits, axis=1)
    pred = np.minimum(pred, n_classes - 1)
    chosen = eval_dist[np.arange(eval_dist.shape[0]), pred]

    return np.log1p(chosen / scales[pred])
