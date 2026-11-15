import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs can be deceptively close to some class while being far from the class selected by the classifier. Normalizing predicted-class Mahalanobis distance by that class's in-scope distance scale makes distances comparable across intents."
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
            scales[c] = np.median(own)
        else:
            scales[c] = np.median(train_dist[:, c])

    scales = np.maximum(scales, 1e-6)
    pred = np.argmax(ev.logits, axis=1)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)
    selected = eval_dist[np.arange(eval_dist.shape[0]), pred]

    return np.log1p(np.maximum(selected, 0.0) / scales[pred])
