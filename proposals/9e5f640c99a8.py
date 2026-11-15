import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query should be geometrically typical of the intent predicted by the classifier, not merely close to some intent prototype. "
    "Class-specific robust normalization makes unusually large predicted-class Mahalanobis radii comparable across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.feats)
    eval_dist = g.class_mahalanobis_sq(ev.feats)

    labels = tr.labels.astype(int)
    own_train = train_dist[np.arange(labels.shape[0]), labels]
    pred = np.argmax(ev.logits, axis=1)
    own_eval = eval_dist[np.arange(pred.shape[0]), pred]

    n_classes = train_dist.shape[1]
    centers = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)

    for c in range(n_classes):
        values = np.log1p(own_train[labels == c])
        if values.size:
            center = np.median(values)
            mad = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(1.4826 * mad, 1e-6)

    standardized = (np.log1p(own_eval) - centers[pred]) / scales[pred]
    return np.maximum(standardized, -50.0)
