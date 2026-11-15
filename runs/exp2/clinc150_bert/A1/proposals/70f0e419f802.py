import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query can be close to some known intent while still being incompatible with the intent selected "
    "by the classifier. Predicted-class Mahalanobis surprise flags disagreement between classifier routing "
    "and class-conditional representation support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    n_classes = train_dist.shape[1]
    train_pred = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    eval_pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    train_selected = train_dist[np.arange(train_dist.shape[0]), train_pred]
    eval_selected = eval_dist[np.arange(eval_dist.shape[0]), eval_pred]

    means = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)
    for c in range(n_classes):
        values = train_selected[train_pred == c]
        if values.size:
            means[c] = np.mean(values)
            scales[c] = max(np.std(values), 1e-6)
        else:
            means[c] = np.mean(train_selected)
            scales[c] = max(np.std(train_selected), 1e-6)

    return (eval_selected - means[eval_pred]) / scales[eval_pred]
