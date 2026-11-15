import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An OOS query may be confidently assigned to a class by the classifier while still lying unusually far from that predicted class's feature distribution. Class-specific distance scaling accounts for differing within-class feature dispersion."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    n_classes = train_dist.shape[1]
    train_labels = np.clip(np.asarray(tr.labels, dtype=int), 0, n_classes - 1)
    train_own = train_dist[np.arange(train_dist.shape[0]), train_labels]

    scales = np.empty(n_classes, dtype=float)
    global_scale = np.median(train_own[np.isfinite(train_own)])
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    for c in range(n_classes):
        values = train_own[train_labels == c]
        values = values[np.isfinite(values)]
        scale = np.median(values) if values.size else global_scale
        scales[c] = scale if np.isfinite(scale) and scale > 1e-12 else global_scale

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, n_classes - 1)
    own_dist = eval_dist[np.arange(eval_dist.shape[0]), predicted]

    scores = np.log1p(np.maximum(own_dist, 0.0) / scales[predicted])
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
