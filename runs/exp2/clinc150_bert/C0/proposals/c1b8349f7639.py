import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs may be assigned a confident predicted class while remaining unusually far from that class's training distribution. Class-wise robust calibration makes distances comparable across intent classes with different within-class spread."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_all = g.class_mahalanobis_sq(tr.pooled)
    train_self = train_all[np.arange(tr.labels.shape[0]), tr.labels]

    n_classes = train_all.shape[1]
    centers = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)

    for c in range(n_classes):
        values = train_self[tr.labels == c]
        if values.size:
            center = np.median(values)
            mad = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(1.4826 * mad, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_all = g.class_mahalanobis_sq(ev.pooled)
    pred_dist = eval_all[np.arange(pred.shape[0]), pred]

    standardized = (pred_dist - centers[pred]) / scales[pred]
    return np.clip(standardized, -50.0, 50.0)
