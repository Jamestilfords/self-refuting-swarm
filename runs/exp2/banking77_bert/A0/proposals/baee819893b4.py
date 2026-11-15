import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query should be close to the representation distribution of the class predicted by the classifier, not merely close to some in-scope class. Per-class train-calibrated predicted-class Mahalanobis distances expose disagreement between classifier assignment and class-conditional feature support."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    n_classes = train_d.shape[1]
    labels = np.clip(tr.labels.astype(int), 0, n_classes - 1)
    own_d = train_d[np.arange(labels.shape[0]), labels]

    centers = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)
    global_scale = np.median(np.abs(own_d - np.median(own_d))) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    for c in range(n_classes):
        vals = own_d[labels == c]
        if vals.size:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) * 1.4826
            centers[c] = med
            scales[c] = max(mad, 0.1 * global_scale, 1e-6)
        else:
            centers[c] = float(np.median(own_d))
            scales[c] = global_scale

    eval_d = g.class_mahalanobis_sq(ev.pooled)
    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, n_classes - 1)
    selected = eval_d[np.arange(predicted.shape[0]), predicted]

    scores = (selected - centers[predicted]) / scales[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
