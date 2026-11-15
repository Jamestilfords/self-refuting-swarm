import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query assigned to a class but lying far from that class's Gaussian cluster is likely OOS, "
    "even when it is accidentally close to another intent cluster. Per-class training-distance calibration "
    "makes predicted-class deviations comparable across intents with different intrinsic spread."
)
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = []
MEMORY_USED = ["m_0001", "m_0006"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_all = g.class_mahalanobis_sq(tr.pooled)
    eval_all = g.class_mahalanobis_sq(ev.pooled)

    labels = np.asarray(tr.labels, dtype=int)
    own_train_dist = train_all[np.arange(labels.shape[0]), labels]

    n_classes = train_all.shape[1]
    med = np.zeros(n_classes, dtype=float)
    scale = np.ones(n_classes, dtype=float)

    global_med = float(np.median(own_train_dist))
    global_iqr = float(
        np.percentile(own_train_dist, 75) - np.percentile(own_train_dist, 25)
    )
    global_iqr = max(global_iqr, 1e-6)

    for c in range(n_classes):
        values = own_train_dist[labels == c]
        if values.size >= 4:
            med[c] = np.median(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            scale[c] = max(float(iqr), 1e-6)
        else:
            med[c] = global_med
            scale[c] = global_iqr

    predicted = np.argmax(ev.logits, axis=1)
    predicted_dist = eval_all[np.arange(predicted.shape[0]), predicted]
    scores = (predicted_dist - med[predicted]) / scale[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
