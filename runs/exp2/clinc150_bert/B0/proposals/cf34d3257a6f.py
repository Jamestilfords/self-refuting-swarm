import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query should remain close to the prototype of the class the classifier itself predicts, "
    "whereas OOS queries can receive a confident predicted intent while lying atypically far from that intent's feature distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_pred = np.argmax(tr.logits, axis=1)
    train_distances = g.class_mahalanobis_sq(tr.pooled)
    train_pred_dist = train_distances[np.arange(train_distances.shape[0]), train_pred]

    n_classes = train_distances.shape[1]
    global_center = np.median(train_pred_dist)
    global_scale = np.quantile(train_pred_dist, 0.75) - np.quantile(train_pred_dist, 0.25)
    global_scale = max(float(global_scale), 1e-6)

    centers = np.full(n_classes, global_center, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_pred_dist[train_pred == c]
        if values.size >= 8:
            centers[c] = np.median(values)
            spread = np.quantile(values, 0.75) - np.quantile(values, 0.25)
            scales[c] = max(float(spread), 1e-6)

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_distances = g.class_mahalanobis_sq(ev.pooled)
    predicted_distance = eval_distances[np.arange(eval_distances.shape[0]), eval_pred]

    return (predicted_distance - centers[eval_pred]) / scales[eval_pred]
