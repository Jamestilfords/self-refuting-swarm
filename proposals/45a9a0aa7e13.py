import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query can be close to some intent while being poorly supported by the intent selected by "
    "the classifier. Class-conditional calibration of distance to the predicted class mean exposes "
    "this classifier-feature disagreement."
)
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = []
MEMORY_USED = ["m_0001", "m_0016"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = train_dist.shape[1]
    pred_eval = np.argmax(ev.logits, axis=1)

    center = np.zeros(n_classes, dtype=float)
    scale = np.ones(n_classes, dtype=float)

    global_values = train_dist[np.arange(labels.size), labels]
    global_center = np.median(global_values)
    global_scale = np.quantile(global_values, 0.90) - global_center
    global_scale = max(float(global_scale), 1e-6)

    for c in range(n_classes):
        values = train_dist[labels == c, c]
        if values.size >= 4:
            med = np.median(values)
            tail = np.quantile(values, 0.90)
            center[c] = med
            scale[c] = max(float(tail - med), 1e-6)
        else:
            center[c] = global_center
            scale[c] = global_scale

    chosen = eval_dist[np.arange(pred_eval.size), pred_eval]
    scores = (chosen - center[pred_eval]) / scale[pred_eval]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
