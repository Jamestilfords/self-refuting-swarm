import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope queries should be close to the mean of the class selected by the classifier, "
    "whereas OOS queries can be assigned a class with an atypically large class-conditional deviation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d_train = g.class_mahalanobis_sq(tr.pooled)
    d_eval = g.class_mahalanobis_sq(ev.pooled)

    n_classes = d_train.shape[1]
    pred_train = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    pred_eval = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    train_dist = d_train[np.arange(d_train.shape[0]), pred_train]
    eval_dist = d_eval[np.arange(d_eval.shape[0]), pred_eval]

    global_center = np.median(train_dist)
    global_scale = np.median(np.abs(train_dist - global_center)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    centers = np.full(n_classes, global_center, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_dist[pred_train == c]
        if values.size >= 8:
            center = np.median(values)
            scale = np.median(np.abs(values - center)) * 1.4826
            centers[c] = center
            scales[c] = max(float(scale), 1e-6)

    scores = (eval_dist - centers[pred_eval]) / scales[pred_eval]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
