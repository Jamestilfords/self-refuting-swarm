import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known intents tend to occupy regions with a clear Mahalanobis gap between the nearest and "
    "second-nearest intent prototypes. A small prototype-margin gap, calibrated for the classifier's "
    "selected intent, flags geometrically ambiguous queries as out-of-scope."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0223", "m_0231"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = np.asarray(tr.labels, dtype=np.int64)
    train_pred = np.argmax(np.asarray(tr.logits), axis=1)
    eval_pred = np.argmax(np.asarray(ev.logits), axis=1)

    g = ctx.h.gda(tr.pooled, labels, cov="tied", whiten=True)
    train_d = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=np.float64)
    eval_d = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=np.float64)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]
    train_gap = np.log1p(np.maximum(train_two[:, 1] - train_two[:, 0], 0.0))
    eval_gap = np.log1p(np.maximum(eval_two[:, 1] - eval_two[:, 0], 0.0))

    n_classes = train_d.shape[1]
    global_med = float(np.median(train_gap))
    global_mad = float(np.median(np.abs(train_gap - global_med))) * 1.4826
    global_mad = max(global_mad, 1e-6)

    med = np.full(n_classes, global_med, dtype=np.float64)
    scale = np.full(n_classes, global_mad, dtype=np.float64)

    for c in range(n_classes):
        values = train_gap[train_pred == c]
        if values.size >= 8:
            center = float(np.median(values))
            spread = float(np.median(np.abs(values - center))) * 1.4826
            med[c] = center
            scale[c] = max(spread, 1e-6)

    valid_pred = np.clip(eval_pred, 0, n_classes - 1)
    scores = (med[valid_pred] - eval_gap) / scale[valid_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
