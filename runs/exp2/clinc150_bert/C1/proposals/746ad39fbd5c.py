import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries can be close to some class while being inconsistent with the classifier's predicted class. "
    "Class-specific calibration makes predicted-class Mahalanobis distance comparable across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    n_classes = train_d.shape[1]
    all_d = train_d[np.arange(train_d.shape[0]), np.clip(train_pred, 0, n_classes - 1)]
    global_med = np.median(all_d)
    global_hi = np.quantile(all_d, 0.90)
    global_scale = max(global_hi - global_med, 1e-6)

    med = np.full(n_classes, global_med, dtype=float)
    scale = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = all_d[train_pred == c]
        if values.size >= 8:
            center = np.median(values)
            upper = np.quantile(values, 0.90)
            med[c] = center
            scale[c] = max(upper - center, 1e-6)

    pred = np.clip(eval_pred, 0, n_classes - 1)
    chosen = eval_d[np.arange(eval_d.shape[0]), pred]
    return (chosen - med[pred]) / scale[pred]
