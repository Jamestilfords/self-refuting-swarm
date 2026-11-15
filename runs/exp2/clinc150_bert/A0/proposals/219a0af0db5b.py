import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query confidently assigned to a class can still be OOS if its representation is atypical for that predicted class. Class-wise robust calibration makes predicted-class Mahalanobis distances comparable across intents with different spread."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_all = g.class_mahalanobis_sq(tr.pooled)
    eval_all = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_d = train_all[np.arange(train_all.shape[0]), train_pred]
    eval_d = eval_all[np.arange(eval_all.shape[0]), eval_pred]

    finite_train = train_d[np.isfinite(train_d)]
    global_med = np.median(finite_train)
    global_mad = np.median(np.abs(finite_train - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    n_classes = train_all.shape[1]
    med = np.full(n_classes, global_med, dtype=float)
    scale = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_d[train_pred == c]
        values = values[np.isfinite(values)]
        if values.size >= 8:
            center = np.median(values)
            spread = np.median(np.abs(values - center))
            med[c] = center
            scale[c] = max(1.4826 * spread, 1e-6)

    z = (eval_d - med[eval_pred]) / scale[eval_pred]
    return np.clip(z, -1e6, 1e6)
