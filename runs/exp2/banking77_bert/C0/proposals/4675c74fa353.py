import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query is suspicious when its classifier-predicted class is not geometrically supported by the encoder representation. Class-specific calibration prevents broad in-scope classes from being penalized merely for having naturally larger within-class distances."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = train_dist.shape[1]

    train_pred = np.clip(train_pred, 0, n_classes - 1)
    eval_pred = np.clip(eval_pred, 0, n_classes - 1)

    train_selected = train_dist[np.arange(train_dist.shape[0]), train_pred]
    eval_selected = eval_dist[np.arange(eval_dist.shape[0]), eval_pred]

    global_mean = np.mean(train_selected)
    global_scale = np.std(train_selected) + 1e-6
    means = np.full(n_classes, global_mean, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        vals = train_selected[train_pred == c]
        if vals.size >= 5:
            means[c] = np.mean(vals)
            scales[c] = np.std(vals) + 1e-6

    return (eval_selected - means[eval_pred]) / scales[eval_pred]
