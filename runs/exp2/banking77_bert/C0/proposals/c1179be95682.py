import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query whose classifier prediction disagrees with its representation is suspicious: it may be assigned a class confidently while lying unusually far from that predicted class's feature distribution. Calibrating the predicted-class Mahalanobis distance separately for each predicted class accounts for differing within-class spreads."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    n_classes = train_dist.shape[1]
    train_pred = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    eval_pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    train_selected = np.log1p(train_dist[np.arange(train_dist.shape[0]), train_pred])
    eval_selected = np.log1p(eval_dist[np.arange(eval_dist.shape[0]), eval_pred])

    global_med = np.median(train_selected)
    global_scale = np.median(np.abs(train_selected - global_med)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)
    for c in range(n_classes):
        values = train_selected[train_pred == c]
        if values.size >= 5:
            med = np.median(values)
            scale = np.median(np.abs(values - med)) * 1.4826
            medians[c] = med
            scales[c] = max(float(scale), 1e-6)

    scores = (eval_selected - medians[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
