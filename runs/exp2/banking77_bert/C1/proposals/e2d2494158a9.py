import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An OOS query can receive a confident predicted label while remaining unusually far from that predicted class's feature distribution. Classwise calibration of predicted-class Mahalanobis distance accounts for differing within-class feature scales."
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

    train_pred_dist = train_dist[np.arange(train_dist.shape[0]), train_pred]
    eval_pred_dist = eval_dist[np.arange(eval_dist.shape[0]), eval_pred]

    global_mean = float(np.mean(train_pred_dist))
    global_std = float(np.std(train_pred_dist))
    global_std = max(global_std, 1e-6)

    means = np.full(train_dist.shape[1], global_mean, dtype=float)
    stds = np.full(train_dist.shape[1], global_std, dtype=float)

    for c in range(train_dist.shape[1]):
        values = train_pred_dist[train_pred == c]
        if values.size >= 2:
            means[c] = np.mean(values)
            stds[c] = max(float(np.std(values)), 1e-6)

    scores = (eval_pred_dist - means[eval_pred]) / stds[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
