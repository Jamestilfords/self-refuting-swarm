import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query assigned to an intent should have a typical class-conditional diagonal Mahalanobis "
    "residual under that intent's shrunk variance profile. Calibrating residuals by the classifier's "
    "predicted class makes the score sensitive to unsupported within-intent feature deviations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0102", "m_0113"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    y_train = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    global_var = np.var(x_train, axis=0) + 1e-6
    means = np.zeros((k, x_train.shape[1]), dtype=float)
    variances = np.zeros_like(means)

    for c in range(k):
        xc = x_train[y_train == c]
        if xc.shape[0] == 0:
            means[c] = np.mean(x_train, axis=0)
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0)
            shrink = 20.0 / (20.0 + xc.shape[0])
            variances[c] = (1.0 - shrink) * class_var + shrink * global_var + 1e-6

    train_pred = np.argmax(tr.logits, axis=1)
    train_dist = np.empty(x_train.shape[0], dtype=float)
    for c in range(k):
        mask = train_pred == c
        if np.any(mask):
            delta = x_train[mask] - means[c]
            train_dist[mask] = np.sum(delta * delta / variances[c], axis=1)

    cal_mean = np.zeros(k, dtype=float)
    cal_scale = np.ones(k, dtype=float)
    for c in range(k):
        vals = train_dist[train_pred == c]
        if vals.size:
            cal_mean[c] = np.mean(vals)
            cal_scale[c] = np.std(vals) + 1e-6
        else:
            cal_mean[c] = np.mean(train_dist)
            cal_scale[c] = np.std(train_dist) + 1e-6

    pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(x_eval.shape[0], dtype=float)
    for c in range(k):
        mask = pred == c
        if np.any(mask):
            delta = x_eval[mask] - means[c]
            dist = np.sum(delta * delta / variances[c], axis=1)
            scores[mask] = (dist - cal_mean[c]) / cal_scale[c]

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
