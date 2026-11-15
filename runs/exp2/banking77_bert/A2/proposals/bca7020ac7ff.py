import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query assigned to an intent should lie near that predicted intent's feature distribution, "
    "whereas OOS queries can be confidently assigned yet remain atypical for the assigned class."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6
    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)

    for c in range(k):
        xc = x[y == c]
        if xc.shape[0] == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0)
            shrink = xc.shape[0] / (xc.shape[0] + 20.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var + 1e-6

    train_diff = x - means[y]
    train_dist = np.sum(train_diff * train_diff / variances[y], axis=1)
    class_center = np.zeros(k, dtype=float)
    class_scale = np.ones(k, dtype=float)

    for c in range(k):
        dc = train_dist[y == c]
        if dc.shape[0] > 0:
            class_center[c] = np.median(dc)
            mad = np.median(np.abs(dc - class_center[c]))
            class_scale[c] = max(1.4826 * mad, 1e-4)

    pred = np.argmax(ev.logits, axis=1)
    diff = z - means[pred]
    dist = np.sum(diff * diff / variances[pred], axis=1)
    return (dist - class_center[pred]) / class_scale[pred]
