import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Intent-specific diagonal variances identify coordinates that are tightly constrained for each intent. "
    "A smooth count of class-specific coordinate violations detects distributed unsupported deviations that a single radius can dilute."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094", "m_0195"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0) + 1e-6
    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        xc = x[y == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = xc.mean(axis=0)
            class_var = xc.var(axis=0)
            shrink = n / (n + 20.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var + 1e-6

    train_pred = np.argmax(tr.logits, axis=1)
    train_diff = (x - means[train_pred]) / np.sqrt(variances[train_pred])
    train_abs = np.abs(train_diff)

    thresholds = np.empty((k, d), dtype=float)
    for c in range(k):
        vals = train_abs[train_pred == c]
        if vals.shape[0] < 8:
            vals = train_abs[y == c]
        if vals.shape[0] < 2:
            thresholds[c] = 2.0
        else:
            thresholds[c] = np.quantile(vals, 0.90, axis=0)

    pred = np.argmax(ev.logits, axis=1)
    standardized = np.abs((z - means[pred]) / np.sqrt(variances[pred]))
    excess = np.maximum(standardized - thresholds[pred], 0.0)
    return np.mean(np.log1p(excess), axis=1)
