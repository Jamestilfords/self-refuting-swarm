import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific shrunk diagonal variances preserve intent-dependent feature scales that a tied covariance loses. Distance to the classifier-predicted intent then flags queries that are confident but geometrically inconsistent with that intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    dim = x.shape[1]

    global_var = np.var(x, axis=0) + 1e-6
    means = np.zeros((n_classes, dim), dtype=float)
    variances = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(y, minlength=n_classes).astype(float)

    for c in range(n_classes):
        xc = x[y == c]
        if xc.shape[0] == 0:
            means[c] = np.mean(x, axis=0)
            variances[c] = global_var
            continue
        means[c] = np.mean(xc, axis=0)
        class_var = np.var(xc, axis=0) if xc.shape[0] > 1 else global_var
        shrink = xc.shape[0] / (xc.shape[0] + 32.0)
        variances[c] = shrink * class_var + (1.0 - shrink) * global_var + 1e-6

    out = np.empty(z.shape[0], dtype=float)
    for c in range(n_classes):
        mask = pred == c
        if np.any(mask):
            delta = z[mask] - means[c]
            out[mask] = np.sum((delta * delta) / variances[c], axis=1) / dim

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
