import numpy as np
from scipy.special import logsumexp

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two margin and the change in log-partition under temperature scaling describe confidence shape beyond raw energy or MSP. Conditioning their typical values on the predicted intent flags outputs whose relative evidence geometry is unusual for the class they claim."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0143"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    ztr = np.asarray(tr.logits, dtype=float)
    zev = np.asarray(ev.logits, dtype=float)
    k = ztr.shape[1]

    def shape_features(z):
        if z.shape[1] > 1:
            top2 = np.partition(z, -2, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
        else:
            margin = np.zeros(z.shape[0], dtype=float)
        e1 = logsumexp(z, axis=1)
        e2 = 2.0 * logsumexp(z / 2.0, axis=1)
        temp_shift = e2 - e1
        return np.column_stack((margin, temp_shift))

    xtr = shape_features(ztr)
    xev = shape_features(zev)
    ptr = np.argmax(ztr, axis=1)
    pev = np.argmax(zev, axis=1)

    global_mean = np.mean(xtr, axis=0)
    global_var = np.var(xtr, axis=0)
    floor = np.maximum(global_var * 1e-3, 1e-8)

    means = np.empty((k, xtr.shape[1]), dtype=float)
    variances = np.empty_like(means)
    for c in range(k):
        xc = xtr[ptr == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var + floor
        else:
            weight = n / (n + 8.0)
            means[c] = weight * np.mean(xc, axis=0) + (1.0 - weight) * global_mean
            local_var = np.var(xc, axis=0)
            variances[c] = weight * local_var + (1.0 - weight) * global_var + floor

    delta = xev - means[pev]
    out = np.sum((delta * delta) / variances[pev], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
