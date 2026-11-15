import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances retain intent-dependent feature scales that a tied covariance discards, while shrinkage toward global variance stabilizes small classes. OOS queries should have high minimum shrunk diagonal Gaussian surprise across all known intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    classes = np.unique(y)
    d = x.shape[1]
    global_var = np.var(x, axis=0) + 1e-6

    means = []
    variances = []
    for c in classes:
        xc = x[y == c]
        n_c = xc.shape[0]
        mu = np.mean(xc, axis=0)
        if n_c > 1:
            local_var = np.var(xc, axis=0, ddof=1)
        else:
            local_var = global_var.copy()
        shrink = d / (d + n_c)
        var = (1.0 - shrink) * local_var + shrink * global_var
        means.append(mu)
        variances.append(np.maximum(var, 1e-6))

    means = np.asarray(means)
    inv_var = 1.0 / np.asarray(variances)
    linear = means * inv_var
    offset = np.sum(means * linear, axis=1)
    logdet = np.sum(np.log(1.0 / inv_var), axis=1)

    out = np.empty(z.shape[0], dtype=np.float64)
    chunk = 1024
    for start in range(0, z.shape[0], chunk):
        zz = z[start:start + chunk]
        quad = (zz * zz) @ inv_var.T - 2.0 * (zz @ linear.T) + offset
        nll = 0.5 * (quad + logdet)
        out[start:start + zz.shape[0]] = np.min(nll, axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
