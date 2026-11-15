import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-specific diagonal variances capture intent-dependent feature dispersion that a tied covariance misses. "
    "Shrinkage toward the global variance keeps rare-intent covariance estimates stable while exposing atypical queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    classes = np.unique(y)
    n, d = x.shape
    global_var = np.var(x, axis=0) + 1e-6
    shrink_strength = 32.0

    best = np.full(z.shape[0], np.inf, dtype=float)
    z_sq = z * z

    for c in classes:
        xc = x[y == c]
        nc = xc.shape[0]
        mu = np.mean(xc, axis=0)

        if nc > 1:
            local_var = np.var(xc, axis=0, ddof=1)
        else:
            local_var = global_var

        weight = nc / (nc + shrink_strength)
        var = weight * local_var + (1.0 - weight) * global_var
        var = np.maximum(var, 1e-6)
        inv_var = 1.0 / var

        quad = (
            z_sq @ inv_var
            - 2.0 * (z @ (mu * inv_var))
            + np.sum(mu * mu * inv_var)
        )
        nll = quad + np.sum(np.log(var))
        best = np.minimum(best, nll)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=-1e12)
