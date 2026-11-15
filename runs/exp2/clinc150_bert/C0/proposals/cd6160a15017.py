import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances retain directions that are stable for one intent but variable for another, while shrinkage toward the global variance prevents unreliable covariance estimates for small classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes = np.unique(y)
    d = x.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    means = np.empty((len(classes), d), dtype=float)
    inv_vars = np.empty((len(classes), d), dtype=float)
    logdets = np.empty(len(classes), dtype=float)

    for j, c in enumerate(classes):
        xc = x[y == c]
        n = xc.shape[0]
        mu = np.mean(xc, axis=0)
        if n > 1:
            var = np.mean((xc - mu) ** 2, axis=0)
        else:
            var = global_var.copy()

        shrink = n / (n + 32.0)
        var = shrink * var + (1.0 - shrink) * global_var
        var = np.maximum(var, floor)

        means[j] = mu
        inv_vars[j] = 1.0 / var
        logdets[j] = np.sum(np.log(var))

    z2 = z * z
    quad = z2 @ inv_vars.T
    quad -= 2.0 * z @ (means * inv_vars).T
    quad += np.sum(means * means * inv_vars, axis=1)

    nll = 0.5 * (quad + logdets[None, :])
    return np.min(nll, axis=1)
