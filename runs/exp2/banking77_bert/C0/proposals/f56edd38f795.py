import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal covariances retain per-intent feature scale information that tied-covariance GDA discards. Shrinkage toward the global variance stabilizes small classes while making atypical queries have low likelihood under every intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    global_var = np.var(x, axis=0)
    scale = float(np.median(global_var[global_var > 0])) if np.any(global_var > 0) else 1.0
    floor = max(scale * 1e-6, 1e-12)
    global_var = np.maximum(global_var, floor)

    classes = np.unique(y)
    means = []
    precisions = []
    logdets = []
    counts = []

    shrink_strength = 20.0
    for c in classes:
        xc = x[y == c]
        n = xc.shape[0]
        mu = np.mean(xc, axis=0)
        if n > 1:
            var = np.mean((xc - mu) ** 2, axis=0)
        else:
            var = global_var.copy()

        var = (n * var + shrink_strength * global_var) / (n + shrink_strength)
        var = np.maximum(var, floor)

        means.append(mu)
        precisions.append(1.0 / var)
        logdets.append(np.sum(np.log(var)))
        counts.append(n)

    means = np.asarray(means)
    precisions = np.asarray(precisions)
    logdets = np.asarray(logdets)
    priors = np.asarray(counts, dtype=float) / float(x.shape[0])

    # Quadratic distances expanded into matrix products, avoiding an
    # (eval_items, classes, dimensions) intermediate.
    z2 = z * z
    quad = z2 @ precisions.T
    quad -= 2.0 * z @ (means * precisions).T
    quad += np.sum(means * means * precisions, axis=1)

    nll = 0.5 * (quad + logdets[None, :]) - np.log(priors[None, :])
    return np.min(nll, axis=1)
