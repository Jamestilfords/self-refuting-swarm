import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances preserve intent-dependent feature spread that a tied covariance loses. Shrinkage toward the global diagonal covariance keeps estimates stable for small classes while assigning high score to points unlikely under every intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    classes, inverse = np.unique(y, return_inverse=True)
    n_total, dim = x.shape
    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    best = np.full(z.shape[0], np.inf, dtype=float)
    for j in range(classes.size):
        members = x[inverse == j]
        count = members.shape[0]
        mean = np.mean(members, axis=0)
        class_var = np.mean((members - mean) ** 2, axis=0)

        shrink = dim / float(dim + count)
        var = (1.0 - shrink) * class_var + shrink * global_var
        var = np.maximum(var, floor)

        diff = z - mean
        quadratic = np.sum((diff * diff) / var, axis=1)
        logdet = np.sum(np.log(var))
        prior_penalty = -np.log(count / float(n_total))
        nll = 0.5 * (quadratic + logdet) + prior_penalty
        best = np.minimum(best, nll)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=-1e12)
