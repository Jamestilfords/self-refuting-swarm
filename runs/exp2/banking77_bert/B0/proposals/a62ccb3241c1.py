import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Each intent can occupy a differently scaled region of bottleneck feature space, so class-specific diagonal variances retain deviations that a tied covariance averages away. Shrinking each class variance toward the global variance stabilizes estimates for intents with limited training examples."
SOURCE = "https://cdn.amazon.science/64/48/4d1e599c4472859e27a9f7647786/few-shot-out-of-domain-intent-detection-with-covariance-corrected-mahalanobis-distance.pdf"
PARENTS = []
MEMORY_USED = ["m_0002"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels)
    q = np.asarray(ev.feats, dtype=np.float64)

    classes = np.unique(y)
    d = x.shape[1]
    global_var = np.var(x, axis=0) + 1e-6

    means = np.empty((classes.size, d), dtype=np.float64)
    inv_vars = np.empty((classes.size, d), dtype=np.float64)

    for i, c in enumerate(classes):
        xc = x[y == c]
        n = xc.shape[0]
        mu = np.mean(xc, axis=0)
        if n > 1:
            class_var = np.var(xc, axis=0)
        else:
            class_var = global_var
        shrink = n / (n + 20.0)
        var = shrink * class_var + (1.0 - shrink) * global_var
        means[i] = mu
        inv_vars[i] = 1.0 / np.maximum(var, 1e-6)

    q_sq = q * q
    linear = means * inv_vars
    constants = np.sum(means * means * inv_vars, axis=1)

    distances = q_sq.dot(inv_vars.T)
    distances -= 2.0 * q.dot(linear.T)
    distances += constants[None, :]
    return np.min(distances, axis=1)
