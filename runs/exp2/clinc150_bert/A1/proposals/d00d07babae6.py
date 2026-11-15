import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A heteroscedastic diagonal Mahalanobis distance lets each intent retain its own "
    "feature-scale profile while shrinkage prevents small classes from producing unstable variances."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    classes = np.unique(labels)
    global_mean = x_train.mean(axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    best = np.full(x_eval.shape[0], np.inf, dtype=np.float64)
    shrink_strength = 32.0

    for c in classes:
        xc = x_train[labels == c]
        n_c = xc.shape[0]
        mean_c = xc.mean(axis=0)

        if n_c > 1:
            var_c = np.mean((xc - mean_c) ** 2, axis=0)
        else:
            var_c = global_var

        weight = n_c / (n_c + shrink_strength)
        var_c = weight * var_c + (1.0 - weight) * global_var
        var_c = np.maximum(var_c, 1e-8)

        diff = x_eval - mean_c
        dist = np.sum((diff * diff) / var_c, axis=1)
        best = np.minimum(best, dist)

    return best
