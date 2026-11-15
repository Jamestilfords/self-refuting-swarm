import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances preserve directions that are naturally broad for one intent but anomalous for another, while shrinkage toward the global variance stabilizes small classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = len(classes)
    dim = x_train.shape[1]

    global_var = np.var(x_train, axis=0)
    scale = float(np.mean(global_var))
    floor = max(scale * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    means = np.zeros((n_classes, dim), dtype=np.float64)
    variances = np.zeros((n_classes, dim), dtype=np.float64)

    for c in range(n_classes):
        members = x_train[inverse == c]
        count = members.shape[0]
        mean = np.mean(members, axis=0)
        means[c] = mean

        if count > 1:
            empirical = np.var(members, axis=0, ddof=1)
        else:
            empirical = global_var

        shrink = dim / (dim + max(count - 1, 1))
        variances[c] = np.maximum(
            (1.0 - shrink) * empirical + shrink * global_var,
            floor,
        )

    precision = 1.0 / variances
    logdet = np.sum(np.log(variances), axis=1)
    weighted_means = means * precision
    mean_quadratic = np.sum(means * weighted_means, axis=1)

    quadratic = (
        (x_eval * x_eval) @ precision.T
        - 2.0 * (x_eval @ weighted_means.T)
        + mean_quadratic[None, :]
    )
    nll = 0.5 * (quadratic + logdet[None, :])
    return np.min(nll, axis=1)
