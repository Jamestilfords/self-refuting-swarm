import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances retain directions whose spread differs substantially across intents, while shrinkage toward the global variance stabilizes small classes. Queries that are implausible under every heteroscedastic class distribution receive high scores."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = len(classes)
    dim = x_train.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    means = np.empty((n_classes, dim), dtype=np.float64)
    variances = np.empty((n_classes, dim), dtype=np.float64)
    counts = np.bincount(inverse, minlength=n_classes).astype(np.float64)

    for c in range(n_classes):
        xc = x_train[inverse == c]
        mean = np.mean(xc, axis=0)
        raw_var = np.mean((xc - mean) ** 2, axis=0)
        shrink = counts[c] / (counts[c] + 20.0)
        means[c] = mean
        variances[c] = np.maximum(
            shrink * raw_var + (1.0 - shrink) * global_var,
            floor,
        )

    inv_var = 1.0 / variances
    log_det = np.sum(np.log(variances), axis=1)
    quadratic_constant = np.sum(means * means * inv_var, axis=1)

    distances = (
        (x_eval * x_eval) @ inv_var.T
        - 2.0 * (x_eval @ (means * inv_var).T)
        + quadratic_constant[None, :]
        + log_det[None, :]
    )

    priors = np.maximum(counts / np.sum(counts), 1e-12)
    neg_log_joint = 0.5 * distances - np.log(priors)[None, :]
    return np.min(neg_log_joint, axis=1)
