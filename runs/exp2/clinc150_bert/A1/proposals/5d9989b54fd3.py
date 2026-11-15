import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A shrunk diagonal covariance for each intent preserves intent-specific feature scales while avoiding unstable full class covariances. OOS queries should have low likelihood under every class-specific diagonal Gaussian."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)

    classes, counts = np.unique(labels, return_counts=True)
    n, d = x_train.shape

    global_var = np.var(x_train, axis=0) + 1e-6
    means = np.empty((len(classes), d), dtype=np.float64)
    variances = np.empty((len(classes), d), dtype=np.float64)

    for j, cls in enumerate(classes):
        xc = x_train[labels == cls]
        means[j] = np.mean(xc, axis=0)
        class_var = np.var(xc, axis=0) if len(xc) > 1 else global_var
        weight = len(xc) / (len(xc) + 20.0)
        variances[j] = weight * class_var + (1.0 - weight) * global_var

    floor = np.maximum(np.median(global_var) * 1e-4, 1e-8)
    variances = np.maximum(variances, floor)

    inv_var = 1.0 / variances
    weighted_means = means * inv_var
    mean_quad = np.sum(means * weighted_means, axis=1)
    log_det = np.sum(np.log(variances), axis=1)
    log_prior = np.log(counts.astype(np.float64) / float(n))

    quadratic = (
        (x_eval * x_eval) @ inv_var.T
        - 2.0 * (x_eval @ weighted_means.T)
        + mean_quad[None, :]
    )
    nll = 0.5 * np.maximum(quadratic, 0.0) + 0.5 * log_det[None, :] - log_prior[None, :]
    return np.nan_to_num(np.min(nll, axis=1), nan=0.0, posinf=1e12, neginf=-1e12)
