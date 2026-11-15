import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A shrunk diagonal QDA mixture models each intent with its own feature-wise spread and scores "
    "the full in-scope mixture likelihood rather than only the nearest class distance."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)
    classes, inverse = np.unique(labels, return_inverse=True)

    n_train, dim = x_train.shape
    n_classes = classes.shape[0]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    scale = max(float(np.mean(global_var)), 1e-10)
    floor = max(1e-8 * scale, 1e-12)
    global_var = np.maximum(global_var, floor)

    log_terms = np.empty((x_eval.shape[0], n_classes), dtype=np.float64)
    counts = np.bincount(inverse, minlength=n_classes).astype(np.float64)
    log_priors = np.log((counts + 1.0) / (n_train + n_classes))
    log_2pi = np.log(2.0 * np.pi)

    for j in range(n_classes):
        members = x_train[inverse == j]
        count = members.shape[0]
        mean = np.mean(members, axis=0)

        if count > 1:
            var = np.mean((members - mean) ** 2, axis=0)
        else:
            var = global_var.copy()

        shrink = count / (count + 24.0)
        var = shrink * var + (1.0 - shrink) * global_var
        var = np.maximum(var, floor)

        diff = x_eval - mean
        quad = np.sum((diff * diff) / var, axis=1)
        log_det = np.sum(np.log(var))
        log_terms[:, j] = log_priors[j] - 0.5 * (quad + log_det + dim * log_2pi)

    scores = -logsumexp(log_terms, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
