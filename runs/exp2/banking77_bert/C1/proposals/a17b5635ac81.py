import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs may show atypical residuals in low-variance within-class directions. A soft nearest-class aggregation over a learned trailing covariance subspace retains uncertainty among several nearby intents instead of using only the single closest class."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = len(classes)
    dim = x_train.shape[1]

    means = np.zeros((n_classes, dim), dtype=float)
    np.add.at(means, inverse, x_train)
    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x_train - means[inverse]
    denom = max(x_train.shape[0] - n_classes, 1)
    within_cov = (centered.T @ centered) / float(denom)

    _, eigvecs = np.linalg.eigh(within_cov)
    tail_dim = min(64, dim)
    tail_basis = eigvecs[:, :tail_dim]

    train_tail = centered @ tail_basis
    class_tail_means = means @ tail_basis
    eval_tail = x_eval @ tail_basis

    train_norm = np.sum(train_tail * train_tail, axis=1, keepdims=True)
    mean_norm = np.sum(class_tail_means * class_tail_means, axis=1)
    train_dist = train_norm + mean_norm[None, :] - 2.0 * train_tail @ class_tail_means.T
    train_dist = np.maximum(train_dist, 0.0)

    temperature = float(np.median(np.min(train_dist, axis=1)))
    temperature = max(temperature, 1e-8)

    eval_norm = np.sum(eval_tail * eval_tail, axis=1, keepdims=True)
    eval_dist = eval_norm + mean_norm[None, :] - 2.0 * eval_tail @ class_tail_means.T
    eval_dist = np.maximum(eval_dist, 0.0)

    scores = -temperature * logsumexp(-eval_dist / temperature, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
