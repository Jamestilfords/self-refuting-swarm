import numpy as np
from scipy.linalg import eigh

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance directions of the pooled within-intent covariance encode semantic constraints shared across known intents. Measuring residual energy in those directions relative to the classifier-predicted intent can expose queries that are locally implausible even when they lie near a broad in-scope region."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n, d = x_train.shape
    k = tr.logits.shape[1]

    counts = np.bincount(labels, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, labels, x_train)
    global_mean = x_train.mean(axis=0)
    means = np.empty((k, d), dtype=float)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]
    means[~present] = global_mean

    centered = x_train - means[labels]
    cov = (centered.T @ centered) / max(n - k, 1)
    cov = 0.5 * (cov + cov.T)

    tail_dim = min(96, d)
    vals, vecs = eigh(cov, subset_by_index=(0, tail_dim - 1),
                      driver="evr", check_finite=False)
    scale = max(float(np.mean(np.diag(cov))), 1e-12)
    vals = np.maximum(vals, scale * 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    residual = x_eval - means[predicted]
    projected = residual @ vecs
    tail_energy = np.sum((projected * projected) / vals[None, :], axis=1)

    scores = np.log1p(np.maximum(tail_energy, 0.0))
    scores += 1e-10 * np.asarray(ctx.h.energy(ev.logits), dtype=float)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
