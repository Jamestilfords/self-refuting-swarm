import numpy as np
from scipy.linalg import eigh

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions are tightly constrained for known intents, so "
    "aggregate standardized residual energy in those directions exposes off-manifold OOS inputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)

    global_mean = x.mean(axis=0)
    means = np.empty((k, d), dtype=np.float64)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]
    means[~present] = global_mean

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x), 1)
    scale = max(float(np.trace(cov)) / max(d, 1), 1e-12)
    cov.flat[:: d + 1] += 1e-4 * scale

    n_tail = min(32, d)
    vals, vecs = eigh(cov, subset_by_index=[0, n_tail - 1], check_finite=False)
    vals = np.maximum(vals, 1e-8 * scale)

    pred = np.argmax(ev.logits, axis=1)
    eval_residuals = z - means[pred]
    projections = eval_residuals @ vecs
    scores = np.sum((projections * projections) / vals[None, :], axis=1)

    return np.asarray(scores, dtype=np.float64)
