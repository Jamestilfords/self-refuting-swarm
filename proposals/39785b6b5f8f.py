import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions are tightly constrained for supported intents. "
    "The fraction and magnitude of class-conditional trailing-coordinate tail violations exposes sparse OOS deviations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0044", "m_0006", "m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = x.mean(axis=0)
    means = np.empty((k, d), dtype=float)

    residuals = np.empty_like(x)
    for c in range(k):
        mask = y == c
        if np.any(mask):
            means[c] = x[mask].mean(axis=0)
        else:
            means[c] = global_mean
    residuals[:] = x - means[y]

    cov = (residuals.T @ residuals) / max(n - k, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    r = min(64, d)
    vals = np.maximum(eigvals[:r], 1e-8)
    trail = eigvecs[:, :r]
    scale = np.sqrt(vals)

    train_proj = (residuals @ trail) / scale
    thresholds = np.empty((k, r), dtype=float)
    for c in range(k):
        mask = y == c
        base = np.abs(train_proj[mask]) if np.any(mask) else np.abs(train_proj)
        thresholds[c] = np.quantile(base, 0.95, axis=0)
    thresholds = np.maximum(thresholds, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_residuals = z - means[pred]
    proj = np.abs((eval_residuals @ trail) / scale)
    excess = np.maximum(proj - thresholds[pred], 0.0)

    return np.mean(excess * excess, axis=1) + 0.25 * np.mean(excess > 0.0, axis=1)
