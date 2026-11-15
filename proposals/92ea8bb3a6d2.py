import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions are tightly constrained for supported intents. "
    "A classifier-selected aggregate of coordinate-wise excess beyond train-fitted trailing residual thresholds detects broad low-variance violations without fitting a full Gaussian density."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006", "m_0170"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    xe = np.asarray(ev.pooled, dtype=np.float64)
    n, d = x.shape
    k = tr.logits.shape[1]

    means = np.zeros((k, d), dtype=np.float64)
    global_mean = x.mean(axis=0)
    counts = np.bincount(y, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)
        else:
            means[c] = global_mean

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(n - k, 1)
    cov.flat[:: d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)

    vals, vecs = np.linalg.eigh(cov)
    q = min(80, d)
    idx = np.argsort(vals)[:q]
    basis = vecs[:, idx]
    scale = np.sqrt(np.maximum(vals[idx], 1e-8))

    train_z = (residuals @ basis) / scale
    thresholds = np.zeros((k, q), dtype=np.float64)
    center = np.zeros(k, dtype=np.float64)
    spread = np.ones(k, dtype=np.float64)

    for c in range(k):
        zc = train_z[y == c] if counts[c] > 3 else train_z
        thresholds[c] = np.quantile(np.abs(zc), 0.975, axis=0)
        excess = np.maximum(np.abs(zc) - thresholds[c], 0.0).sum(axis=1)
        center[c] = np.median(excess)
        spread[c] = np.median(np.abs(excess - center[c])) * 1.4826 + 1e-6

    pred = np.argmax(ev.logits, axis=1)
    z_eval = ((xe - means[pred]) @ basis) / scale
    excess_eval = np.maximum(np.abs(z_eval) - thresholds[pred], 0.0).sum(axis=1)
    return (excess_eval - center[pred]) / spread[pred]
