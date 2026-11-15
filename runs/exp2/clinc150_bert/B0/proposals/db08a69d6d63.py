import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions can expose small but systematic deviations that are suppressed by dominant semantic variation. A nearest-class tail-subspace Mahalanobis score emphasizes these directions without relying on evaluation-set calibration."
SOURCE = "https://www.emergentmind.com/papers/2101.03778"
PARENTS = []
MEMORY_USED = ["m_0001", "m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)
    q = np.asarray(ev.pooled, dtype=np.float64)

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    d = x.shape[1]

    counts = np.bincount(inverse, minlength=n_classes).astype(np.float64)
    means = np.zeros((n_classes, d), dtype=np.float64)
    np.add.at(means, inverse, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[inverse]
    cov = (residuals.T @ residuals) / max(len(x) - n_classes, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    tail_dim = min(96, d)
    tail_vals = eigvals[:tail_dim]
    tail_vecs = eigvecs[:, :tail_dim]

    scale_ref = max(float(eigvals[-1]), 1e-12)
    variance_floor = scale_ref * 1e-5
    inv_scale = 1.0 / np.sqrt(np.maximum(tail_vals, variance_floor))

    q_tail = (q @ tail_vecs) * inv_scale
    mean_tail = (means @ tail_vecs) * inv_scale

    q_norm = np.sum(q_tail * q_tail, axis=1, keepdims=True)
    mean_norm = np.sum(mean_tail * mean_tail, axis=1)[None, :]
    distances = q_norm + mean_norm - 2.0 * (q_tail @ mean_tail.T)

    scores = np.min(distances, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
