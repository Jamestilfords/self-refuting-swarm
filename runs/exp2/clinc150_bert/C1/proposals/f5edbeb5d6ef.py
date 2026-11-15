import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions capture deviations that can be invisible in dominant semantic directions. The trailing-eigenspace residual distance measures whether a query violates class-conditional fine-grained constraints."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    classes, yi = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    dim = x.shape[1]

    means = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(yi, minlength=n_classes).astype(float)
    np.add.at(means, yi, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[yi]
    cov = (residuals.T @ residuals) / max(len(x) - n_classes, 1)
    vals, vecs = np.linalg.eigh(cov)

    n_trailing = max(8, min(dim // 4, 128))
    trailing_vecs = vecs[:, :n_trailing]
    trailing_vals = vals[:n_trailing]
    floor = max(float(np.median(vals)) * 1e-4, 1e-8)
    inv_scale = 1.0 / np.maximum(trailing_vals, floor)

    centered = z[:, None, :] - means[None, :, :]
    projected = np.matmul(centered, trailing_vecs)
    distances = np.sum(projected * projected * inv_scale[None, None, :], axis=2)
    return np.min(distances, axis=1)
