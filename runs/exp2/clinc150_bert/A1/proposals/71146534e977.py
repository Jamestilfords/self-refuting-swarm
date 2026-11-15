import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "The class-mean subspace contains representation directions that separate known intents. "
    "After removing that discriminative component, OOS queries should have weak local support among in-scope residual patterns."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n_classes = int(np.max(y)) + 1
    counts = np.bincount(y, minlength=n_classes).astype(float)
    means = np.zeros((n_classes, x.shape[1]), dtype=float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    center = np.mean(x, axis=0)
    active_means = means[counts > 0] - center
    if active_means.shape[0] > 1:
        _, singular, vt = np.linalg.svd(active_means, full_matrices=False)
        tol = (
            max(active_means.shape)
            * np.finfo(float).eps
            * (singular[0] if singular.size else 1.0)
        )
        rank = int(np.sum(singular > tol))
        basis = vt[:max(rank, 1)]
        xr = (x - center) - ((x - center) @ basis.T) @ basis
        zr = (z - center) - ((z - center) @ basis.T) @ basis
    else:
        xr = x - center
        zr = z - center

    scale = np.sqrt(np.mean(xr * xr, axis=0))
    scale = np.maximum(scale, 1e-6)
    xr = xr / scale
    zr = zr / scale

    k = min(15, max(1, xr.shape[0]))
    local_distance = ctx.h.knn(xr, zr, k=k, l2=False)

    train_norm = np.sum(xr * xr, axis=1)
    eval_norm = np.sum(zr * zr, axis=1)
    norm_scale = max(float(np.median(train_norm)), 1e-6)

    return np.log1p(np.maximum(local_distance, 0.0)) + 0.05 * np.log1p(
        np.maximum(eval_norm, 0.0) / norm_scale
    )
