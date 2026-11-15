import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions explain representation variation used to discriminate known intents. "
    "Large coordinate excursions in the orthogonal residual subspace indicate semantic content unsupported by those discriminative directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=float)

    mean = x.mean(axis=0)
    xc = x - mean
    zc = z - mean

    classes, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    class_means = np.zeros((classes.size, x.shape[1]), dtype=float)
    np.add.at(class_means, inv, x)
    class_means /= np.maximum(counts[:, None], 1.0)
    class_means -= mean

    _, singular, vt = np.linalg.svd(class_means, full_matrices=False)
    if singular.size:
        keep = singular > (singular[0] * 1e-8 + 1e-12)
    else:
        keep = np.zeros(0, dtype=bool)
    u = vt[keep].T

    if u.shape[1]:
        xr = xc - (xc @ u) @ u.T
        zr = zc - (zc @ u) @ u.T
    else:
        xr = xc
        zr = zc

    cov = (xr.T @ xr) / max(xr.shape[0] - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    largest = max(float(eigval[-1]), 1e-12)
    active = eigval > largest * 1e-6

    if not np.any(active):
        train_raw = np.sum(xr * xr, axis=1)
        eval_raw = np.sum(zr * zr, axis=1)
    else:
        vals = eigval[active]
        vecs = eigvec[:, active]
        take = min(128, vals.size)
        vals = vals[:take]
        vecs = vecs[:, :take]

        train_coord = (xr @ vecs) / np.sqrt(vals + 1e-12)
        eval_coord = (zr @ vecs) / np.sqrt(vals + 1e-12)
        train_raw = np.max(train_coord * train_coord, axis=1)
        eval_raw = np.max(eval_coord * eval_coord, axis=1)

    ordered = np.sort(train_raw)
    tail = np.searchsorted(ordered, eval_raw, side="right") / float(ordered.size)
    return tail + 1e-4 * np.log1p(np.maximum(eval_raw, 0.0))
