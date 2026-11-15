import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-mean directions encode the classifier's most discriminative variation, so removing them exposes residual semantic variation that should remain close to training examples for in-scope queries. OOS inputs are scored by their residual-space neighborhood distance."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=float)

    classes, inv = np.unique(y, return_inverse=True)
    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inv, x)
    counts = np.bincount(inv, minlength=len(classes)).astype(float)
    means /= np.maximum(counts[:, None], 1.0)

    global_mean = np.average(means, axis=0, weights=counts)
    between = (means - global_mean) * np.sqrt(counts[:, None])

    _, s, vt = np.linalg.svd(between, full_matrices=False)
    tol = max(float(s[0]) if s.size else 0.0, 1.0) * 1e-10
    rank = int(np.sum(s > tol))
    rank = min(rank, 64)

    if rank > 0:
        basis = vt[:rank].T
        xr = x - (x - global_mean) @ basis @ basis.T
        zr = z - (z - global_mean) @ basis @ basis.T
    else:
        xr = x
        zr = z

    scale = np.std(xr, axis=0)
    scale = np.maximum(scale, 1e-6)
    xr = xr / scale
    zr = zr / scale

    return ctx.h.knn(xr, zr, k=10, l2=False)
