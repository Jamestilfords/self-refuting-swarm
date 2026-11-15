import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean discriminative subspace leaves representation components not explained by known intent identity. "
    "OOS queries should have both large residual energy and weak local support among training residual directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0130", "m_0132"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    classes = np.unique(y)
    center = x.mean(axis=0)
    means = np.vstack([x[y == c].mean(axis=0) for c in classes])
    between = means - center

    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size:
        tol = singular[0] * max(between.shape) * np.finfo(np.float64).eps
        rank = int(np.sum(singular > tol))
    else:
        rank = 0

    xc = x - center
    zc = z - center
    if rank > 0:
        basis = vt[:rank]
        xr = xc - (xc @ basis.T) @ basis
        zr = zc - (zc @ basis.T) @ basis
    else:
        xr = xc
        zr = zc

    train_norm = np.sqrt(np.sum(xr * xr, axis=1))
    eval_norm = np.sqrt(np.sum(zr * zr, axis=1))

    k = min(10, len(xr))
    local_train = ctx.h.knn(xr, xr, k=k, l2=True)
    local_eval = ctx.h.knn(xr, zr, k=k, l2=True)

    train_score = train_norm * (1.0 + np.maximum(local_train, 0.0))
    eval_score = eval_norm * (1.0 + np.maximum(local_eval, 0.0))

    median = np.median(train_score)
    scale = np.median(np.abs(train_score - median))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(train_score)
    scale = max(float(scale), 1e-12)

    return np.nan_to_num((eval_score - median) / scale, nan=0.0, posinf=1e12, neginf=-1e12)
