import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean discriminative subspace isolates encoder content not needed "
    "to distinguish known intents. OOS queries can have both unusual residual magnitude "
    "and unsupported residual direction even when their discriminative projection is confident."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0041", "m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=float)

    classes = np.unique(y)
    means = np.vstack([x[y == c].mean(axis=0) for c in classes])
    center = x.mean(axis=0)
    mean_offsets = means - center

    _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    tol = max(mean_offsets.shape) * np.finfo(float).eps * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > tol))
    basis = vt[:rank]

    x0 = x - center
    z0 = z - center
    if rank:
        xr = x0 - (x0 @ basis.T) @ basis
        zr = z0 - (z0 @ basis.T) @ basis
    else:
        xr = x0
        zr = z0

    train_radius = np.linalg.norm(xr, axis=1)
    eval_radius = np.linalg.norm(zr, axis=1)

    train_direction = ctx.h.knn(xr, xr, k=10, l2=True)
    eval_direction = ctx.h.knn(xr, zr, k=10, l2=True)

    def upper_ecdf(reference, values):
        ref = np.sort(np.asarray(reference, dtype=float))
        return np.searchsorted(ref, values, side="right") / float(ref.size + 1)

    radius_tail = upper_ecdf(train_radius, eval_radius)
    direction_tail = upper_ecdf(train_direction, eval_direction)

    return 0.45 * radius_tail + 0.55 * direction_tail
