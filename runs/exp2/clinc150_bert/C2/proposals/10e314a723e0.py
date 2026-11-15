import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class means define a low-rank discriminative subspace; in-scope examples should concentrate near this subspace while OOS queries retain larger residual components orthogonal to class-discriminative variation."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=float)

    classes, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv, minlength=len(classes)).astype(float)
    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inv, x)
    means /= counts[:, None]

    center = x.mean(axis=0)
    between = means - center
    _, singular, vt = np.linalg.svd(between, full_matrices=False)

    if singular.size == 0 or singular[0] <= 1e-12:
        residual = z - center
        return np.sum(residual * residual, axis=1)

    keep = singular > (singular[0] * 1e-8)
    basis = vt[keep]
    dz = z - center
    projected = dz @ basis.T
    residual_sq = np.sum(dz * dz, axis=1) - np.sum(projected * projected, axis=1)
    return np.maximum(residual_sq, 0.0)
