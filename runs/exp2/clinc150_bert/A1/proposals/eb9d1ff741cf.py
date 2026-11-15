import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A supervised class-mean subspace retains directions that distinguish known intents, "
    "so a large residual orthogonal to that subspace indicates variation unexplained by in-scope intent structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    mean = x.mean(axis=0)
    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(classes)).astype(np.float64)
    class_sums = np.zeros((len(classes), x.shape[1]), dtype=np.float64)
    np.add.at(class_sums, inverse, x)
    class_means = class_sums / np.maximum(counts[:, None], 1.0)

    between = (class_means - mean) * np.sqrt(counts[:, None])
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size:
        tol = singular[0] * max(between.shape) * np.finfo(np.float64).eps
        rank = max(1, int(np.sum(singular > tol)))
        basis = vt[:rank]
        centered = z - mean
        projection = centered @ basis.T
        residual = centered - projection @ basis
    else:
        residual = z - mean

    return np.sum(residual * residual, axis=1)
