import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Projecting out the class-mean discriminative subspace leaves within-intent residual "
    "structure, where OOS queries should have weak local support from training examples."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0047", "m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    classes = np.unique(y)
    mean = x.mean(axis=0)
    class_means = np.empty((classes.size, x.shape[1]), dtype=np.float64)
    for i, c in enumerate(classes):
        class_means[i] = x[y == c].mean(axis=0) - mean

    _, s, vt = np.linalg.svd(class_means, full_matrices=False)
    tol = max(class_means.shape) * np.finfo(np.float64).eps * max(s[0], 1.0)
    rank = int(np.sum(s > tol))

    if rank == 0:
        train_residual = x - mean
        eval_residual = z - mean
    else:
        basis = vt[:rank]
        xc = x - mean
        zc = z - mean
        train_residual = xc - (xc @ basis.T) @ basis
        eval_residual = zc - (zc @ basis.T) @ basis

    k = min(20, max(1, len(tr.labels) - 1))
    return ctx.h.knn(train_residual, eval_residual, k=k, l2=True)
