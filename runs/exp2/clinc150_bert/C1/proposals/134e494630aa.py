import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-mean directions encode features most useful for separating known intents; OOS queries should retain atypical structure in the orthogonal remainder. A kNN score in this residual space measures novelty after removing discriminative in-scope variation."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(classes)).astype(float)
    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inverse, x)
    means /= counts[:, None]

    global_mean = x.mean(axis=0)
    mean_offsets = means - global_mean
    _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)

    tol = max(mean_offsets.shape) * np.finfo(float).eps * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > tol))
    rank = min(rank, max(0, len(classes) - 1))

    xc = x - global_mean
    zc = z - global_mean
    if rank > 0:
        basis = vt[:rank].T
        train_residual = xc - (xc @ basis) @ basis.T
        eval_residual = zc - (zc @ basis) @ basis.T
    else:
        train_residual = xc
        eval_residual = zc

    k = min(20, len(train_residual))
    return np.asarray(ctx.h.knn(train_residual, eval_residual, k=k, l2=True), dtype=float)
