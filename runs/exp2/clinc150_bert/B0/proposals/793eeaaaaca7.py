import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions encode representation variation that separates known intents, "
    "so OOS queries can be detected by poor local support in the complementary residual space."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0124", "m_0129", "m_0131"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)

    mean = x_train.mean(axis=0, keepdims=True)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    classes, inverse = np.unique(labels, return_inverse=True)
    class_means = np.zeros((classes.size, x_train.shape[1]), dtype=np.float64)
    np.add.at(class_means, inverse, centered_train)
    counts = np.bincount(inverse, minlength=classes.size).astype(np.float64)
    class_means /= np.maximum(counts[:, None], 1.0)

    _, singular_values, vt = np.linalg.svd(class_means, full_matrices=False)
    tol = np.finfo(np.float64).eps * max(class_means.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tol))
    if rank == 0:
        residual_train = centered_train
        residual_eval = centered_eval
    else:
        basis = vt[:rank].T
        residual_train = centered_train - (centered_train @ basis) @ basis.T
        residual_eval = centered_eval - (centered_eval @ basis) @ basis.T

    return ctx.h.knn(residual_train, residual_eval, k=10, l2=False)
