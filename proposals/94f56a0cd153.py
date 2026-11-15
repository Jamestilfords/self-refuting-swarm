import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "The span of class centroids captures directions used to discriminate known intents, "
    "while local support in its orthogonal complement detects novel residual semantic content."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    mean = np.mean(x, axis=0)
    xc = x - mean
    zc = z - mean

    classes, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    class_means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(class_means, inv, x)
    class_means /= counts[:, None]
    class_means -= mean

    _, singular, vt = np.linalg.svd(class_means, full_matrices=False)
    tol = max(class_means.shape) * np.finfo(float).eps * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > tol))

    if rank > 0:
        basis = vt[:rank]
        train_residual = xc - (xc @ basis.T) @ basis
        eval_residual = zc - (zc @ basis.T) @ basis
    else:
        train_residual = xc
        eval_residual = zc

    distances = ctx.h.knn(train_residual, eval_residual, k=15, l2=False)
    residual_energy = np.sum(eval_residual * eval_residual, axis=1)
    scores = np.asarray(distances, dtype=float) + 1e-10 * residual_energy
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
