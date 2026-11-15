import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture encoder variation that discriminates known intents, while "
    "large residual energy in their orthogonal complement indicates semantic content unexplained by the intent taxonomy."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0069"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    center = x.mean(axis=0)
    classes, inverse = np.unique(labels, return_inverse=True)
    means = np.zeros((classes.size, x.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse, minlength=classes.size).astype(np.float64)
    np.add.at(means, inverse, x)
    means /= counts[:, None]
    mean_offsets = means - center

    _, singular, vh = np.linalg.svd(mean_offsets, full_matrices=False)
    tol = max(mean_offsets.shape) * np.finfo(np.float64).eps * singular[0] if singular.size else 0.0
    rank = int(np.sum(singular > tol))
    rank = max(1, min(rank, 64, vh.shape[0]))
    basis = vh[:rank]

    train_centered = x - center
    train_proj = train_centered @ basis.T
    train_residual = np.sum(train_centered * train_centered, axis=1) - np.sum(train_proj * train_proj, axis=1)
    train_residual = np.maximum(train_residual, 0.0)
    scale = np.median(train_residual) + 1e-8

    eval_centered = z - center
    eval_proj = eval_centered @ basis.T
    residual = np.sum(eval_centered * eval_centered, axis=1) - np.sum(eval_proj * eval_proj, axis=1)
    residual = np.maximum(residual, 0.0)

    return np.log1p(residual / scale) + 1e-8 * ctx.h.energy(ev.logits)
