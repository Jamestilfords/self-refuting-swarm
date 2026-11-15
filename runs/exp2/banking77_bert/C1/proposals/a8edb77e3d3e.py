import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean differences define directions that are useful for discriminating known intents. Queries with unusually large residual energy outside this discriminative subspace are more likely to be out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    center = x.mean(axis=0)
    classes = np.unique(y)
    means = np.empty((classes.size, x.shape[1]), dtype=np.float64)
    for i, c in enumerate(classes):
        means[i] = x[y == c].mean(axis=0) - center

    _, s, vt = np.linalg.svd(means, full_matrices=False)
    if s.size:
        tol = max(means.shape) * np.finfo(np.float64).eps * s[0]
        rank = max(1, int(np.sum(s > tol)))
        basis = vt[:rank]
    else:
        basis = np.zeros((1, x.shape[1]), dtype=np.float64)

    def residual_energy(a):
        centered = a - center
        total = np.sum(centered * centered, axis=1)
        projected = np.sum((centered @ basis.T) ** 2, axis=1)
        return np.maximum(total - projected, 0.0)

    train_residual = residual_energy(x)
    eval_residual = residual_energy(z)
    med = np.median(train_residual)
    mad = np.median(np.abs(train_residual - med))
    scale = max(1.4826 * mad, np.finfo(np.float64).eps)
    return (eval_residual - med) / scale
