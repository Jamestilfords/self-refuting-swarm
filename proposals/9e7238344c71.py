import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The span of train class means contains directions that distinguish known intents. "
    "OOS queries can retain confidence yet have substantial encoder components outside this discriminative class-mean subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)

    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    global_mean = np.mean(x, axis=0)

    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inverse, x)
    means /= counts[:, None]

    between = (means - global_mean) * np.sqrt(counts)[:, None]
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    tol = np.finfo(float).eps * max(between.shape) * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > tol))
    basis = vt[:rank].T

    def residual_sq(a):
        centered = a - global_mean
        if rank:
            centered = centered - (centered @ basis) @ basis.T
        return np.sum(centered * centered, axis=1)

    train_residual = residual_sq(x)
    scale = np.median(train_residual) + 1e-12
    eval_residual = residual_sq(z)
    return np.log1p(eval_residual / scale)
