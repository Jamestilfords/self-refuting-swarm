import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "The span of class-mean differences contains directions that separate known intents, while the orthogonal complement captures variation not useful for intent discrimination. OOS queries can retain unusually large residual components outside this discriminative in-scope subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0135"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=np.float64)

    classes, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    means = np.zeros((classes.size, x.shape[1]), dtype=np.float64)
    np.add.at(means, inv, x)
    means /= counts[:, None]

    global_mean = np.average(means, axis=0, weights=counts)
    between = (means - global_mean) * np.sqrt(counts[:, None])

    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    tol = singular[0] * max(between.shape) * np.finfo(np.float64).eps if singular.size else 0.0
    rank = int(np.sum(singular > tol))
    if rank == 0:
        rank = 1
        basis = np.zeros((x.shape[1], 1), dtype=np.float64)
        basis[0, 0] = 1.0
    else:
        basis = vt[:rank].T

    def residual_norm(a):
        centered = a - global_mean
        total_sq = np.sum(centered * centered, axis=1)
        projected = centered @ basis
        proj_sq = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total_sq - proj_sq, 0.0))

    train_residual = residual_norm(x)
    eval_residual = residual_norm(z)

    median = np.median(train_residual)
    mad = np.median(np.abs(train_residual - median))
    scale = max(1.4826 * mad, np.std(train_residual), 1e-8)

    return np.log1p(np.maximum(eval_residual - median, 0.0) / scale)
