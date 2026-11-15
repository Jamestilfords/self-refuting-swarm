import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-mean directions encode the most discriminative in-scope intent variation, so removing them "
    "reveals residual encoder variation that should be small for training-like utterances but larger for OOS inputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005", "m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    mu = x.mean(axis=0)
    classes = np.unique(y)
    class_means = np.vstack([x[y == c].mean(axis=0) for c in classes])
    mean_offsets = class_means - mu

    # The span of class-mean offsets is the supervised discriminative subspace.
    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    if singular_values.size:
        tol = singular_values[0] * 1e-10
        rank = int(np.sum(singular_values > tol))
    else:
        rank = 0
    basis = vt[:rank]

    def residual(a):
        centered = a - mu
        if rank:
            centered = centered - (centered @ basis.T) @ basis
        return centered

    train_residual = residual(x)
    eval_residual = residual(z)

    # Diagonal whitening on the class-mean complement, with a train-derived floor.
    var = np.mean(train_residual * train_residual, axis=0)
    positive = var[var > 0]
    floor = np.median(positive) * 1e-6 if positive.size else 1e-8
    var = np.maximum(var, max(floor, 1e-12))

    score_value = np.sum((eval_residual * eval_residual) / var, axis=1)
    return np.nan_to_num(score_value, nan=0.0, posinf=1e12, neginf=0.0)
