import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The span of class means captures directions that distinguish known intents, while residual "
    "encoder energy measures variation not explained by those discriminative directions. OOS queries "
    "should retain more energy in this class-mean-orthogonal remainder."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    center = x.mean(axis=0)
    n_classes = int(np.max(y)) + 1
    means = np.empty((n_classes, x.shape[1]), dtype=np.float64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)

    for c in range(n_classes):
        members = x[y == c]
        means[c] = members.mean(axis=0) if len(members) else center

    between = (means - center) * np.sqrt(np.maximum(counts, 1.0))[:, None]
    _, singular_values, vt = np.linalg.svd(between, full_matrices=False)

    if singular_values.size:
        tol = singular_values[0] * max(between.shape) * np.finfo(np.float64).eps
        rank = int(np.sum(singular_values > tol))
    else:
        rank = 0

    if rank:
        basis = vt[:rank]
        train_residual = x - center - (x - center) @ basis.T @ basis
        eval_residual = z - center - (z - center) @ basis.T @ basis
    else:
        train_residual = x - center
        eval_residual = z - center

    train_energy = np.sum(train_residual * train_residual, axis=1)
    eval_energy = np.sum(eval_residual * eval_residual, axis=1)

    scale = np.quantile(train_energy, 0.95)
    scale = max(float(scale), np.finfo(np.float64).eps)
    return np.nan_to_num(eval_energy / scale, nan=0.0, posinf=1e12, neginf=0.0)
