import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The subspace spanned by class-mean differences contains directions most useful for separating "
    "known intents; OOS queries are expected to retain unusually large standardized energy in its orthogonal remainder."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0080", "m_0081"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=np.float64)

    mean = x.mean(axis=0)
    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)

    class_means = np.zeros((classes.size, x.shape[1]), dtype=np.float64)
    np.add.at(class_means, inverse, x)
    class_means /= counts[:, None]

    centered = x - class_means[inverse]
    within_var = np.mean(centered * centered, axis=0)
    scale = np.sqrt(np.maximum(within_var, 1e-8))

    mean_subspace = (class_means - mean) / scale
    _, singular, vt = np.linalg.svd(mean_subspace, full_matrices=False)

    power = singular * singular
    if power.sum() > 1e-12:
        cumulative = np.cumsum(power) / power.sum()
        rank = int(np.searchsorted(cumulative, 0.95) + 1)
    else:
        rank = 1
    rank = max(1, min(rank, vt.shape[0], 128))
    basis = vt[:rank].T

    train_standard = (x - mean) / scale
    eval_standard = (z - mean) / scale

    train_residual = train_standard - (train_standard @ basis) @ basis.T
    eval_residual = eval_standard - (eval_standard @ basis) @ basis.T

    train_energy = np.sum(train_residual * train_residual, axis=1)
    eval_energy = np.sum(eval_residual * eval_residual, axis=1)

    location = np.median(train_energy)
    spread = np.median(np.abs(train_energy - location))
    spread = max(1.4826 * spread, 1e-6)

    return (eval_energy - location) / spread
