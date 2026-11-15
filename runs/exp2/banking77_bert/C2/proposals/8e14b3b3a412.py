import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class means span directions that are explicitly useful for separating known intents. OOS inputs are expected to retain unusually large or atypical residual components outside this discriminative class-mean subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels)

    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    global_mean = x.mean(axis=0)

    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inverse, x)
    means /= counts[:, None]

    weighted_means = (means - global_mean) * np.sqrt(counts[:, None] / counts.sum())
    _, singular, vt = np.linalg.svd(weighted_means, full_matrices=True)

    if singular.size:
        threshold = singular[0] * 1e-8
        rank = int(np.sum(singular > threshold))
    else:
        rank = 0
    rank = min(rank, x.shape[1] - 1)

    residual_basis = vt[rank:]
    train_residual = (x - global_mean) @ residual_basis.T
    eval_residual = (z - global_mean) @ residual_basis.T

    scale = np.sqrt(np.mean(train_residual * train_residual, axis=0))
    floor = max(float(np.median(scale)) * 1e-3, 1e-8)
    scale = np.maximum(scale, floor)

    return np.sum((eval_residual / scale) ** 2, axis=1)
