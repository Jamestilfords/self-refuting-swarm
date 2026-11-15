import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture features most useful for separating known intents. "
    "OOS queries should retain unusually large standardized energy in the orthogonal remainder."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels)

    classes, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    global_mean = x.mean(axis=0)

    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inv, x)
    means /= counts[:, None]

    between = (means - global_mean) * np.sqrt(counts[:, None] / counts.sum())
    _, singular, vt = np.linalg.svd(between, full_matrices=False)

    if singular.size:
        tol = singular[0] * max(between.shape) * np.finfo(float).eps
        rank = int(np.sum(singular > tol))
    else:
        rank = 0

    if rank > 0:
        basis = vt[:rank].T
        train_centered = x - global_mean
        eval_centered = z - global_mean
        train_resid = train_centered - (train_centered @ basis) @ basis.T
        eval_resid = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_resid = x - global_mean
        eval_resid = z - global_mean

    scale = np.mean(train_resid * train_resid, axis=0)
    floor = max(float(np.mean(scale)) * 1e-8, 1e-12)
    scale = np.maximum(scale, floor)

    train_energy = np.mean((train_resid * train_resid) / scale, axis=1)
    center = float(np.median(train_energy))
    spread = float(np.median(np.abs(train_energy - center)))
    spread = max(1.4826 * spread, 1e-6)

    eval_energy = np.mean((eval_resid * eval_resid) / scale, axis=1)
    return (eval_energy - center) / spread
