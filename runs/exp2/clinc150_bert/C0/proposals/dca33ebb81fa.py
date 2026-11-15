import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions encode the representation components most useful for separating known intents. OOS inputs should retain unusually large energy in the orthogonal remainder, which is weakly explained by the in-scope class-mean subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    classes, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inv, x)
    means /= counts[:, None]

    global_mean = np.mean(x, axis=0)
    mean_offsets = means - global_mean
    _, s, vt = np.linalg.svd(mean_offsets, full_matrices=False)

    if s.size == 0:
        return np.sum((z - global_mean) ** 2, axis=1)

    tol = max(float(s[0]) * 1e-10, 1e-12)
    rank = int(np.sum(s > tol))
    rank = max(1, min(rank, 64, vt.shape[0]))
    basis = vt[:rank].T

    train_centered = x - global_mean
    eval_centered = z - global_mean

    train_total = np.sum(train_centered * train_centered, axis=1)
    train_proj = np.sum((train_centered @ basis) ** 2, axis=1)
    train_residual = np.maximum(train_total - train_proj, 0.0)

    eval_total = np.sum(eval_centered * eval_centered, axis=1)
    eval_proj = np.sum((eval_centered @ basis) ** 2, axis=1)
    eval_residual = np.maximum(eval_total - eval_proj, 0.0)

    scale = np.median(train_residual) + 1e-12
    return np.log1p(eval_residual / scale)
