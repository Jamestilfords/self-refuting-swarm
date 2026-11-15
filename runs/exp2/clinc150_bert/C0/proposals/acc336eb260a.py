import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "The span of class means captures directions used to separate known intents. Queries with substantial energy outside this discriminative subspace are less consistent with the in-scope class structure."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    center = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.maximum(scale, 1e-6)

    z_train = (x_train - center) / scale
    z_eval = (x_eval - center) / scale

    classes, inverse = np.unique(labels, return_inverse=True)
    means = np.zeros((classes.size, z_train.shape[1]), dtype=float)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    np.add.at(means, inverse, z_train)
    means /= np.maximum(counts[:, None], 1.0)
    means -= np.mean(means, axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(means, full_matrices=False)
    if singular_values.size:
        threshold = max(singular_values[0] * 1e-8, 1e-10)
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0

    if rank == 0:
        residual_sq = np.sum(z_eval * z_eval, axis=1)
    else:
        basis = vt[:rank]
        projected_sq = np.sum((z_eval @ basis.T) ** 2, axis=1)
        total_sq = np.sum(z_eval * z_eval, axis=1)
        residual_sq = np.maximum(total_sq - projected_sq, 0.0)

    return np.nan_to_num(residual_sq / z_eval.shape[1], nan=0.0, posinf=1e12, neginf=0.0)
