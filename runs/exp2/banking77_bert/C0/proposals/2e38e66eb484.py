import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-mean directions capture the subspace most useful for separating known intents. Measuring local nearest-neighbor novelty after removing that subspace highlights inputs with unusual nondiscriminative residual structure."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    center = np.mean(x_train, axis=0)
    classes, inverse = np.unique(labels, return_inverse=True)
    means = np.zeros((len(classes), x_train.shape[1]), dtype=float)
    np.add.at(means, inverse, x_train)
    counts = np.bincount(inverse, minlength=len(classes)).astype(float)
    means /= np.maximum(counts[:, None], 1.0)

    offsets = means - center
    _, singular, vt = np.linalg.svd(offsets, full_matrices=False)
    if singular.size:
        threshold = np.max(singular) * 1e-8
        rank = int(np.sum(singular > threshold))
    else:
        rank = 0

    if rank:
        basis = vt[:rank].T
        train_centered = x_train - center
        eval_centered = x_eval - center
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = x_train - center
        eval_residual = x_eval - center

    scale = np.sqrt(np.mean(train_residual * train_residual, axis=0))
    scale = np.maximum(scale, 1e-6)
    train_residual = train_residual / scale
    eval_residual = eval_residual / scale

    k = min(15, len(train_residual))
    distances = ctx.h.knn(train_residual, eval_residual, k=k, l2=False)
    return np.log1p(np.maximum(distances, 0.0))
