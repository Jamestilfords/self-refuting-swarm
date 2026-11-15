import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions capture the most discriminative in-scope variation, while unusually large residual variation orthogonal to that subspace indicates queries that do not align with known intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=float)

    center = x.mean(axis=0)
    classes = np.unique(y)
    means = np.empty((len(classes), x.shape[1]), dtype=float)
    for i, c in enumerate(classes):
        means[i] = x[y == c].mean(axis=0) - center

    _, singular, vt = np.linalg.svd(means, full_matrices=False)
    max_rank = min(64, vt.shape[0])
    if singular.size:
        threshold = singular[0] * 1e-8
        rank = min(max_rank, max(1, int(np.sum(singular > threshold))))
    else:
        rank = 0

    if rank > 0:
        basis = vt[:rank]
        train_centered = x - center
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_centered = z - center
        residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = x - center
        residual = z - center

    scale = np.mean(train_residual * train_residual, axis=0)
    scale = np.maximum(scale, 1e-8)
    scores = np.sum((residual * residual) / scale, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
