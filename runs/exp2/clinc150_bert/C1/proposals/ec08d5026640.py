import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions capture feature variation used to discriminate known intents. Queries with unusually large energy in the orthogonal residual subspace are less consistent with the in-scope intent manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    center = np.mean(x_train, axis=0)
    classes = np.unique(labels)
    means = np.vstack([np.mean(x_train[labels == c], axis=0) for c in classes])
    between = means - np.mean(means, axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(between, full_matrices=False)
    if singular_values.size:
        tol = np.finfo(float).eps * max(between.shape) * singular_values[0]
        rank = int(np.sum(singular_values > tol))
    else:
        rank = 0

    train_centered = x_train - center
    eval_centered = x_eval - center

    if rank > 0:
        basis = vt[:rank].T
        train_proj_sq = np.sum((train_centered @ basis) ** 2, axis=1)
        eval_proj_sq = np.sum((eval_centered @ basis) ** 2, axis=1)
        train_residual_sq = np.maximum(
            np.sum(train_centered ** 2, axis=1) - train_proj_sq, 0.0
        )
        eval_residual_sq = np.maximum(
            np.sum(eval_centered ** 2, axis=1) - eval_proj_sq, 0.0
        )
    else:
        train_residual_sq = np.sum(train_centered ** 2, axis=1)
        eval_residual_sq = np.sum(eval_centered ** 2, axis=1)

    scale = np.median(train_residual_sq)
    scale = max(float(scale), 1e-12)
    return np.nan_to_num(eval_residual_sq / scale, nan=0.0, posinf=1e12, neginf=0.0)
