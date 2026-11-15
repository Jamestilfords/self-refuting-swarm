import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture features most useful for separating known intents. "
    "After removing them, OOS inputs should have residual directions unlike the residual variation seen among training queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    classes = np.unique(labels)
    center = np.mean(x_train, axis=0)
    class_means = np.vstack([
        np.mean(x_train[labels == c], axis=0) for c in classes
    ])
    offsets = class_means - center

    _, singular_values, vt = np.linalg.svd(offsets, full_matrices=False)
    if singular_values.size:
        threshold = singular_values[0] * 1e-7
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0
    rank = max(1, min(rank, vt.shape[0]))
    basis = vt[:rank]

    train_centered = x_train - center
    eval_centered = x_eval - center
    train_residual = train_centered - (train_centered @ basis.T) @ basis
    eval_residual = eval_centered - (eval_centered @ basis.T) @ basis

    residual_knn = ctx.h.knn(train_residual, eval_residual, k=10, l2=True)

    train_norm = np.linalg.norm(train_residual, axis=1)
    norm_scale = max(float(np.median(train_norm)), 1e-8)
    eval_norm = np.linalg.norm(eval_residual, axis=1)

    scores = residual_knn + 0.15 * np.log1p(eval_norm / norm_scale)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
