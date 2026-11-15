import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "After removing the span of class-mean directions that explain intent-discriminative variation, "
    "in-scope queries should retain locally supported residual semantic directions. OOS queries often "
    "contain residual content not represented by nearby training examples."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0089"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)

    center = x_train.mean(axis=0, keepdims=True)
    classes = np.unique(tr.labels)
    means = np.vstack([
        x_train[tr.labels == c].mean(axis=0) - center[0]
        for c in classes
    ])

    _, singular_values, vt = np.linalg.svd(means, full_matrices=False)
    if singular_values.size:
        tol = np.finfo(float).eps * max(means.shape) * singular_values[0]
        rank = max(1, int(np.sum(singular_values > tol)))
        basis = vt[:rank].T
        train_centered = x_train - center
        eval_centered = x_eval - center
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = x_train - center
        eval_residual = x_eval - center

    k = max(1, min(10, len(train_residual)))
    return ctx.h.knn(train_residual, eval_residual, k=k, l2=True)
