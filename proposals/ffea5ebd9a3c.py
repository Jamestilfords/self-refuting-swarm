import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "After removing the class-mean discriminative subspace, supported in-scope queries "
    "should retain locally familiar residual encoder content. A residual-space kNN radius "
    "detects unsupported semantic directions beyond classifier-discriminative variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0067", "m_0095", "m_0132"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    mean = x_train.mean(axis=0)
    classes = np.unique(labels)
    class_means = np.stack(
        [x_train[labels == c].mean(axis=0) - mean for c in classes],
        axis=0,
    )

    _, singular_values, vt = np.linalg.svd(class_means, full_matrices=False)
    keep = singular_values > (singular_values[0] * 1e-8 if singular_values.size else 0.0)
    basis = vt[keep].T

    train_centered = x_train - mean
    eval_centered = x_eval - mean
    if basis.shape[1]:
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    scale = train_residual.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    train_residual = train_residual / scale
    eval_residual = eval_residual / scale

    return ctx.h.knn(train_residual, eval_residual, k=10, l2=True)
