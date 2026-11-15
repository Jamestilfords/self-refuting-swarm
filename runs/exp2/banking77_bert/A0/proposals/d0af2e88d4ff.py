import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions primarily encode intent-discriminative variation, so removing them isolates "
    "within-intent residual structure. OOS queries may project plausibly onto known intent directions but lack "
    "local support in the residual manifold occupied by in-scope examples."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0107"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    center = x_train.mean(axis=0)
    classes, inverse = np.unique(labels, return_inverse=True)
    sums = np.zeros((classes.size, x_train.shape[1]), dtype=float)
    np.add.at(sums, inverse, x_train)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    class_means = sums / np.maximum(counts[:, None], 1.0)

    between = class_means - center
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    tol = np.finfo(float).eps * max(between.shape) * singular[0] if singular.size else 0.0
    rank = int(np.sum(singular > tol))

    train_centered = x_train - center
    eval_centered = x_eval - center
    if rank:
        basis = vt[:rank]
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    return ctx.h.knn(train_residual, eval_residual, k=10, l2=True)
