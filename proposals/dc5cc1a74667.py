import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "After removing the subspace spanned by class-mean differences, in-scope examples should retain "
    "locally supported residual structure. OOS inputs can have residual semantic variation that is unlike "
    "the within-intent residuals seen during training."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0158", "m_0183"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels)

    center = np.mean(x_train, axis=0)
    classes = np.unique(labels)
    means = np.vstack([np.mean(x_train[labels == c], axis=0) for c in classes])
    between = means - np.mean(means, axis=0, keepdims=True)

    _, s, vt = np.linalg.svd(between, full_matrices=False)
    tol = np.finfo(float).eps * max(between.shape) * (s[0] if s.size else 1.0)
    rank = int(np.sum(s > tol))
    if rank > 0:
        basis = vt[:rank].T
        train_centered = x_train - center
        eval_centered = x_eval - center
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = x_train - center
        eval_residual = x_eval - center

    scale = np.sqrt(np.mean(train_residual * train_residual, axis=0))
    scale = np.maximum(scale, 1e-8)
    train_residual = train_residual / scale
    eval_residual = eval_residual / scale

    k = min(20, max(1, len(train_residual)))
    return ctx.h.knn(train_residual, eval_residual, k=k, l2=True)
