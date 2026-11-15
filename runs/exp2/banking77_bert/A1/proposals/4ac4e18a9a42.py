import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean discriminative subspace leaves within-intent residual variation, "
    "where in-scope queries should retain local training support and OOS queries should be isolated."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0091", "m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    center = x_train.mean(axis=0)
    classes = np.unique(labels)
    means = np.vstack([x_train[labels == c].mean(axis=0) for c in classes])
    between = means - center

    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size:
        keep = singular > (singular[0] * 1e-8)
    else:
        keep = np.zeros(0, dtype=bool)

    if np.any(keep):
        basis = vt[keep]
        train_centered = x_train - center
        eval_centered = x_eval - center
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = x_train - center
        eval_residual = x_eval - center

    k = min(15, len(train_residual))
    values = ctx.h.knn(train_residual, eval_residual, k=k, l2=True)
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
