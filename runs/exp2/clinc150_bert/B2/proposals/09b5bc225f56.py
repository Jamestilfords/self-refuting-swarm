import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean discriminative subspace leaves nuisance and semantic residual directions "
    "that should retain local support for in-scope queries but be poorly supported for OOS queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0076", "m_0080"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    global_mean = x_train.mean(axis=0)
    classes = np.unique(labels)
    class_means = np.vstack([
        x_train[labels == c].mean(axis=0) - global_mean for c in classes
    ])

    _, singular_values, vt = np.linalg.svd(class_means, full_matrices=False)
    if singular_values.size:
        tol = singular_values[0] * 1e-10
        rank = max(1, int(np.sum(singular_values > tol)))
        basis = vt[:rank]
        train_centered = x_train - global_mean
        eval_centered = x_eval - global_mean
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = x_train - global_mean
        eval_residual = x_eval - global_mean

    distances = ctx.h.knn(train_residual, eval_residual, k=20, l2=True)
    radial = np.linalg.norm(eval_residual, axis=1)
    radial_center = np.median(np.linalg.norm(train_residual, axis=1))
    scores = np.asarray(distances, dtype=float) + 1e-6 * np.abs(radial - radial_center)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
