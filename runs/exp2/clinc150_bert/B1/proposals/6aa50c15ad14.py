import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean discriminative subspace isolates representation variation not needed to "
    "separate known intents. OOS queries can then be detected by their novelty among training residuals."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0126"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    center = x_train.mean(axis=0)
    classes = np.unique(labels)
    mean_offsets = np.vstack(
        [x_train[labels == c].mean(axis=0) - center for c in classes]
    )

    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    if singular_values.size:
        threshold = max(singular_values[0] * 1e-8, 1e-12)
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0

    train_centered = x_train - center
    eval_centered = x_eval - center

    if rank > 0:
        basis = vt[:rank]
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    residual_knn = ctx.h.knn(train_residual, eval_residual, k=10, l2=False)
    train_radius = np.sqrt(np.sum(train_residual * train_residual, axis=1))
    scale = np.median(train_radius) + 1e-12
    eval_radius = np.sqrt(np.sum(eval_residual * eval_residual, axis=1))

    return np.asarray(residual_knn, dtype=np.float64) + 1e-3 * eval_radius / scale
