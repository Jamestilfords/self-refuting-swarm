import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Classifier-discriminative class-mean directions explain variation that separates known intents, "
    "so local support in their orthogonal residual space detects semantic variation not represented by any intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0103", "m_0114"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels)

    center = x_train.mean(axis=0)
    n_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    means = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(means, labels, x_train)
    means /= np.maximum(counts[:, None], 1.0)

    weighted_means = (means - center) * np.sqrt(np.maximum(counts[:, None], 1.0))
    _, singular_values, vt = np.linalg.svd(weighted_means, full_matrices=False)
    tol = max(weighted_means.shape) * np.finfo(float).eps * (
        singular_values[0] if singular_values.size else 1.0
    )
    rank = int(np.sum(singular_values > tol))
    rank = min(rank, 64)

    train_centered = x_train - center
    eval_centered = x_eval - center
    if rank > 0:
        basis = vt[:rank]
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    local_residual_distance = ctx.h.knn(
        train_residual, eval_residual, k=20, l2=False
    )
    residual_norm = np.sqrt(np.sum(eval_residual * eval_residual, axis=1))
    scores = np.asarray(local_residual_distance, dtype=float) + 1e-6 * residual_norm
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
