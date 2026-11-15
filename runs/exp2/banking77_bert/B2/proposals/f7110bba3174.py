import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions identify the supervised subspace used to distinguish known intents. "
    "OOS queries should retain more energy in the orthogonal residual subspace than in-scope queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    global_mean = x_train.mean(axis=0)
    classes = np.unique(labels)
    class_means = np.vstack([
        x_train[labels == c].mean(axis=0) for c in classes
    ])

    between = class_means - global_mean
    _, singular_values, vt = np.linalg.svd(between, full_matrices=False)

    tol = np.finfo(np.float64).eps * max(between.shape) * (
        singular_values[0] if singular_values.size else 1.0
    )
    rank = int(np.sum(singular_values > tol))
    rank = max(1, min(rank, vt.shape[0]))

    directions = vt[:rank]
    centered_eval = x_eval - global_mean
    projected = centered_eval @ directions.T
    total_sq = np.sum(centered_eval * centered_eval, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    train_centered = x_train - global_mean
    train_projected = train_centered @ directions.T
    train_residual = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_projected * train_projected, axis=1),
        0.0,
    )
    scale = np.median(train_residual) + 1e-12

    return np.nan_to_num(residual_sq / scale, nan=0.0, posinf=1e12, neginf=0.0)
