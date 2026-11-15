import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture supervised variation that separates known intents. "
    "OOS queries should retain a larger fraction of their feature energy outside this discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0023"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    center = x_train.mean(axis=0)
    classes = np.unique(labels)
    means = np.vstack([x_train[labels == c].mean(axis=0) for c in classes])
    mean_directions = means - center

    _, singular_values, vt = np.linalg.svd(mean_directions, full_matrices=False)
    tol = max(mean_directions.shape) * np.finfo(np.float64).eps
    keep = singular_values > tol * max(singular_values[0], 1.0)
    basis = vt[keep].T

    def residual_fraction(x):
        centered = x - center
        total = np.sum(centered * centered, axis=1)
        if basis.shape[1] == 0:
            residual = total
        else:
            projected = centered @ basis
            residual = total - np.sum(projected * projected, axis=1)
        return np.maximum(residual, 0.0) / (total + 1e-12)

    train_ratio = residual_fraction(x_train)
    eval_ratio = residual_fraction(x_eval)

    train_median = np.median(train_ratio)
    train_mad = np.median(np.abs(train_ratio - train_median))
    scale = max(1.4826 * train_mad, 1e-8)
    return np.clip((eval_ratio - train_median) / scale, -1e6, 1e6)
