import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions encode variation that discriminates known intents, while local support "
    "in their orthogonal residual space detects novel variation not explained by intent identity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0131", "m_0133"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)
    classes = np.unique(labels)

    mean = np.mean(x_train, axis=0)
    prototypes = np.vstack([np.mean(x_train[labels == c], axis=0) for c in classes])
    offsets = prototypes - mean

    _, singular_values, vt = np.linalg.svd(offsets, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        residual_train = x_train - mean
        residual_eval = x_eval - mean
    else:
        threshold = singular_values[0] * max(offsets.shape) * np.finfo(np.float64).eps
        rank = max(1, int(np.sum(singular_values > threshold)))
        basis = vt[:rank]

        centered_train = x_train - mean
        centered_eval = x_eval - mean
        residual_train = centered_train - (centered_train @ basis.T) @ basis
        residual_eval = centered_eval - (centered_eval @ basis.T) @ basis

    train_scale = np.sqrt(np.mean(residual_train * residual_train, axis=0))
    train_scale = np.maximum(train_scale, 1e-6)
    residual_train = residual_train / train_scale
    residual_eval = residual_eval / train_scale

    return ctx.h.knn(residual_train, residual_eval, k=20, l2=False)
