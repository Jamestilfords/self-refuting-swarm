import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions span the representation subspace most directly associated with intent discrimination. "
    "OOS queries should have unusually large train-calibrated residual energy outside this supervised subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)

    mean = x_train.mean(axis=0)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    class_sums = np.zeros((classes.size, x_train.shape[1]), dtype=np.float64)
    np.add.at(class_sums, inverse, centered_train)
    class_means = class_sums / np.maximum(counts[:, None], 1.0)

    _, singular_values, vt = np.linalg.svd(class_means, full_matrices=False)
    if singular_values.size:
        threshold = singular_values[0] * max(class_means.shape) * np.finfo(np.float64).eps
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0

    if rank > 0:
        basis = vt[:rank]
        train_projection = centered_train @ basis.T
        eval_projection = centered_eval @ basis.T
        train_residual = np.sum(centered_train * centered_train, axis=1) - np.sum(
            train_projection * train_projection, axis=1
        )
        eval_residual = np.sum(centered_eval * centered_eval, axis=1) - np.sum(
            eval_projection * eval_projection, axis=1
        )
    else:
        train_residual = np.sum(centered_train * centered_train, axis=1)
        eval_residual = np.sum(centered_eval * centered_eval, axis=1)

    train_residual = np.maximum(train_residual, 0.0)
    eval_residual = np.maximum(eval_residual, 0.0)

    train_log_residual = np.log1p(train_residual)
    eval_log_residual = np.log1p(eval_residual)
    median = np.median(train_log_residual)
    q75, q25 = np.percentile(train_log_residual, [75.0, 25.0])
    scale = max(q75 - q25, 1e-8)

    return np.clip((eval_log_residual - median) / scale, -1e6, 1e6)
