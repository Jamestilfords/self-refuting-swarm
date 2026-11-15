import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Intent-class mean differences define supervised discriminative directions, while the orthogonal remainder captures semantic variation not explained by any in-scope intent. Class-conditional robust calibration highlights queries with unusually large residual energy for their predicted intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0082"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]

    global_mean = x_train.mean(axis=0)
    class_means = np.empty((n_classes, x_train.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        members = x_train[y == c]
        class_means[c] = members.mean(axis=0) if len(members) else global_mean

    mean_center = class_means.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(
        class_means - mean_center, full_matrices=False
    )
    if singular_values.size and singular_values[0] > 0:
        rank = int(np.sum(singular_values > singular_values[0] * 1e-8))
        basis = vt[:max(rank, 1)].T
        train_proj = (x_train - global_mean) @ basis
        eval_proj = (x_eval - global_mean) @ basis
        train_residual = x_train - global_mean - train_proj @ basis.T
        eval_residual = x_eval - global_mean - eval_proj @ basis.T
    else:
        train_residual = x_train - global_mean
        eval_residual = x_eval - global_mean

    train_energy = np.sum(train_residual * train_residual, axis=1)
    eval_energy = np.sum(eval_residual * eval_residual, axis=1)

    centers = np.empty(n_classes, dtype=np.float64)
    scales = np.empty(n_classes, dtype=np.float64)
    global_center = np.median(train_energy)
    global_scale = np.median(np.abs(train_energy - global_center)) + 1e-8
    for c in range(n_classes):
        values = train_energy[y == c]
        if len(values):
            center = np.median(values)
            scale = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = scale if scale > 1e-8 else global_scale
        else:
            centers[c] = global_center
            scales[c] = global_scale

    predicted = np.argmax(ev.logits, axis=1)
    scores = (eval_energy - centers[predicted]) / scales[predicted]
    return np.clip(scores, -1e6, 1e6)
