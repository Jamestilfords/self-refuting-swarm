import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions define the representation subspace most discriminative for known intents. "
    "After assigning an item by its discriminative projection, OOS queries should have unusually large residual energy relative to that predicted class's residual distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = classes.size
    dim = x_train.shape[1]

    global_mean = x_train.mean(axis=0)
    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    class_means = np.zeros((n_classes, dim), dtype=float)
    np.add.at(class_means, inverse, x_train)
    class_means /= np.maximum(counts[:, None], 1.0)

    centered_means = class_means - global_mean
    _, singular_values, vt = np.linalg.svd(centered_means, full_matrices=False)
    if singular_values.size:
        cutoff = max(float(singular_values[0]) * 1e-7, 1e-12)
        rank = int(np.sum(singular_values > cutoff))
        rank = max(1, rank)
        basis = vt[:rank]
    else:
        basis = np.zeros((0, dim), dtype=float)

    train_centered = x_train - global_mean
    eval_centered = x_eval - global_mean

    if basis.shape[0]:
        train_disc = train_centered @ basis.T
        eval_disc = eval_centered @ basis.T
        prototype_disc = centered_means @ basis.T
        dist = (
            np.sum(eval_disc * eval_disc, axis=1, keepdims=True)
            + np.sum(prototype_disc * prototype_disc, axis=1)[None, :]
            - 2.0 * (eval_disc @ prototype_disc.T)
        )
        assigned = np.argmin(dist, axis=1)
        train_residual = train_centered - (train_disc @ basis)
        eval_residual = eval_centered - (eval_disc @ basis)
    else:
        assigned = np.zeros(x_eval.shape[0], dtype=int)
        train_residual = train_centered
        eval_residual = eval_centered

    residual_means = np.zeros((n_classes, dim), dtype=float)
    np.add.at(residual_means, inverse, train_residual)
    residual_means /= np.maximum(counts[:, None], 1.0)

    pooled_var = np.mean(
        (train_residual - residual_means[inverse]) ** 2, axis=0
    )
    pooled_floor = max(float(np.median(pooled_var)) * 1e-6, 1e-10)
    pooled_var = np.maximum(pooled_var, pooled_floor)

    centered_residuals = train_residual - residual_means[inverse]
    class_var = np.zeros((n_classes, dim), dtype=float)
    np.add.at(class_var, inverse, centered_residuals * centered_residuals)
    class_var /= np.maximum(counts[:, None], 1.0)

    shrink = counts / (counts + 20.0)
    class_var = (
        shrink[:, None] * class_var
        + (1.0 - shrink[:, None]) * pooled_var[None, :]
    )
    class_var = np.maximum(class_var, pooled_floor)

    chosen_mean = residual_means[assigned]
    chosen_var = class_var[assigned]
    z = eval_residual - chosen_mean
    scores = np.mean((z * z) / chosen_var, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
