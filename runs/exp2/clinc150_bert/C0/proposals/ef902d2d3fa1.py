import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances capture anisotropic intent manifolds while shrinkage toward the global variance prevents unstable estimates for small classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)

    classes = np.unique(labels)
    n_classes = len(classes)
    dim = x_train.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    scale = max(float(np.median(global_var)), 1e-8)
    global_var = np.maximum(global_var, scale * 1e-4)

    means = np.empty((n_classes, dim), dtype=np.float64)
    variances = np.empty((n_classes, dim), dtype=np.float64)

    for i, cls in enumerate(classes):
        members = x_train[labels == cls]
        count = members.shape[0]
        mean = np.mean(members, axis=0)
        if count > 1:
            class_var = np.mean((members - mean) ** 2, axis=0)
        else:
            class_var = global_var.copy()

        shrink = count / (count + 32.0)
        means[i] = mean
        variances[i] = np.maximum(
            shrink * class_var + (1.0 - shrink) * global_var,
            scale * 1e-4,
        )

    inv_var = 1.0 / variances
    weighted_means = means * inv_var
    offsets = np.sum(means * weighted_means, axis=1) + np.sum(
        np.log(variances), axis=1
    )

    scores = np.empty(x_eval.shape[0], dtype=np.float64)
    block_size = 2048

    for start in range(0, x_eval.shape[0], block_size):
        stop = min(start + block_size, x_eval.shape[0])
        block = x_eval[start:stop]
        quadratic = (
            (block * block) @ inv_var.T
            - 2.0 * (block @ weighted_means.T)
            + offsets
        )
        scores[start:stop] = 0.5 * np.min(quadratic, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
