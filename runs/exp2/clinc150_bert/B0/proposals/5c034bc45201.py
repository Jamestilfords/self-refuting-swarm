import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture feature variation used to distinguish known intents, while the orthogonal complement captures residual variation unexplained by intent identity. "
    "A class-conditional diagonal residual model flags examples whose remainder is atypical for the classifier's predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0103"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    global_mean = x_train.mean(axis=0)
    global_scale = x_train.std(axis=0)
    global_scale = np.maximum(global_scale, 1e-6)

    z_train = (x_train - global_mean) / global_scale
    z_eval = (x_eval - global_mean) / global_scale

    classes, inverse = np.unique(labels, return_inverse=True)
    class_means = np.zeros((classes.size, z_train.shape[1]), dtype=float)
    np.add.at(class_means, inverse, z_train)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    class_means /= np.maximum(counts[:, None], 1.0)
    class_means -= class_means.mean(axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(class_means, full_matrices=True)
    if singular_values.size:
        tol = max(class_means.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.sum(singular_values > tol))
    else:
        rank = 0

    residual_basis = vt[rank:]
    if residual_basis.shape[0] == 0:
        residual_basis = vt[-1:]

    r_train = z_train @ residual_basis.T
    r_eval = z_eval @ residual_basis.T

    global_var = np.var(r_train, axis=0) + 1e-5
    class_var = np.tile(global_var, (n_classes, 1))
    label_counts = np.bincount(labels, minlength=n_classes).astype(float)

    sums = np.zeros((n_classes, r_train.shape[1]), dtype=float)
    sums_sq = np.zeros((n_classes, r_train.shape[1]), dtype=float)
    np.add.at(sums, labels, r_train)
    np.add.at(sums_sq, labels, r_train * r_train)

    valid = label_counts > 1
    raw_var = np.zeros_like(class_var)
    raw_var[valid] = (
        sums_sq[valid] / label_counts[valid, None]
        - (sums[valid] / label_counts[valid, None]) ** 2
    )
    shrink = label_counts / (label_counts + 20.0)
    class_var = shrink[:, None] * np.maximum(raw_var, 0.0) + (
        1.0 - shrink[:, None]
    ) * global_var[None, :]
    class_var = np.maximum(class_var, 1e-5)

    train_pred = np.argmax(tr.logits, axis=1)
    train_quad = np.mean(
        (r_train * r_train) / class_var[train_pred],
        axis=1,
    )

    sorted_scores = []
    for c in range(n_classes):
        values = np.sort(train_quad[train_pred == c])
        if values.size < 8:
            values = np.sort(train_quad)
        sorted_scores.append(values)

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_quad = np.mean(
        (r_eval * r_eval) / class_var[eval_pred],
        axis=1,
    )

    out = np.empty(eval_quad.shape[0], dtype=float)
    for c in range(n_classes):
        mask = eval_pred == c
        if np.any(mask):
            reference = sorted_scores[c]
            out[mask] = np.searchsorted(
                reference, eval_quad[mask], side="right"
            ) / float(reference.size)

    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
