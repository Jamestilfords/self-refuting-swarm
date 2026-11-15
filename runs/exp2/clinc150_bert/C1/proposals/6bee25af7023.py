import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean subspace isolates feature variation not explained by discriminative intent "
    "directions. OOS queries should be less locally supported by training examples in this residual space."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    overall_mean = np.mean(x_train, axis=0)
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)

    class_means = np.zeros((classes.size, x_train.shape[1]), dtype=float)
    np.add.at(class_means, inverse, x_train)
    class_means /= np.maximum(counts[:, None], 1.0)

    mean_offsets = class_means - overall_mean
    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)

    if singular_values.size:
        tol = max(mean_offsets.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.sum(singular_values > tol))
    else:
        rank = 0

    train_centered = x_train - overall_mean
    eval_centered = x_eval - overall_mean

    if rank > 0:
        basis = vt[:rank]
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    residual_scale = np.std(train_residual, axis=0)
    residual_scale = np.maximum(residual_scale, 1e-6)
    train_residual = train_residual / residual_scale
    eval_residual = eval_residual / residual_scale

    return ctx.h.knn(train_residual, eval_residual, k=20, l2=False)
