import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "The span of class means captures directions most associated with intent discrimination. OOS examples should retain unusually large energy in the orthogonal residual subspace that is not explained by in-scope class separation."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    classes, inv = np.unique(labels, return_inverse=True)
    counts = np.bincount(inv, minlength=len(classes)).astype(float)
    means = np.zeros((len(classes), x_train.shape[1]), dtype=float)
    np.add.at(means, inv, x_train)
    means /= counts[:, None]

    global_mean = np.mean(x_train, axis=0)
    centered_means = means - global_mean

    _, singular_values, vt = np.linalg.svd(centered_means, full_matrices=False)
    tol = max(centered_means.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tol * max(singular_values[0], 1.0)))

    train_centered = x_train - global_mean
    eval_centered = x_eval - global_mean

    if rank > 0:
        basis = vt[:rank].T
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    train_energy = np.sum(train_residual * train_residual, axis=1)
    eval_energy = np.sum(eval_residual * eval_residual, axis=1)

    median = np.median(train_energy)
    mad = np.median(np.abs(train_energy - median))
    scale = max(1.4826 * mad, np.std(train_energy), 1e-8)

    return (eval_energy - median) / scale
