import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class means define directions used to separate known intents; OOS queries often retain unusually large residual components outside this discriminative subspace. A variance-normalized residual score measures novelty in the class-mean complement."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels)

    classes, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv, minlength=len(classes)).astype(float)
    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inv, x)
    means /= counts[:, None]

    center = means.mean(axis=0)
    between = means - center
    _, s, vt = np.linalg.svd(between, full_matrices=False)

    if s.size:
        tol = max(between.shape) * np.finfo(float).eps * s[0]
        rank = int(np.sum(s > tol))
    else:
        rank = 0

    if rank == 0:
        train_residual = x - x.mean(axis=0)
        eval_residual = z - x.mean(axis=0)
    else:
        basis = vt[:rank]
        train_centered = x - center
        eval_centered = z - center
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis

    residual_var = np.mean(train_residual * train_residual, axis=0)
    floor = max(float(np.mean(residual_var)) * 1e-6, 1e-12)
    residual_var = np.maximum(residual_var, floor)

    squared = np.sum((eval_residual * eval_residual) / residual_var, axis=1)
    return np.log1p(squared / residual_var.size)
