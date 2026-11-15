import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "The span of class means contains directions that are explicitly discriminative for in-scope intents. "
    "OOS inputs should retain unusually large, variance-normalized residual energy outside this supervised subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels)

    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    means = np.zeros((classes.size, x.shape[1]), dtype=float)
    np.add.at(means, inverse, x)
    means /= counts[:, None]

    center = np.mean(x, axis=0)
    mean_offsets = means - center
    _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    tol = np.finfo(float).eps * max(mean_offsets.shape) * singular[0]
    rank = max(1, int(np.sum(singular > tol)))
    basis = vt[:rank]

    train_centered = x - center
    train_residual = train_centered - (train_centered @ basis.T) @ basis
    residual_var = np.mean(train_residual * train_residual, axis=0)
    residual_var = np.maximum(residual_var, 1e-8)

    eval_centered = z - center
    eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    return np.sum((eval_residual * eval_residual) / residual_var, axis=1)
