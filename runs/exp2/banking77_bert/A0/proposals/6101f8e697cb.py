import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "The span of class means captures directions used to distinguish known intents. "
    "OOS queries can retain energy in the orthogonal remainder that is not explained by this discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    center = x.mean(axis=0)
    classes = np.unique(y)
    means = np.vstack([x[y == c].mean(axis=0) for c in classes]) - center

    _, singular, vt = np.linalg.svd(means, full_matrices=False)
    if singular.size:
        tol = singular[0] * max(means.shape) * np.finfo(float).eps
        rank = max(1, int(np.sum(singular > tol)))
        basis = vt[:rank]
        train_proj = (x - center) @ basis.T
        eval_proj = (z - center) @ basis.T
        train_resid = (x - center) - train_proj @ basis
        eval_resid = (z - center) - eval_proj @ basis
    else:
        train_resid = x - center
        eval_resid = z - center

    train_norm2 = np.sum(train_resid * train_resid, axis=1)
    eval_norm2 = np.sum(eval_resid * eval_resid, axis=1)
    scale = np.median(train_norm2) + 1e-12
    return np.log1p(eval_norm2 / scale)
