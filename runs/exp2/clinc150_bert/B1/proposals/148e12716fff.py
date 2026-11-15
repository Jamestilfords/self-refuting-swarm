import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class means define directions used to discriminate the known intents, while the orthogonal "
    "residual captures semantic content not explained by that intent subspace. OOS queries should "
    "therefore have unusually large residual energy relative to training examples."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0014"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)
    z = np.asarray(ev.pooled, dtype=np.float64)

    center = x.mean(axis=0)
    classes = np.unique(y)
    means = np.vstack([x[y == c].mean(axis=0) for c in classes]) - center

    _, s, vt = np.linalg.svd(means, full_matrices=False)
    tol = np.finfo(np.float64).eps * max(means.shape) * (s[0] if s.size else 1.0)
    rank = int(np.sum(s > tol))
    rank = max(1, min(rank, vt.shape[0]))
    basis = vt[:rank].T

    train_centered = x - center
    eval_centered = z - center
    train_residual = train_centered - (train_centered @ basis) @ basis.T
    eval_residual = eval_centered - (eval_centered @ basis) @ basis.T

    train_energy = np.sum(train_residual * train_residual, axis=1)
    eval_energy = np.sum(eval_residual * eval_residual, axis=1)

    med = np.median(train_energy)
    mad = np.median(np.abs(train_energy - med))
    scale = max(1.4826 * mad, np.finfo(np.float64).eps)
    return np.nan_to_num((eval_energy - med) / scale, nan=0.0, posinf=1e12, neginf=-1e12)
