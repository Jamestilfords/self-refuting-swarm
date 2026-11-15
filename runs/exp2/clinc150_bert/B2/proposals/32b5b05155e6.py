import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Intent-discriminative class-mean directions capture variation useful for separating known intents, while residual energy measures variation unsupported by those directions. OOS queries should retain larger residual components after projection onto the train-derived class-mean subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    classes, inv = np.unique(y, return_inverse=True)
    n_classes = classes.size
    dim = x.shape[1]

    counts = np.bincount(inv, minlength=n_classes).astype(np.float64)
    means = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(means, inv, x)
    means /= np.maximum(counts[:, None], 1.0)

    center = x.mean(axis=0)
    mean_offsets = means - center
    _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)

    if singular.size == 0 or singular[0] <= 0.0:
        residual = np.asarray(ev.pooled, dtype=np.float64) - center
        return np.sum(residual * residual, axis=1)

    rank = int(np.sum(singular > singular[0] * 1e-8))
    rank = max(1, min(rank, vt.shape[0]))
    basis = vt[:rank].T

    train_centered = x - center
    train_proj = train_centered @ basis
    train_residual = train_centered - train_proj @ basis.T
    train_energy = np.sum(train_residual * train_residual, axis=1)
    scale = np.median(train_energy) + 1e-12

    eval_centered = np.asarray(ev.pooled, dtype=np.float64) - center
    eval_proj = eval_centered @ basis
    eval_residual = eval_centered - eval_proj @ basis.T
    return np.sum(eval_residual * eval_residual, axis=1) / scale
