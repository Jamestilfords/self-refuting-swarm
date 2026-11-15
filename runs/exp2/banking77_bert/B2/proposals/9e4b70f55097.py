import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Intent-class mean differences define directions used to separate known classes, so OOS inputs "
    "should retain unusually large residual content outside this discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0056", "m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels)

    center = x.mean(axis=0)
    classes = np.unique(y)
    means = np.empty((classes.size, x.shape[1]), dtype=float)
    for i, c in enumerate(classes):
        means[i] = x[y == c].mean(axis=0) - center

    _, singular, vt = np.linalg.svd(means, full_matrices=False)
    if singular.size:
        cutoff = singular[0] * 1e-8
        rank = max(1, int(np.sum(singular > cutoff)))
        basis = vt[:rank].T
        tr_centered = x - center
        ev_centered = np.asarray(ev.feats, dtype=float) - center
        tr_residual = tr_centered - (tr_centered @ basis) @ basis.T
        ev_residual = ev_centered - (ev_centered @ basis) @ basis.T
    else:
        tr_residual = x - center
        ev_residual = np.asarray(ev.feats, dtype=float) - center

    train_energy = np.sum(tr_residual * tr_residual, axis=1)
    scale = np.median(train_energy) + 1e-12
    return np.sum(ev_residual * ev_residual, axis=1) / scale
