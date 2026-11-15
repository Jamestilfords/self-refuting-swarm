import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class residual directions can expose subtle deviations that broad semantic directions miss. This score measures trailing-eigenspace deviation from the classifier-predicted class, with a small train-calibrated energy term to break ties between similarly shaped residuals."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)
    k = tr.logits.shape[1]

    means = np.zeros((k, x.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = residuals.T @ residuals / max(len(x) - k, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tail_dim = min(64, x.shape[1])
    tail_vals = np.maximum(eigvals[:tail_dim], 1e-7)
    tail_vecs = eigvecs[:, :tail_dim]

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_proj = (x - means[train_pred]) @ tail_vecs
    eval_proj = (z - means[eval_pred]) @ tail_vecs
    train_tail = np.sum((train_proj * train_proj) / tail_vals, axis=1)
    eval_tail = np.sum((eval_proj * eval_proj) / tail_vals, axis=1)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    tail_scale = np.std(train_tail) + 1e-8
    energy_scale = np.std(train_energy) + 1e-8
    return (eval_tail - np.mean(train_tail)) / tail_scale + 0.05 * (
        (eval_energy - np.mean(train_energy)) / energy_scale
    )
