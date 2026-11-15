import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions capture variation useful for separating known intents, while OOS inputs are expected to retain unusually large energy in the orthogonal residual subspace. The residual energy is calibrated only with training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    mu = x.mean(axis=0)
    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    means = np.zeros((len(classes), x.shape[1]), dtype=float)
    np.add.at(means, inverse, x)
    means /= counts[:, None]

    between = (means - mu) * np.sqrt(counts)[:, None]
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size:
        keep = singular > (singular[0] * 1e-8)
        basis = vt[keep]
    else:
        basis = vt[:0]

    def residual_log_energy(a):
        centered = a - mu
        total = np.sum(centered * centered, axis=1)
        if basis.shape[0]:
            projected = centered @ basis.T
            explained = np.sum(projected * projected, axis=1)
            residual = np.maximum(total - explained, 0.0)
        else:
            residual = np.maximum(total, 0.0)
        return np.log1p(residual)

    train_energy = residual_log_energy(x)
    center = np.median(train_energy)
    scale = np.median(np.abs(train_energy - center))
    scale = max(1.4826 * scale, 1e-6)

    scores = (residual_log_energy(z) - center) / scale
    return np.clip(scores, -1e6, 1e6)
