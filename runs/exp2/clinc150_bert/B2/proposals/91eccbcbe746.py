import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions capture feature variation that is useful for separating known intents. Energy remaining orthogonal to this train-fitted discriminative subspace highlights queries whose representation changes in directions not explained by intent separation."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.maximum(scale, 1e-6)

    classes = np.unique(labels)
    centroids = np.vstack([
        np.mean((x_train[labels == c] - mean) / scale, axis=0)
        for c in classes
    ])
    centroids = centroids - np.mean(centroids, axis=0, keepdims=True)

    _, singular, vh = np.linalg.svd(centroids, full_matrices=False)
    total = float(np.sum(singular * singular))
    if total > 1e-12:
        cumulative = np.cumsum(singular * singular)
        q = int(np.searchsorted(cumulative, 0.95 * total) + 1)
        q = min(q, 64, vh.shape[0])
        basis = vh[:q]
    else:
        q = 0
        basis = vh[:0]

    z = (x_eval - mean) / scale
    total_energy = np.sum(z * z, axis=1)
    explained = np.sum((z @ basis.T) ** 2, axis=1) if q else 0.0
    residual_energy = np.maximum(total_energy - explained, 0.0)
    return np.sqrt(residual_energy / max(z.shape[1] - q, 1))
