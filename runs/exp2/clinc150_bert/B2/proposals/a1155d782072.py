import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "The subspace spanned by class means contains intent-discriminative variation, so "
    "nearest-neighbor support in its orthogonal residual isolates whether within-intent detail is in-distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0122"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    means = np.zeros((len(classes), x_train.shape[1]), dtype=np.float64)
    np.add.at(means, inverse, x_train)
    means /= counts[:, None]

    global_mean = x_train.mean(axis=0, keepdims=True)
    mean_offsets = means - global_mean
    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)

    if singular_values.size:
        tol = singular_values[0] * max(mean_offsets.shape) * np.finfo(np.float64).eps
        rank = int(np.sum(singular_values > tol))
    else:
        rank = 0

    centered_train = x_train - global_mean
    centered_eval = x_eval - global_mean
    if rank > 0:
        basis = vt[:rank].T
        residual_train = centered_train - (centered_train @ basis) @ basis.T
        residual_eval = centered_eval - (centered_eval @ basis) @ basis.T
    else:
        residual_train = centered_train
        residual_eval = centered_eval

    scale = np.sqrt(np.mean(residual_train * residual_train, axis=0))
    scale = np.maximum(scale, 1e-8)
    residual_train /= scale
    residual_eval /= scale

    k = min(10, len(residual_train))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(residual_train)
    distances, _ = nn.kneighbors(residual_eval, return_distance=True)
    return np.maximum(distances[:, -1], 0.0)
