import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A locally normalized kNN score compares a query's distance to nearby training examples against the neighborhood radii of those examples. This flags points that are isolated relative to the local in-scope manifold, rather than merely far from a globally dense region."
SOURCE = "https://icml.cc/media/icml-2022/Slides/16494.pdf"
PARENTS = []
MEMORY_USED = ["m_0056", "m_0123"]


def score(ctx):
    x_train = np.asarray(ctx.train.pooled, dtype=np.float64)
    x_eval = np.asarray(ctx.eval.pooled, dtype=np.float64)
    n_train = x_train.shape[0]

    if n_train < 3:
        center = np.mean(x_train, axis=0, keepdims=True)
        return np.sqrt(np.sum((x_eval - center) ** 2, axis=1))

    k = min(12, n_train - 1)
    nn_train = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_train.fit(x_train)
    train_dist, _ = nn_train.kneighbors(x_train)
    train_radius = train_dist[:, k]

    nn_eval = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn_eval.fit(x_train)
    eval_dist, eval_idx = nn_eval.kneighbors(x_eval)

    neighbor_radius = train_radius[eval_idx]
    reachability = np.maximum(eval_dist, neighbor_radius)
    local_scale = np.mean(neighbor_radius, axis=1)
    scores = np.mean(reachability, axis=1) / (local_scale + 1e-12)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
