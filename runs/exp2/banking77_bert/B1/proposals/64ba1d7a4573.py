import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Averaging several nearby L2-normalized pooled-representation distances estimates local "
    "in-scope support more robustly than a single boundary neighbor. OOS queries should lack "
    "a compact semantic neighborhood among known-intent training examples."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    x_train = x_train / np.maximum(train_norm, 1e-12)
    x_eval = x_eval / np.maximum(eval_norm, 1e-12)

    k = min(24, len(x_train))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    nn.fit(x_train)
    distances, _ = nn.kneighbors(x_eval, return_distance=True)

    weights = np.linspace(1.5, 0.5, k, dtype=np.float64)
    weights /= weights.sum()
    return distances @ weights
