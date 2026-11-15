import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The mean distance to a moderately sized neighborhood in normalized encoder space measures local support rather than relying on a single nearest example. OOS queries should have consistently more distant in-scope neighbors."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = np.asarray(tr.pooled, dtype=np.float64)
    eval_x = np.asarray(ev.pooled, dtype=np.float64)

    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    eval_x = eval_x / np.maximum(np.linalg.norm(eval_x, axis=1, keepdims=True), 1e-12)

    k = min(24, len(train_x))
    nn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(train_x)
    distances, _ = nn.kneighbors(eval_x, return_distance=True)
    return np.mean(distances, axis=1)
