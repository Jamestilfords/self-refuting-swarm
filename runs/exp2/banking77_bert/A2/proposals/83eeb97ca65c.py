import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-conditional local support in normalized bottleneck space detects queries that may lie near the global in-scope manifold but are poorly supported by examples of their predicted intent. Averaging several neighbors is less sensitive to isolated training points than a single-neighbor distance."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(x_eval), dtype=float)

    global_k = min(8, len(x_train))
    global_nn = NearestNeighbors(n_neighbors=global_k, metric="euclidean")
    global_nn.fit(x_train)

    for cls in np.unique(pred):
        query_idx = np.flatnonzero(pred == cls)
        train_idx = np.flatnonzero(tr.labels == cls)
        if len(train_idx) >= 2:
            k = min(8, len(train_idx))
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
            nn.fit(x_train[train_idx])
            distances = nn.kneighbors(x_eval[query_idx], return_distance=True)[0]
        else:
            distances = global_nn.kneighbors(
                x_eval[query_idx], return_distance=True
            )[0]
        scores[query_idx] = np.mean(distances, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
