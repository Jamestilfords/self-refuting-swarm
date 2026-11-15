import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query should be closer to examples that the classifier assigns to the same intent than "
    "to the training set globally. The ratio of prediction-conditioned to global local distance "
    "flags classifier-feature mismatches characteristic of near-OOS inputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005", "m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = ctx.h.l2(tr.feats)
    eval_x = ctx.h.l2(ev.feats)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    k_global = min(25, len(train_x))
    global_nn = NearestNeighbors(n_neighbors=k_global, metric="euclidean")
    global_nn.fit(train_x)
    global_dist = global_nn.kneighbors(eval_x, return_distance=True)[0].mean(axis=1)

    out = np.empty(len(eval_x), dtype=float)
    for cls in np.unique(eval_pred):
        query_idx = np.flatnonzero(eval_pred == cls)
        member_idx = np.flatnonzero(train_pred == cls)

        if len(member_idx) == 0:
            local_dist = global_dist[query_idx]
        else:
            k_local = min(25, len(member_idx))
            local_nn = NearestNeighbors(n_neighbors=k_local, metric="euclidean")
            local_nn.fit(train_x[member_idx])
            local_dist = local_nn.kneighbors(
                eval_x[query_idx], return_distance=True
            )[0].mean(axis=1)

        out[query_idx] = np.log1p(local_dist) - np.log1p(global_dist[query_idx])

    return out
