import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query should have several nearby examples from the intent it is predicted to belong to, "
    "not merely one accidental neighbor. Prediction-conditioned average k-nearest-neighbor distance "
    "captures weak local in-scope support while avoiding cross-intent neighbors."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = ctx.h.l2(tr.feats)
    eval_x = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(ev.logits, axis=1)

    global_k = min(12, len(train_x))
    global_nn = NearestNeighbors(
        n_neighbors=global_k,
        metric="euclidean",
        algorithm="brute",
        n_jobs=1,
    ).fit(train_x)
    fallback = global_nn.kneighbors(eval_x, return_distance=True)[0].mean(axis=1)

    out = fallback.copy()
    for cls in np.unique(pred):
        eval_idx = np.flatnonzero(pred == cls)
        train_idx = np.flatnonzero(labels == cls)
        if train_idx.size == 0:
            continue

        k = min(12, train_idx.size)
        nn = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean",
            algorithm="brute",
            n_jobs=1,
        ).fit(train_x[train_idx])
        distances = nn.kneighbors(eval_x[eval_idx], return_distance=True)[0]
        out[eval_idx] = distances.mean(axis=1)

    return np.asarray(out, dtype=float)
