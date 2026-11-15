import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "An OOS query may be globally similar to the training corpus while remaining atypical "
    "among examples of the class predicted by the classifier. Combining predicted-class and "
    "global cosine kNN distances measures both local intent support and broad in-domain coverage."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    n_train = train_x.shape[0]

    global_k = min(32, n_train)
    global_nn = NearestNeighbors(
        n_neighbors=global_k,
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    )
    global_nn.fit(train_x)
    global_dist = global_nn.kneighbors(
        eval_x, return_distance=True
    )[0][:, -1]

    pred = np.argmax(ev.logits, axis=1)
    class_dist = global_dist.copy()

    for cls in np.unique(pred):
        train_idx = np.flatnonzero(tr.labels == cls)
        eval_idx = np.flatnonzero(pred == cls)

        if train_idx.size == 0 or eval_idx.size == 0:
            continue

        local_k = min(16, train_idx.size)
        local_nn = NearestNeighbors(
            n_neighbors=local_k,
            metric="cosine",
            algorithm="brute",
            n_jobs=1,
        )
        local_nn.fit(train_x[train_idx])
        class_dist[eval_idx] = local_nn.kneighbors(
            eval_x[eval_idx], return_distance=True
        )[0][:, -1]

    scores = 0.7 * class_dist + 0.3 * global_dist
    return np.nan_to_num(scores, nan=0.0, posinf=2.0, neginf=0.0)
