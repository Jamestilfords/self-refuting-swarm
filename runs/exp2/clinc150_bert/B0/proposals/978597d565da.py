import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-conditional local support in standardized bottleneck space measures whether an item resembles "
    "nearby examples of the intent predicted by the classifier, rather than merely being close to any intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0036"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    pred = np.asarray(ev.logits).argmax(axis=1)

    global_mean = x_train.mean(axis=0)
    global_scale = x_train.std(axis=0)
    global_scale = np.maximum(global_scale, 1e-6)
    global_train = (x_train - global_mean) / global_scale
    global_eval = (x_eval - global_mean) / global_scale

    global_k = min(12, len(global_train))
    global_nn = NearestNeighbors(
        n_neighbors=global_k, metric="euclidean", algorithm="auto"
    ).fit(global_train)
    scores = global_nn.kneighbors(global_eval, return_distance=True)[0][:, -1]

    for c in np.unique(pred):
        query_idx = np.flatnonzero(pred == c)
        train_idx = np.flatnonzero(labels == c)
        if len(train_idx) < 2:
            continue

        class_train = x_train[train_idx]
        class_mean = class_train.mean(axis=0)
        class_scale = np.maximum(class_train.std(axis=0), 1e-6)
        standardized_train = (class_train - class_mean) / class_scale
        standardized_eval = (x_eval[query_idx] - class_mean) / class_scale

        k = min(8, len(train_idx))
        nn = NearestNeighbors(
            n_neighbors=k, metric="euclidean", algorithm="auto"
        ).fit(standardized_train)
        scores[query_idx] = nn.kneighbors(
            standardized_eval, return_distance=True
        )[0][:, -1]

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
