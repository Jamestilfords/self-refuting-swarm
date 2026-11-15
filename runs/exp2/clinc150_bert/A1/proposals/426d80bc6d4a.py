import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Distance to nearby training examples within the classifier's predicted intent measures whether a query is locally compatible with that intent, rather than merely close to any in-scope class."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    x_train = normalize(tr.pooled)
    x_eval = normalize(ev.pooled)
    predicted = np.argmax(ev.logits, axis=1)
    scores = np.empty(x_eval.shape[0], dtype=float)

    global_nn = NearestNeighbors(
        n_neighbors=min(15, len(x_train)),
        metric="euclidean",
        algorithm="auto",
    ).fit(x_train)

    for cls in np.unique(predicted):
        idx_eval = np.where(predicted == cls)[0]
        idx_train = np.where(tr.labels == cls)[0]

        if len(idx_train) == 0:
            distances = global_nn.kneighbors(
                x_eval[idx_eval], return_distance=True
            )[0]
        else:
            nn = NearestNeighbors(
                n_neighbors=min(15, len(idx_train)),
                metric="euclidean",
                algorithm="auto",
            ).fit(x_train[idx_train])
            distances = nn.kneighbors(x_eval[idx_eval], return_distance=True)[0]

        scores[idx_eval] = distances.mean(axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=2.0, neginf=0.0)
