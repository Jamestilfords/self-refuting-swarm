import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query assigned to an intent should have several nearby training paraphrases within that "
    "predicted intent in encoder space; OOS queries may be confidently classified but remain locally isolated."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007", "m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    x_train = normalize(tr.pooled)
    x_eval = normalize(ev.pooled)
    labels = np.asarray(tr.labels, dtype=int)
    predicted = np.argmax(np.asarray(ev.logits), axis=1)

    global_k = min(8, len(x_train))
    global_nn = NearestNeighbors(
        n_neighbors=max(1, global_k), metric="cosine", algorithm="brute"
    ).fit(x_train)
    scores = global_nn.kneighbors(x_eval, return_distance=True)[0].mean(axis=1)

    for cls in np.unique(predicted):
        query_idx = np.flatnonzero(predicted == cls)
        train_idx = np.flatnonzero(labels == cls)
        if len(train_idx) == 0:
            continue
        k = min(8, len(train_idx))
        nn = NearestNeighbors(
            n_neighbors=max(1, k), metric="cosine", algorithm="brute"
        ).fit(x_train[train_idx])
        distances = nn.kneighbors(x_eval[query_idx], return_distance=True)[0]
        scores[query_idx] = distances.mean(axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=2.0, neginf=0.0)
