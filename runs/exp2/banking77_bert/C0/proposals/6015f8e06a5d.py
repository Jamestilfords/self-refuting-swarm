import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Queries assigned to an in-scope class should lie near that class's local encoder manifold, whereas OOS queries often have a larger average distance to neighbours from their predicted class. Averaging several neighbours reduces sensitivity to isolated training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = np.asarray(tr.pooled, dtype=float)
    eval_x = np.asarray(ev.pooled, dtype=float)

    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    eval_x = eval_x / np.maximum(np.linalg.norm(eval_x, axis=1, keepdims=True), 1e-12)

    predicted = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(eval_x), dtype=float)

    for cls in np.unique(predicted):
        query_idx = np.where(predicted == cls)[0]
        class_x = train_x[tr.labels == cls]

        if len(class_x) == 0:
            d = np.linalg.norm(eval_x[query_idx], axis=1)
            scores[query_idx] = d
            continue

        k = min(12, len(class_x))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
        nn.fit(class_x)
        distances, _ = nn.kneighbors(eval_x[query_idx])
        scores[query_idx] = np.mean(distances, axis=1)

    return scores
