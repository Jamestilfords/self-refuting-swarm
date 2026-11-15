import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query should have close local support within the intent predicted by the classifier, not merely somewhere in the global training set. "
    "Averaging several class-conditional normalized-neighbor radii reduces sensitivity to a single accidental close example while exposing unsupported predicted-intent assignments."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0060"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_x = np.asarray(tr.pooled, dtype=np.float64)
    eval_x = np.asarray(ev.pooled, dtype=np.float64)
    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), eps)
    eval_x = eval_x / np.maximum(np.linalg.norm(eval_x, axis=1, keepdims=True), eps)

    pred = np.argmax(ev.logits, axis=1)
    labels = np.asarray(tr.labels)
    scores = np.empty(len(eval_x), dtype=np.float64)

    for c in np.unique(pred):
        query_idx = np.flatnonzero(pred == c)
        class_x = train_x[labels == c]

        if len(class_x) == 0:
            scores[query_idx] = 2.0
            continue

        k = min(12, len(class_x))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(class_x)
        distances = nn.kneighbors(eval_x[query_idx], return_distance=True)[0]

        start = min(3, k - 1)
        local_radii = distances[:, start:]
        weights = np.linspace(1.0, 2.0, local_radii.shape[1])
        scores[query_idx] = (local_radii * weights).sum(axis=1) / weights.sum()

    return np.nan_to_num(scores, nan=2.0, posinf=2.0, neginf=0.0)
