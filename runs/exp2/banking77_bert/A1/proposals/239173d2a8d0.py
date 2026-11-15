import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query may receive a confident intent prediction while lacking local support among examples of that predicted intent. Class-conditional cosine neighbourhood distance measures this predicted-intent support rather than allowing nearby examples from competing intents to mask novelty."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = np.asarray(tr.feats, dtype=float)
    eval_x = np.asarray(ev.feats, dtype=float)
    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    eval_x = eval_x / np.maximum(np.linalg.norm(eval_x, axis=1, keepdims=True), 1e-12)

    predicted = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(eval_x), dtype=float)
    labels = np.asarray(tr.labels)

    for cls in np.unique(predicted):
        query_idx = np.flatnonzero(predicted == cls)
        class_x = train_x[labels == cls]
        k = min(12, len(class_x))

        if k == 0:
            scores[query_idx] = 2.0
            continue

        model = NearestNeighbors(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        model.fit(class_x)
        distances, _ = model.kneighbors(eval_x[query_idx])
        scores[query_idx] = np.mean(distances, axis=1)

    return np.nan_to_num(scores, nan=2.0, posinf=2.0, neginf=0.0)
