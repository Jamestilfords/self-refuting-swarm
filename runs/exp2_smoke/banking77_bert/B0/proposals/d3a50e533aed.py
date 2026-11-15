import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query that is assigned to an intent class but remains far from that class's normalized encoder examples has weak class-conditional local support, which is characteristic of OOS inputs."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = np.asarray(tr.pooled, dtype=float)
    eval_x = np.asarray(ev.pooled, dtype=float)

    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    eval_x = eval_x / np.maximum(np.linalg.norm(eval_x, axis=1, keepdims=True), 1e-12)

    predicted = np.argmax(ev.logits, axis=1)
    labels = np.asarray(tr.labels)
    scores = np.empty(eval_x.shape[0], dtype=float)

    global_k = min(12, train_x.shape[0])
    global_nn = NearestNeighbors(n_neighbors=global_k, metric="euclidean")
    global_nn.fit(train_x)

    for cls in np.unique(predicted):
        eval_mask = predicted == cls
        class_x = train_x[labels == cls]

        if class_x.shape[0] == 0:
            distances = global_nn.kneighbors(eval_x[eval_mask], return_distance=True)[0]
        else:
            k = min(12, class_x.shape[0])
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
            nn.fit(class_x)
            distances = nn.kneighbors(eval_x[eval_mask], return_distance=True)[0]

        scores[eval_mask] = np.mean(distances, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
