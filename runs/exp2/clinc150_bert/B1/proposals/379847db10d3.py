import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query that is close to the overall training manifold but far from neighbours assigned "
    "to its predicted intent is likely misrouted or semantically out of scope. Train-only "
    "empirical tail calibration makes class-specific kNN distances comparable across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k_target = 20

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)

    x_train = normalize(np.asarray(tr.pooled, dtype=float))
    x_eval = normalize(np.asarray(ev.pooled, dtype=float))
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    n_train = len(x_train)
    if n_train < 2:
        return np.linalg.norm(x_eval - x_train.mean(axis=0, keepdims=True), axis=1)

    global_k = min(k_target, n_train - 1)
    global_nn = NearestNeighbors(
        n_neighbors=global_k + 1, metric="euclidean", algorithm="auto"
    ).fit(x_train)
    global_ref = global_nn.kneighbors(x_train, return_distance=True)[0][:, -1]
    global_dist = global_nn.kneighbors(
        x_eval, n_neighbors=global_k, return_distance=True
    )[0][:, -1]

    global_sorted = np.sort(global_ref)

    def tail_score(dist, reference):
        rank = np.searchsorted(reference, dist, side="right")
        tail = (len(reference) - rank + 1.0) / (len(reference) + 1.0)
        return -np.log(np.maximum(tail, 1.0 / (len(reference) + 1.0)))

    global_tail = tail_score(global_dist, global_sorted)
    result = global_tail.copy()

    for cls in np.unique(eval_pred):
        eval_mask = eval_pred == cls
        train_mask = train_pred == cls
        class_x = x_train[train_mask]
        n_class = len(class_x)

        if n_class < 2:
            continue

        class_k = min(k_target, n_class - 1)
        class_nn = NearestNeighbors(
            n_neighbors=class_k + 1, metric="euclidean", algorithm="auto"
        ).fit(class_x)
        class_ref = class_nn.kneighbors(class_x, return_distance=True)[0][:, -1]
        class_dist = class_nn.kneighbors(
            x_eval[eval_mask], n_neighbors=class_k, return_distance=True
        )[0][:, -1]

        class_tail = tail_score(class_dist, np.sort(class_ref))
        result[eval_mask] = np.maximum(global_tail[eval_mask], class_tail)

    scale = np.median(global_ref) + eps
    result += 0.03 * np.log1p(global_dist / scale)
    return np.nan_to_num(result, nan=0.0, posinf=50.0, neginf=0.0)
