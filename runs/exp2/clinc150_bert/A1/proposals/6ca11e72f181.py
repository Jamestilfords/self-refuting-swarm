import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A predicted intent should have substantially denser same-intent support than the "
    "unrestricted in-scope neighborhood. OOS queries often have nearby training examples "
    "only across mixed intents, producing a large class-conditional-to-global distance ratio."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    x_train = normalize(tr.pooled)
    x_eval = normalize(ev.pooled)
    pred = np.argmax(ev.logits, axis=1)
    k_global = min(8, len(x_train))

    global_nn = NearestNeighbors(n_neighbors=k_global, metric="euclidean")
    global_nn.fit(x_train)
    global_dist = global_nn.kneighbors(x_eval, return_distance=True)[0][:, -1]

    own_dist = np.empty(len(x_eval), dtype=float)
    for cls in np.unique(pred):
        query_idx = np.flatnonzero(pred == cls)
        class_x = x_train[tr.labels == cls]
        if len(class_x) == 0:
            own_dist[query_idx] = global_dist[query_idx]
            continue

        k_class = min(16, len(class_x))
        class_nn = NearestNeighbors(n_neighbors=k_class, metric="euclidean")
        class_nn.fit(class_x)
        own_dist[query_idx] = class_nn.kneighbors(
            x_eval[query_idx], return_distance=True
        )[0][:, -1]

    score = own_dist / np.maximum(global_dist, 1e-8)
    return np.nan_to_num(score, nan=1.0, posinf=1e6, neginf=0.0)
