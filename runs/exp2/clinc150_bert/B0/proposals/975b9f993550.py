import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query should be locally supported by training examples from its predicted intent; OOS queries can receive a confident class prediction while remaining unusually far from that class's within-intent neighborhood. Per-class neighborhood-radius calibration avoids penalizing naturally broad intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0109", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    y_train = np.asarray(tr.labels)
    pred = np.argmax(ev.logits, axis=1)

    n_eval = x_eval.shape[0]
    out = np.empty(n_eval, dtype=np.float64)

    global_k = min(8, max(1, x_train.shape[0]))
    global_nn = NearestNeighbors(n_neighbors=global_k, metric="euclidean")
    global_nn.fit(x_train)
    global_dist = global_nn.kneighbors(x_eval, return_distance=True)[0][:, -1]

    for c in np.unique(pred):
        idx_eval = np.flatnonzero(pred == c)
        idx_train = np.flatnonzero(y_train == c)

        if idx_train.size < 2:
            out[idx_eval] = global_dist[idx_eval]
            continue

        k = min(8, idx_train.size)
        class_x = x_train[idx_train]
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(class_x)
        test_dist = nn.kneighbors(x_eval[idx_eval], return_distance=True)[0][:, -1]

        train_k = min(k + 1, idx_train.size)
        train_nn = NearestNeighbors(n_neighbors=train_k, metric="euclidean")
        train_nn.fit(class_x)
        train_dist = train_nn.kneighbors(class_x, return_distance=True)[0]
        if train_k > 1:
            radius = train_dist[:, -1]
        else:
            radius = np.full(idx_train.size, 1.0)

        med = np.median(radius)
        q75 = np.percentile(radius, 75)
        q25 = np.percentile(radius, 25)
        scale = max(q75 - q25, 1e-6)
        out[idx_eval] = (test_dist - med) / scale

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
