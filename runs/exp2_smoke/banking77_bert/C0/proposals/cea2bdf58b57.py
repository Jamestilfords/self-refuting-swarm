import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging distances to several nearby normalized encoder embeddings is less sensitive to isolated training points than a single k-th-neighbor distance. OOS queries should have consistently larger local neighborhood distances."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    # Chunked exact cosine-distance kNN, avoiding a large eval-by-train matrix.
    k_values = np.array([5, 10, 20, 40], dtype=int)
    k_max = int(k_values[-1])
    n_train = x_train.shape[0]
    k_max = min(k_max, n_train)

    out = np.empty(x_eval.shape[0], dtype=float)
    chunk = 256
    for start in range(0, x_eval.shape[0], chunk):
        stop = min(start + chunk, x_eval.shape[0])
        sim = x_eval[start:stop] @ x_train.T
        dist = 1.0 - sim
        nearest = np.partition(dist, k_max - 1, axis=1)[:, :k_max]
        nearest.sort(axis=1)
        valid_k = np.minimum(k_values, k_max)
        local = np.mean(nearest[:, valid_k - 1], axis=1)
        out[start:stop] = local

    return out
