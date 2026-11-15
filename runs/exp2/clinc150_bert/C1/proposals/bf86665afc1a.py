import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several nearby normalized encoder-neighbor distances is less sensitive to isolated training examples than a single k-th-neighbor radius. OOS queries should lie farther from the local in-scope manifold in pooled representation space."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    # Exact blockwise cosine-distance kNN avoids materializing an eval-by-train
    # matrix while using local density information from multiple neighbor ranks.
    n_train = x_train.shape[0]
    ks = np.array([5, 15, 40], dtype=int)
    k_max = int(min(ks[-1], n_train))
    ks = np.minimum(ks, k_max)

    scores = np.empty(x_eval.shape[0], dtype=float)
    block = 256
    for start in range(0, x_eval.shape[0], block):
        stop = min(start + block, x_eval.shape[0])
        sim = x_eval[start:stop] @ x_train.T
        dist = 1.0 - sim
        nearest = np.partition(dist, k_max - 1, axis=1)[:, :k_max]
        nearest.sort(axis=1)
        local = nearest[:, ks - 1]
        scores[start:stop] = np.mean(local, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=2.0, neginf=0.0)
