import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging distances to several nearby normalized encoder embeddings reduces sensitivity to isolated training points while measuring local in-distribution support."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    # Mean distance to the 20 nearest training examples, computed in batches
    # to avoid constructing a full evaluation-by-training distance matrix.
    k = min(20, x_train.shape[0])
    batch = 256
    out = np.empty(x_eval.shape[0], dtype=float)

    train_sq = np.sum(x_train * x_train, axis=1)
    for start in range(0, x_eval.shape[0], batch):
        stop = min(start + batch, x_eval.shape[0])
        z = x_eval[start:stop]
        dist_sq = (
            np.sum(z * z, axis=1, keepdims=True)
            + train_sq[None, :]
            - 2.0 * z @ x_train.T
        )
        dist_sq = np.maximum(dist_sq, 0.0)
        nearest = np.partition(dist_sq, k - 1, axis=1)[:, :k]
        out[start:stop] = np.mean(np.sqrt(nearest), axis=1)

    return out
