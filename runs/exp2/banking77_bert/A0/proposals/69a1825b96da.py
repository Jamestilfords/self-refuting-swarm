import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Average distance to the nearest normalized encoder representations measures local in-scope support more smoothly than a single neighbor boundary. OOS queries should have consistently larger cosine-space distances to their closest known examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    n_train = x_train.shape[0]
    k = min(25, n_train)
    block = 512
    out = np.empty(x_eval.shape[0], dtype=float)

    train_sq = np.sum(x_train * x_train, axis=1)
    for start in range(0, x_eval.shape[0], block):
        stop = min(start + block, x_eval.shape[0])
        xb = x_eval[start:stop]
        dist_sq = (
            np.sum(xb * xb, axis=1, keepdims=True)
            + train_sq[None, :]
            - 2.0 * xb @ x_train.T
        )
        np.maximum(dist_sq, 0.0, out=dist_sq)
        nearest = np.partition(dist_sq, k - 1, axis=1)[:, :k]
        out[start:stop] = np.mean(np.sqrt(nearest), axis=1)

    return out
