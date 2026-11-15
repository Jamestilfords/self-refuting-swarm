import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging distances to multiple nearby normalized pooled embeddings estimates local in-scope support more smoothly than a single-neighbor radius. OOS queries should have consistently more distant local neighborhoods even when one accidental neighbor is close."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0111"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    k = min(24, x_train.shape[0])

    train_sq = np.sum(x_train * x_train, axis=1)
    scores = np.empty(x_eval.shape[0], dtype=float)
    chunk = 256

    for start in range(0, x_eval.shape[0], chunk):
        stop = min(start + chunk, x_eval.shape[0])
        q = x_eval[start:stop]
        d2 = (
            np.sum(q * q, axis=1, keepdims=True)
            + train_sq[None, :]
            - 2.0 * q @ x_train.T
        )
        d2 = np.maximum(d2, 0.0)
        nearest = np.partition(d2, k - 1, axis=1)[:, :k]
        scores[start:stop] = np.mean(np.sqrt(nearest), axis=1)

    return scores
