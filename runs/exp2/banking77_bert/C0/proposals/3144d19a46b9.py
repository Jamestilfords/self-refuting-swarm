import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The mean distance to several nearby normalized encoder embeddings measures local support rather than relying on a single neighbor. OOS queries should lie farther from the in-scope embedding manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    k = min(25, len(x_train))
    if k < 1:
        return np.arange(len(x_eval), dtype=float)
    chunk = 512
    out = np.empty(len(x_eval), dtype=float)
    train_norm = np.sum(x_train * x_train, axis=1)
    for start in range(0, len(x_eval), chunk):
        stop = min(start + chunk, len(x_eval))
        q = x_eval[start:stop]
        dist2 = (
            np.sum(q * q, axis=1, keepdims=True)
            + train_norm[None, :]
            - 2.0 * q @ x_train.T
        )
        np.maximum(dist2, 0.0, out=dist2)
        nearest = np.partition(dist2, k - 1, axis=1)[:, :k]
        out[start:stop] = np.mean(np.sqrt(nearest), axis=1)
    return out
