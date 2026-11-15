import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The mean distance to several nearby normalized encoder embeddings estimates local in-distribution support more robustly than a single neighbor. OOS queries should lie farther from their local training neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    # Compute exact local support in blocks to avoid materializing an
    # eval-by-train distance matrix for all evaluation items at once.
    k = min(25, len(x_train))
    out = np.empty(len(x_eval), dtype=float)
    block = 256

    for start in range(0, len(x_eval), block):
        stop = min(start + block, len(x_eval))
        sim = x_eval[start:stop] @ x_train.T
        dist_sq = np.maximum(0.0, 2.0 - 2.0 * sim)
        nearest = np.partition(dist_sq, k - 1, axis=1)[:, :k]
        out[start:stop] = np.sqrt(nearest).mean(axis=1)

    return out
