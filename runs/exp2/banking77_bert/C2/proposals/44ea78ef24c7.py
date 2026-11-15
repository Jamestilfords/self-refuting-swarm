import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The mean distance to several nearby normalized encoder representations measures local in-distribution support more robustly than a single-neighbor radius. OOS queries should have consistently larger neighborhood distances."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)

    # Fixed projection preserves neighborhood structure while keeping brute-force
    # distance computation tractable for large training sets.
    rng = np.random.default_rng(1729)
    proj = rng.standard_normal((train_x.shape[1], 96))
    proj /= np.sqrt(96.0)
    train_z = train_x @ proj
    eval_z = eval_x @ proj

    # Distances are computed in deterministic batches without materializing
    # an eval-by-train tensor.
    k = min(25, train_z.shape[0])
    out = np.empty(eval_z.shape[0], dtype=float)
    train_norm = np.sum(train_z * train_z, axis=1)

    batch = 128
    for start in range(0, eval_z.shape[0], batch):
        stop = min(start + batch, eval_z.shape[0])
        z = eval_z[start:stop]
        d2 = np.sum(z * z, axis=1, keepdims=True) + train_norm[None, :] - 2.0 * (z @ train_z.T)
        d2 = np.maximum(d2, 0.0)
        nearest = np.partition(d2, k - 1, axis=1)[:, :k]
        out[start:stop] = np.mean(np.sqrt(nearest), axis=1)

    return out
