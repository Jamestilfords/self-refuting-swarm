import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging distances to several nearby normalized encoder embeddings is less sensitive to isolated training exemplars than a single-neighbor score. OOS queries should have consistently larger local angular distances to the in-scope training manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    k = min(25, len(train_x))
    if k < 1:
        return np.arange(len(eval_x), dtype=float) * 1e-12

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(train_x)
        distances, _ = nn.kneighbors(eval_x, return_distance=True)
        weights = np.linspace(1.0, 2.0, k)
        scores = (distances * weights[None, :]).sum(axis=1) / weights.sum()
    except Exception:
        scores = ctx.h.knn(train_x, eval_x, k=k, l2=False)

    return np.asarray(scores, dtype=float)
