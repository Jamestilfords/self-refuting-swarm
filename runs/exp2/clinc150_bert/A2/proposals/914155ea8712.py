import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Averaging an annulus of normalized bottleneck-space neighbour distances measures sustained "
    "local support rather than relying on one accidentally close training example. OOS queries "
    "should have consistently weaker semantic neighbourhood support across these scales."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(np.asarray(tr.feats, dtype=float))
    eval_x = ctx.h.l2(np.asarray(ev.feats, dtype=float))

    n_train = train_x.shape[0]
    k_hi = min(32, n_train)
    k_lo = min(8, k_hi)

    if k_hi < 2:
        return np.sqrt(np.sum((eval_x - train_x[0]) ** 2, axis=1))

    nn = NearestNeighbors(
        n_neighbors=k_hi,
        algorithm="brute",
        metric="euclidean",
    )
    nn.fit(train_x)
    distances = nn.kneighbors(eval_x, return_distance=True)[0]

    annulus = distances[:, k_lo - 1:k_hi]
    return np.mean(annulus, axis=1)
