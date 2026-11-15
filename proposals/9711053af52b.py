import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "The normalized softmax vector encodes the full pattern of competition among intents, not just the winning confidence. "
    "A multiscale average of nearest in-scope predictive shapes flags queries whose classifier response is locally unsupported."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_p = ctx.h.l2(ctx.h.softmax(tr.logits))
    eval_p = ctx.h.l2(ctx.h.softmax(ev.logits))

    n_train = train_p.shape[0]
    k_max = min(48, n_train)
    nn = NearestNeighbors(n_neighbors=k_max, metric="euclidean", algorithm="auto")
    nn.fit(train_p)
    distances = nn.kneighbors(eval_p, return_distance=True)[0]

    radii = np.array(
        [min(4, k_max), min(16, k_max), k_max],
        dtype=int,
    ) - 1
    return np.mean(distances[:, radii], axis=1)
