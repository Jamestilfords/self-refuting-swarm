import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Average local support in the fine-tuned bottleneck space measures whether an input lies near a sustained neighborhood of known intent examples. L2 normalization emphasizes learned representation direction and reduces sensitivity to embedding magnitude."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0071"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.feats)
    eval_x = ctx.h.l2(ev.feats)
    k = min(20, train_x.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(train_x)
    distances, _ = nn.kneighbors(eval_x, return_distance=True)
    scores = np.mean(distances, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
