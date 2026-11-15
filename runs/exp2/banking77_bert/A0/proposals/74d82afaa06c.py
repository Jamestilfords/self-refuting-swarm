import numpy as np
from sklearn.neighbors import LocalOutlierFactor

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Local Outlier Factor compares each query's neighborhood density with the densities of its "
    "nearest training neighbors, identifying queries that fall in sparse or locally inconsistent regions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = ctx.h.pca(tr.pooled, dim=64)
    train_x = pca.transform(tr.pooled)
    eval_x = pca.transform(ev.pooled)

    train_x = ctx.h.l2(train_x)
    eval_x = ctx.h.l2(eval_x)

    lof = LocalOutlierFactor(
        n_neighbors=35,
        algorithm="auto",
        metric="euclidean",
        novelty=True,
        contamination="auto",
    )
    lof.fit(train_x)

    scores = -lof.score_samples(eval_x)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
