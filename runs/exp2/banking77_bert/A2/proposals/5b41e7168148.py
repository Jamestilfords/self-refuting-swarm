import numpy as np
from sklearn.neighbors import LocalOutlierFactor

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Local outlier factor compares a query's neighborhood density with the densities of its nearby "
    "in-scope neighbors, detecting OOS queries that lie in sparse or locally inconsistent regions. "
    "PCA-compressed, L2-normalized pooled embeddings emphasize semantic direction while reducing noise."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = ctx.h.pca(tr.pooled, 64)
    train_x = pca.transform(tr.pooled)
    eval_x = pca.transform(ev.pooled)

    train_x = ctx.h.l2(train_x)
    eval_x = ctx.h.l2(eval_x)

    n_train = train_x.shape[0]
    if n_train < 3:
        center = np.mean(train_x, axis=0, keepdims=True)
        scores = np.sum((eval_x - center) ** 2, axis=1)
        return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)

    k = min(35, n_train - 1)
    lof = LocalOutlierFactor(
        n_neighbors=k,
        metric="euclidean",
        novelty=True,
        contamination="auto",
        algorithm="brute",
        n_jobs=-1,
    )
    lof.fit(train_x)
    scores = -lof.score_samples(eval_x)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
