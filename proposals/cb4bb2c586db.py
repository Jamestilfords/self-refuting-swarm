import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "Out-of-scope inputs often retain logit confidence while lying outside the principal subspace occupied by training encoder representations. The residual norm after projecting onto that train-fitted subspace measures this unsupported representation component."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pca = PCA(n_components=64, svd_solver="randomized", random_state=314159)
    pca.fit(tr.pooled)
    centered = ev.pooled - pca.mean_
    projected = pca.transform(ev.pooled)
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    return np.sqrt(np.maximum(total_sq - explained_sq, 0.0))
