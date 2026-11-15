import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOD inputs may retain some intent-related components while having unusually large representation energy outside the principal subspace occupied by training queries. The residual norm measures this unsupported feature component without requiring class-wise density estimates."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(128, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=max(1, dim), svd_solver="randomized", random_state=314159)
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = centered @ pca.components_.T
    total_sq = np.sum(centered * centered, axis=1)
    kept_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - kept_sq, 0.0)
    return np.sqrt(residual_sq + 1e-12)
