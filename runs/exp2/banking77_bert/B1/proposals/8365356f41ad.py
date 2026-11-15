import numpy as np
from sklearn.decomposition import PCA


FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope representations concentrate near a principal subspace of training encoder features, whereas unseen intents often retain a larger orthogonal residual. The residual norm measures this deviation without relying on classifier confidence."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_comp = min(128, tr.pooled.shape[0] - 1, tr.pooled.shape[1])
    n_comp = max(1, n_comp)

    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        random_state=314159,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    total_sq = np.sum(centered * centered, axis=1)
    projection = centered @ pca.components_.T
    explained_sq = np.sum(projection * projection, axis=1)

    return np.sqrt(np.maximum(total_sq - explained_sq, 0.0))
