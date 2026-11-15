import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "Out-of-subspace residual norm measures how much an encoder representation lies outside the principal in-scope feature subspace. OOS queries should require larger residual components than training intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    q = min(128, tr.pooled.shape[0], tr.pooled.shape[1])
    pca = PCA(
        n_components=q,
        svd_solver="randomized",
        random_state=1729,
        iterated_power=4,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = pca.transform(ev.pooled)
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)
    return np.sqrt(residual_sq + 1e-12)
