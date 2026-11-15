import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM-style residual norms measure feature variation outside the principal in-scope representation subspace. OOS queries should retain larger off-subspace components even when their nearest known-intent direction appears plausible."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d = tr.pooled.shape[1]
    n_components = min(128, d - 1, tr.pooled.shape[0] - 1)
    n_components = max(1, n_components)

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=2718,
        iterated_power=3,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = centered @ pca.components_.T
    residual_sq = np.sum(centered * centered, axis=1) - np.sum(projected * projected, axis=1)
    return np.sqrt(np.maximum(residual_sq, 0.0))
