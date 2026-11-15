import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "Out-of-scope queries should retain more encoder-feature energy outside the principal subspace learned from in-scope training features. This residual norm captures deviations that may not be reflected in classifier logits."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(
        n_components=128,
        svd_solver="randomized",
        random_state=1729,
        whiten=False,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = centered @ pca.components_.T
    total_sq = np.sum(centered * centered, axis=1)
    projected_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - projected_sq, 0.0)

    return np.sqrt(residual_sq + 1e-12)
