import numpy as np
from sklearn.decomposition import PCA

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Out-of-scope queries can retain moderate classifier confidence while lying outside the "
    "principal subspace occupied by training encoder representations. The residual norm measures "
    "this unsupported representation component using a subspace fitted only on training data."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    # Retain the dominant in-scope representation manifold while reserving a
    # substantial orthogonal complement for residual/OOS evidence.
    n_components = min(256, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=1729,
        iterated_power=4,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = centered @ pca.components_.T
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    return np.log1p(residual_sq)
