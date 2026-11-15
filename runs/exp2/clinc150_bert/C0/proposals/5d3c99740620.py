import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A normalized residual-subspace score measures the fraction of each representation's centered energy lying outside the principal in-scope subspace. Unlike raw residual norm, this angular residual is less sensitive to overall feature magnitude and emphasizes representations pointing away from the training manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_components = min(96, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=2718,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    total_sq = np.sum(centered * centered, axis=1)
    projected = centered @ pca.components_.T
    kept_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - kept_sq, 0.0)

    score_values = residual_sq / np.maximum(total_sq, 1e-12)
    return np.nan_to_num(score_values, nan=0.0, posinf=1.0, neginf=0.0)
