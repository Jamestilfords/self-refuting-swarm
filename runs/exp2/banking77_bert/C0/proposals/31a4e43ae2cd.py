import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope encoder representations concentrate near a principal subspace learned from training data, whereas OOS queries tend to retain larger orthogonal residual components. Using the residual alone avoids inheriting the ranking of energy-based baselines."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    max_dim = min(96, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(
        n_components=max_dim,
        svd_solver="randomized",
        random_state=271,
        whiten=False,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = centered @ pca.components_.T
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    return np.nan_to_num(
        np.sqrt(residual_sq),
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )
