import numpy as np
from sklearn.decomposition import PCA

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "The norm outside the principal in-scope encoder subspace measures representation "
    "variation not explained by dominant training-feature structure, which should increase for OOS queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    max_components = min(x_train.shape[0], x_train.shape[1])
    n_components = max(1, min(256, max_components))

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=1729,
        power_iteration_normalizer="QR",
    )
    pca.fit(x_train)

    centered = x_eval - pca.mean_
    projected = centered @ pca.components_.T
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    train_centered = x_train - pca.mean_
    train_projected = train_centered @ pca.components_.T
    train_residual_sq = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_projected * train_projected, axis=1),
        0.0,
    )
    scale = np.median(np.sqrt(train_residual_sq)) + 1e-12
    return np.sqrt(residual_sq) / scale
