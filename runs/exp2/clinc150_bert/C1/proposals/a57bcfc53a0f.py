import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "Queries outside the training feature manifold retain a larger component orthogonal to the principal subspace learned from in-scope encoder representations."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    max_components = min(128, x_train.shape[1], max(1, x_train.shape[0] - 1))
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=1729,
        whiten=False,
    )
    pca.fit(x_train)

    centered = x_eval - pca.mean_
    projected = pca.transform(x_eval)
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    return np.nan_to_num(residual_sq, nan=0.0, posinf=1e12, neginf=0.0)
