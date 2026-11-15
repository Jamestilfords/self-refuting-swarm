import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A global train-fitted principal subspace captures dominant in-scope encoder variation, "
    "while OOS inputs should retain more energy in discarded residual directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(256, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=173)
    pca.fit(tr.pooled)

    center = pca.mean_
    components = pca.components_
    x = ev.pooled - center
    projected = (x @ components.T) @ components
    residual_sq = np.sum((x - projected) ** 2, axis=1)

    train_x = tr.pooled - center
    train_projected = (train_x @ components.T) @ components
    train_residual = np.sum((train_x - train_projected) ** 2, axis=1)
    scale = np.median(train_residual) + 1e-12

    return np.log1p(residual_sq / scale)
