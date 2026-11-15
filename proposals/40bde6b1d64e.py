import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A train-fitted principal subspace captures shared in-scope encoder variation, while the residual L1 magnitude "
    "emphasizes broad unsupported deviations outside that subspace without relying on classifier confidence."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0147", "m_0042"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d = tr.pooled.shape[1]
    n_comp = max(1, min(128, tr.pooled.shape[0], d))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = pca.inverse_transform(pca.transform(ev.pooled)) - pca.mean_
    residual = centered - projected

    train_centered = tr.pooled - pca.mean_
    train_projected = pca.inverse_transform(pca.transform(tr.pooled)) - pca.mean_
    train_residual_l1 = np.mean(np.abs(train_centered - train_projected), axis=1)
    scale = np.median(train_residual_l1) + 1e-12

    return np.mean(np.abs(residual), axis=1) / scale
