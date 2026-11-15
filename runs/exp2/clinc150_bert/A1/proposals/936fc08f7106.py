import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope encoder representations should be well explained by a principal subspace learned from training features. "
    "A large fraction of energy in the orthogonal residual space indicates semantic content outside the learned intent manifold."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0091"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(
        n_components=min(192, tr.pooled.shape[1], tr.pooled.shape[0] - 1),
        svd_solver="randomized",
        random_state=2718,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = pca.transform(ev.pooled)
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    train_centered = tr.pooled - pca.mean_
    train_total_sq = np.sum(train_centered * train_centered, axis=1)
    scale = np.median(train_total_sq) + 1e-12

    return residual_sq / (total_sq + 0.05 * scale + 1e-12)
