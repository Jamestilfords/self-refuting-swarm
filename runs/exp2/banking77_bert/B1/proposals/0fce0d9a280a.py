import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Known-intent bottleneck features should concentrate in a principal in-scope subspace, "
    "whereas OOS queries retain larger residual energy outside that subspace."
)
SOURCE = "https://github.com/haoqiwang/vim/"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0066"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d = tr.feats.shape[1]
    rank = min(128, d, max(1, tr.feats.shape[0] - 1))

    pca = PCA(
        n_components=rank,
        svd_solver="randomized",
        random_state=2718,
    )
    pca.fit(tr.feats)

    centered = ev.feats - pca.mean_
    coords = pca.transform(ev.feats)
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(coords * coords, axis=1)
    residual = np.sqrt(np.maximum(total_sq - explained_sq, 0.0))

    train_centered = tr.feats - pca.mean_
    train_coords = pca.transform(tr.feats)
    train_residual = np.sqrt(
        np.maximum(
            np.sum(train_centered * train_centered, axis=1)
            - np.sum(train_coords * train_coords, axis=1),
            0.0,
        )
    )
    scale = np.median(train_residual) + 1e-8
    return residual / scale
