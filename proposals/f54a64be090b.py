import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "The fraction of centered encoder energy lying outside a train-fitted principal subspace measures unsupported representation content while discounting ordinary feature-norm variation. OOS queries should retain a larger residual-energy share than in-scope encoder features."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0188", "m_0197", "m_0147"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(
        n_components=96,
        svd_solver="randomized",
        random_state=271828,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    total_sq = np.sum(centered * centered, axis=1)

    coords = pca.transform(ev.pooled)
    explained_sq = np.sum(coords * coords, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    eps = np.finfo(np.float64).eps
    return residual_sq / (total_sq + eps)
