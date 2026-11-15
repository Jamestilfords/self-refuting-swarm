import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "Rather than using raw residual magnitude, this measures the fraction of a query's centered representation that lies outside the dominant in-scope principal subspace. This angular residual is less sensitive to feature norm and highlights directions poorly explained by the training manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(
        n_components=96,
        svd_solver="randomized",
        random_state=2718,
        iterated_power=4,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = pca.transform(ev.pooled)

    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    angular_residual = residual_sq / np.maximum(total_sq, 1e-12)
    return np.nan_to_num(
        angular_residual,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
