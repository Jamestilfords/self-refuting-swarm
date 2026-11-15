import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOS examples can place a larger fraction of their centered encoder-feature energy outside the dominant in-scope principal subspace. Normalizing residual energy by total feature energy emphasizes directional novelty rather than merely feature magnitude."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(n_components=96, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    total_sq = np.sum(centered * centered, axis=1)

    coords = pca.transform(ev.pooled)
    explained_sq = np.sum(coords * coords, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    scores = residual_sq / np.maximum(total_sq, 1e-12)
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
