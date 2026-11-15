import numpy as np
from sklearn.decomposition import PCA

FAMILY = "density"
ARM = "ce"
RATIONALE = "Out-of-scope inputs tend to retain substantial feature energy outside the principal subspace occupied by in-scope encoder representations. The residual norm measures this unsupported variation without requiring eval-set calibration."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_components = min(128, tr.pooled.shape[0], tr.pooled.shape[1])
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=314159,
    )
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = pca.transform(ev.pooled)
    total_sq = np.sum(centered * centered, axis=1)
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)
    return np.sqrt(residual_sq)
