import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM-style residual magnitude measures how far an encoder representation lies outside the principal subspace occupied by training intents. OOS queries often retain a larger component in these discarded directions than in-scope queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    max_rank = min(tr.pooled.shape[0], tr.pooled.shape[1])
    n_components = min(128, max(1, max_rank - 1))

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=271828,
        iterated_power=3,
    )
    pca.fit(tr.pooled)

    z = pca.transform(ev.pooled)
    reconstruction = z @ pca.components_ + pca.mean_
    residual = ev.pooled - reconstruction
    return np.sum(residual * residual, axis=1)
