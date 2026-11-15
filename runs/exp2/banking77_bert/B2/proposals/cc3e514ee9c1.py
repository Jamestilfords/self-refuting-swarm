import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope encoder representations should lie near a principal subspace learned from training data, while OOS queries often retain a larger residual component outside that subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = centered @ pca.components_.T
    residual = centered - projected @ pca.components_
    scores = np.sum(residual * residual, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e30, neginf=0.0)
