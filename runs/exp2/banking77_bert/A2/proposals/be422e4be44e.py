import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style residual norm measures feature mass outside the dominant in-scope encoder subspace. OOS inputs can remain logit-confident while violating the low-dimensional structure shared by training queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(192, tr.pooled.shape[1] - 1, tr.pooled.shape[0] - 1)
    pca = PCA(n_components=max(1, dim), svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    z = pca.transform(ev.pooled)
    recon = pca.inverse_transform(z)
    residual = ev.pooled - recon
    scores = np.sum(residual * residual, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
