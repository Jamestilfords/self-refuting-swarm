import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "Out-of-scope queries often retain logit confidence while lying outside the principal feature subspace occupied by training intents. The residual norm after projecting onto that subspace measures this representation-space novelty."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d = tr.pooled.shape[1]
    n_comp = min(128, tr.pooled.shape[0], d)

    if n_comp >= d:
        n_comp = max(1, d - 1)

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = pca.inverse_transform(pca.transform(ev.pooled)) - pca.mean_
    residual = centered - projected
    residual_norm = np.linalg.norm(residual, axis=1)

    train_centered = tr.pooled - pca.mean_
    train_projected = pca.inverse_transform(pca.transform(tr.pooled)) - pca.mean_
    train_residual = np.linalg.norm(train_centered - train_projected, axis=1)
    scale = np.percentile(train_residual, 90)
    scale = max(float(scale), 1e-8)

    return np.nan_to_num(residual_norm / scale, nan=0.0, posinf=1e12, neginf=0.0)
