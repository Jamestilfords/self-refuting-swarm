import numpy as np
from sklearn.decomposition import PCA

FAMILY = "knn"
ARM = "ce"
RATIONALE = "ViM-style residual features remove dominant in-scope representation directions; OOS examples should have unusual residual directions as well as larger residual magnitude. A residual-space neighbor score captures local novelty that a global residual norm alone misses."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(n_components=96, svd_solver="randomized", random_state=41)
    pca.fit(tr.pooled)

    tr_centered = tr.pooled - pca.mean_
    ev_centered = ev.pooled - pca.mean_

    tr_residual = tr_centered - (tr_centered @ pca.components_.T) @ pca.components_
    ev_residual = ev_centered - (ev_centered @ pca.components_.T) @ pca.components_

    tr_norm = np.sqrt(np.maximum(np.sum(tr_residual * tr_residual, axis=1), 0.0))
    ev_norm = np.sqrt(np.maximum(np.sum(ev_residual * ev_residual, axis=1), 0.0))

    tr_knn = ctx.h.knn(tr_residual, tr_residual, k=15, l2=True)
    ev_knn = ctx.h.knn(tr_residual, ev_residual, k=15, l2=True)

    norm_scale = max(float(np.median(tr_norm)), 1e-8)
    knn_scale = max(float(np.median(tr_knn)), 1e-8)

    scores = np.log1p(ev_norm / norm_scale) + ev_knn / knn_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
