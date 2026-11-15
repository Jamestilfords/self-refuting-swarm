import numpy as np
from sklearn.decomposition import PCA

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "After removing the principal in-scope encoder subspace, OOS queries should have both "
    "large residual content and residual directions unsupported by nearby training examples."
)
SOURCE = "https://deel-ai.github.io/oodeel/latest/notebooks/tensorflow/demo_vim_tf/"
PARENTS = []
MEMORY_USED = ["m_0045", "m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d = tr.pooled.shape[1]
    n_components = max(1, min(128, d, tr.pooled.shape[0]))

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=1729,
    ).fit(tr.pooled)

    def residual(x):
        centered = x - pca.mean_
        return centered - (centered @ pca.components_.T) @ pca.components_

    r_train = residual(tr.pooled)
    r_eval = residual(ev.pooled)

    train_norm = np.linalg.norm(r_train, axis=1)
    scale = np.median(train_norm)
    scale = max(float(scale), 1e-8)

    residual_norm = np.linalg.norm(r_eval, axis=1)
    residual_knn = ctx.h.knn(r_train, r_eval, k=10, l2=True)

    scores = np.log1p(residual_norm / scale) + np.log1p(
        np.maximum(residual_knn, 0.0) / scale
    )
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
