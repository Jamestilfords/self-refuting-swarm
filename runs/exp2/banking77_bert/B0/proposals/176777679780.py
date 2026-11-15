import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope encoder representations should concentrate near a principal training-feature subspace. "
    "A large norm in its orthogonal residual space indicates feature content unsupported by known intents."
)
SOURCE = "https://deel-ai.github.io/oodeel/latest/notebooks/torch/demo_vim_torch/"
PARENTS = []
MEMORY_USED = ["m_0068"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(128, tr.pooled.shape[1] - 1, tr.pooled.shape[0] - 1)
    pca = PCA(n_components=max(1, dim), svd_solver="randomized", random_state=173)
    pca.fit(tr.pooled)

    centered = ev.pooled - pca.mean_
    projected = pca.transform(ev.pooled) @ pca.components_
    residual = centered - projected
    return np.sqrt(np.maximum(np.sum(residual * residual, axis=1), 0.0))
