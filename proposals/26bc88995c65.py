import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A high-order norm of the component outside the train-fitted principal encoder subspace emphasizes sparse unsupported feature deviations that can be diluted in the usual residual L2 norm."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0027", "m_0034"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(192, tr.pooled.shape[1] - 1, tr.pooled.shape[0] - 1)
    n_comp = max(1, n_comp)
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=173)
    pca.fit(tr.pooled)

    z = pca.transform(ev.pooled)
    recon = z @ pca.components_ + pca.mean_
    residual = ev.pooled - recon

    # Fourth-order residual norm gives extra weight to isolated unsupported
    # encoder coordinates while remaining a genuine residual-space statistic.
    score_values = np.sum(residual * residual * residual * residual, axis=1)
    score_values = np.sqrt(np.maximum(score_values, 0.0))
    return np.nan_to_num(score_values, nan=0.0, posinf=1e30, neginf=0.0)
