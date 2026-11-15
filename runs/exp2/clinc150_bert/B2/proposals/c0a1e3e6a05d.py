import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM measures representation energy outside the principal in-distribution feature subspace, where OOS queries should have larger residuals. Combining this residual with logit normalization creates a virtual OOD logit without fitting on evaluation data."
SOURCE = "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html"
PARENTS = []
MEMORY_USED = ["m_0063", "m_0065"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(256, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=17)
    pca.fit(tr.pooled)

    tr_centered = tr.pooled - pca.mean_
    ev_centered = ev.pooled - pca.mean_
    components = pca.components_

    tr_proj = tr_centered @ components.T
    ev_proj = ev_centered @ components.T
    tr_residual_sq = np.maximum(
        np.sum(tr_centered * tr_centered, axis=1) - np.sum(tr_proj * tr_proj, axis=1),
        0.0,
    )
    ev_residual_sq = np.maximum(
        np.sum(ev_centered * ev_centered, axis=1) - np.sum(ev_proj * ev_proj, axis=1),
        0.0,
    )
    tr_residual = np.sqrt(tr_residual_sq)
    ev_residual = np.sqrt(ev_residual_sq)

    denom = max(float(np.mean(tr_residual)), 1e-12)
    alpha = float(np.mean(np.max(tr.logits, axis=1))) / denom
    scores = alpha * ev_residual - logsumexp(ev.logits, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
