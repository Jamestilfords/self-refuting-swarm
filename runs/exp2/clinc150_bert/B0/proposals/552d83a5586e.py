import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM-style residual energy measures feature variation outside the dominant in-scope representation subspace, while the logit partition term preserves classifier confidence. OOS queries are expected to have larger residuals relative to their logit evidence."
SOURCE = "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html"
PARENTS = []
MEMORY_USED = ["m_0065", "m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_comp = min(64, tr.pooled.shape[1] - 1, tr.pooled.shape[0] - 1)
    n_comp = max(1, n_comp)

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    train_proj = pca.transform(tr.pooled)
    eval_proj = pca.transform(ev.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_
    train_resid = np.sqrt(np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    ))
    eval_resid = np.sqrt(np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    ))

    train_lse = logsumexp(tr.logits, axis=1)
    alpha = np.mean(train_lse) / max(np.mean(train_resid), 1e-8)
    return alpha * eval_resid - logsumexp(ev.logits, axis=1)
