import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "ViM-style residual norms measure feature energy outside the principal in-scope representation subspace. "
    "Calibrating that residual as a virtual OOS logit and competing it against classifier logits combines representation support with intent evidence."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0006", "m_0024"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=173)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        projected = centered @ pca.components_.T
        residual_sq = np.sum(centered * centered, axis=1) - np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(residual_sq, 0.0))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)

    logit_scale = np.mean(np.abs(np.max(tr.logits, axis=1)))
    residual_scale = np.mean(train_residual)
    alpha = logit_scale / max(residual_scale, 1e-12)

    virtual_logit = alpha * eval_residual
    all_logits = np.concatenate([ev.logits, virtual_logit[:, None]], axis=1)
    return virtual_logit - logsumexp(all_logits, axis=1)
