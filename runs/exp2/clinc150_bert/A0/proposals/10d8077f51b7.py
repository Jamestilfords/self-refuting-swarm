import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM treats the norm outside the principal in-scope representation subspace as a virtual OOS logit. Queries with substantial residual feature energy receive high virtual-class probability even when an in-scope intent logit is confident."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(96, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=137)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        projected = (centered @ pca.components_.T) @ pca.components_
        return np.sqrt(np.sum((centered - projected) ** 2, axis=1))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)

    mean_residual = max(float(np.mean(train_residual)), 1e-8)
    train_peak = np.max(tr.logits, axis=1)
    logit_scale = float(np.mean(np.maximum(train_peak, 0.0)))
    if logit_scale <= 1e-8:
        logit_scale = float(np.std(tr.logits))
    alpha = max(logit_scale, 1e-6) / mean_residual

    virtual_logit = alpha * eval_residual
    normalizer = logsumexp(ev.logits, axis=1)
    return virtual_logit - np.logaddexp(virtual_logit, normalizer)
