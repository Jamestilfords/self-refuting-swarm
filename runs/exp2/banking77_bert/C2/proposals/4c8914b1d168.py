import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style virtual OOS logit uses the norm outside the in-scope principal feature subspace, while competing it against the classifier's in-scope logits. This changes the residual-only ranking by discounting residuals when the classifier remains strongly confident."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        projected = centered @ pca.components_.T
        residual_sq = np.sum(centered * centered, axis=1)
        residual_sq -= np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(residual_sq, 0.0))

    train_residual = residual_norm(tr.pooled)
    denom = max(float(np.mean(train_residual)), 1e-8)
    train_peak = np.max(tr.logits, axis=1)
    scale = float(np.mean(np.maximum(train_peak, 0.0))) / denom
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0 / denom

    virtual_logit = scale * residual_norm(ev.pooled)
    in_scope_logsumexp = logsumexp(ev.logits, axis=1)
    return virtual_logit - np.logaddexp(virtual_logit, in_scope_logsumexp)
