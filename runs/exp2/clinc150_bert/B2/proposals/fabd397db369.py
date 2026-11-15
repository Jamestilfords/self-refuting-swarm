import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A ViM-style virtual out-of-scope logit is derived from feature energy outside a train-fitted principal subspace. "
    "Its posterior probability rises when residual representation shift is large relative to the classifier's in-scope logits."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_components = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=314159,
    ).fit(tr.pooled)

    def residual_norm(x):
        centered_sq = np.sum((x - pca.mean_) ** 2, axis=1)
        projected_sq = np.sum(pca.transform(x) ** 2, axis=1)
        return np.sqrt(np.maximum(centered_sq - projected_sq, 0.0))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)

    train_peak = np.max(tr.logits, axis=1)
    residual_scale = np.mean(train_residual)
    if not np.isfinite(residual_scale) or residual_scale < 1e-8:
        residual_scale = 1.0

    alpha = np.mean(np.abs(train_peak)) / residual_scale
    if not np.isfinite(alpha) or alpha < 1e-8:
        alpha = 1.0

    virtual_logit = alpha * eval_residual
    known_log_norm = logsumexp(ev.logits, axis=1)
    return virtual_logit - np.logaddexp(known_log_norm, virtual_logit)
