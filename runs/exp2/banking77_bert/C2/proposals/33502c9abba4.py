import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style virtual OOS logit is calibrated from training representations: points with large residual energy outside the in-scope principal subspace gain virtual-class probability. Unlike residual norm alone, this score also accounts for the item's competing classifier logits."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    train_logits = np.asarray(tr.logits, dtype=np.float64)
    eval_logits = np.asarray(ev.logits, dtype=np.float64)

    n_components = min(96, x_train.shape[1], max(1, x_train.shape[0] - 1))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=913)
    pca.fit(x_train)

    center = pca.mean_
    basis = pca.components_

    def residual_norm(x):
        centered = x - center
        projection = centered @ basis.T
        residual_sq = np.sum(centered * centered, axis=1) - np.sum(
            projection * projection, axis=1
        )
        return np.sqrt(np.maximum(residual_sq, 0.0))

    train_residual = residual_norm(x_train)
    eval_residual = residual_norm(x_eval)

    train_confidence = np.max(train_logits, axis=1)
    scale = np.mean(train_residual)
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0

    alpha = np.mean(np.maximum(train_confidence, 0.0)) / scale
    if not np.isfinite(alpha) or alpha < 1e-8:
        alpha = 1.0

    virtual_logit = alpha * eval_residual
    in_scope_log_norm = logsumexp(eval_logits, axis=1)
    score = virtual_logit - np.logaddexp(virtual_logit, in_scope_log_norm)
    return np.nan_to_num(score, nan=-1e6, posinf=0.0, neginf=-1e6)
