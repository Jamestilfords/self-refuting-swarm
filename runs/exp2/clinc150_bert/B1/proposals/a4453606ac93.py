import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style virtual OOS logit is formed from the norm of the encoder component outside a train-fitted principal subspace. Normalizing its scale to the training logit scale lets residual-space novelty compete directly against the in-scope intent logits."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0082"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.pooled.shape[1] - 1, tr.pooled.shape[0] - 1)
    n_comp = max(1, n_comp)

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    tr_centered = tr.pooled - pca.mean_
    ev_centered = ev.pooled - pca.mean_

    tr_proj = pca.transform(tr.pooled)
    ev_proj = pca.transform(ev.pooled)

    tr_resid_sq = np.maximum(
        np.sum(tr_centered * tr_centered, axis=1) - np.sum(tr_proj * tr_proj, axis=1),
        0.0,
    )
    ev_resid_sq = np.maximum(
        np.sum(ev_centered * ev_centered, axis=1) - np.sum(ev_proj * ev_proj, axis=1),
        0.0,
    )

    tr_resid = np.sqrt(tr_resid_sq + 1e-12)
    ev_resid = np.sqrt(ev_resid_sq + 1e-12)

    tr_peak = np.max(tr.logits, axis=1)
    logit_scale = np.sqrt(np.mean(tr_peak * tr_peak) + 1e-12)
    residual_scale = np.mean(tr_resid) + 1e-12
    alpha = logit_scale / residual_scale

    virtual_logit = alpha * ev_resid
    in_scope_log_partition = logsumexp(ev.logits, axis=1)

    return virtual_logit - np.logaddexp(virtual_logit, in_scope_log_partition)
