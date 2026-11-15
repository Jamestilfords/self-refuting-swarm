import numpy as np
from scipy.special import expit, logsumexp
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style virtual-class probability compares the residual component outside the train principal feature subspace against the classifier's known-class logit evidence. OOS queries should have a stronger residual-derived virtual logit while retaining sensitivity to their classifier confidence."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(n_components=128, svd_solver="randomized", random_state=314159)
    pca.fit(tr.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_
    components = pca.components_

    train_proj = train_centered @ components.T
    eval_proj = eval_centered @ components.T

    train_res_sq = np.sum(train_centered * train_centered, axis=1)
    train_res_sq -= np.sum(train_proj * train_proj, axis=1)
    eval_res_sq = np.sum(eval_centered * eval_centered, axis=1)
    eval_res_sq -= np.sum(eval_proj * eval_proj, axis=1)

    train_residual = np.sqrt(np.maximum(train_res_sq, 0.0))
    eval_residual = np.sqrt(np.maximum(eval_res_sq, 0.0))

    train_lse = logsumexp(tr.logits, axis=1)
    r50, r90 = np.quantile(train_residual, [0.50, 0.90])
    l50, l90 = np.quantile(train_lse, [0.50, 0.90])

    residual_span = max(float(r90 - r50), 1e-8)
    logit_span = max(float(l90 - l50), 1e-4)
    scale = logit_span / residual_span
    offset = l50 - scale * r50

    virtual_logit = offset + scale * eval_residual
    known_log_evidence = logsumexp(ev.logits, axis=1)

    return expit(np.clip(virtual_logit - known_log_evidence, -60.0, 60.0))
