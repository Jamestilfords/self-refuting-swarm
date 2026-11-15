import numpy as np
from scipy.special import expit, logsumexp
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style virtual OOS logit is formed from representation energy outside a train-fitted principal subspace and compared directly against the classifier's in-scope logit evidence. Inputs with large unexplained residual representations receive high virtual-class probability even when one known-class logit is moderately large."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    tr_proj = pca.transform(tr.pooled)
    ev_proj = pca.transform(ev.pooled)
    tr_centered = tr.pooled - pca.mean_
    ev_centered = ev.pooled - pca.mean_

    tr_total = np.sum(tr_centered * tr_centered, axis=1)
    ev_total = np.sum(ev_centered * ev_centered, axis=1)
    tr_kept = np.sum(tr_proj * tr_proj, axis=1)
    ev_kept = np.sum(ev_proj * ev_proj, axis=1)

    tr_residual = np.sqrt(np.maximum(tr_total - tr_kept, 0.0))
    ev_residual = np.sqrt(np.maximum(ev_total - ev_kept, 0.0))

    scale = np.mean(np.abs(np.max(tr.logits, axis=1)))
    alpha = scale / (np.mean(tr_residual) + 1e-8)

    virtual_logit = alpha * ev_residual
    known_evidence = logsumexp(ev.logits, axis=1)
    scores = expit(virtual_logit - known_evidence)
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
