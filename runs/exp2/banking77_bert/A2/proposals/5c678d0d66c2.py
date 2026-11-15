import numpy as np
from sklearn.decomposition import PCA
from scipy.special import expit, logsumexp

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures feature mass outside the dominant in-scope representation subspace. Treating this residual as a virtual OOS logit combines representation novelty with the classifier's native logit evidence."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(256, tr.pooled.shape[1] - 1, tr.pooled.shape[0] - 1)
    dim = max(1, dim)
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=17)
    pca.fit(tr.pooled)

    tr_proj = pca.inverse_transform(pca.transform(tr.pooled))
    ev_proj = pca.inverse_transform(pca.transform(ev.pooled))

    tr_residual = np.linalg.norm(tr.pooled - tr_proj, axis=1)
    ev_residual = np.linalg.norm(ev.pooled - ev_proj, axis=1)

    mean_residual = max(float(np.mean(tr_residual)), 1e-8)
    mean_max_logit = float(np.mean(np.max(tr.logits, axis=1)))
    alpha = max(mean_max_logit, 1e-3) / mean_residual

    virtual_logit = alpha * ev_residual
    known_log_partition = logsumexp(ev.logits, axis=1)
    return expit(virtual_logit - known_log_partition)
